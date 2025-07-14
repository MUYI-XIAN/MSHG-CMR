import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from models.model_utils import *  # 包含 SNN_Block、Attn_Net_Gated、BilinearFusion 等组件

###############################################
# 下面的两个函数用于实现 rotary embedding  #
###############################################
def rotate_every_two(x):
    """
    对输入张量最后一个维度，每隔两个元素分组，将其中一分量反转，再重排
    """
    # x[..., ::2] 取偶数下标；x[..., 1::2]取奇数下标
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # 交换 x2 的符号，并将 x1 与 -x2 交替堆叠
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    """
    应用旋转变换，将 x 进行 rotary embedding 操作
    x: 输入张量，最后一维要求为偶数
    sin, cos: 与 x 最后维度形状相同的旋转参数
    """
    return (x * cos) + (rotate_every_two(x) * sin)

###############################################
#             基础编码与融合组件              #
###############################################
class PositionalEncoding(nn.Module):
    """
    位置编码模块（Positional Encoding）
    """
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, d_model)
        return x + self.pe[:x.size(0), :]

class GatedFusion(nn.Module):
    """
    门控融合模块（Gated Fusion）
    """
    def __init__(self, d_model):
        super().__init__()
        self.w_g = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x1, x2):
        gate = torch.sigmoid(self.w_g(torch.cat([x1, x2], dim=-1)))
        return gate * x1 + (1 - gate) * x2

class FeatureFusionAttention(nn.Module):
    """
    融合注意力模块（Feature Fusion Attention）
    """
    def __init__(self, input_dim, hidden_dim):
        super(FeatureFusionAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, path_feat, omic_feat):
        combined = torch.cat([path_feat, omic_feat], dim=1)
        attn_weights = self.attn(combined)
        attn_weights = torch.sigmoid(attn_weights)
        fused_feat = attn_weights * path_feat + (1 - attn_weights) * omic_feat
        return fused_feat

###############################################
#         跨模态细化Transformer模块          #
###############################################
class CrossModalRefinementTransformer(nn.Module):
    """
    跨模态细化Transformer（Cross-Modal Refinement Transformer）
    
    在原有模块基础上，融入 RMT 思想，通过 rotary embedding 及基于 token 序号的衰减机制，
    对模态交互过程中计算注意力得分前对 x 与 y 的投影进行旋转变换，进而调节信息交互。
    
    参数：
        d_model: 隐层特征维度（要求为偶数，用于 rotary embedding）。
        num_stages: 细化阶段数。
    """
    def __init__(self, d_model, num_stages=2):
        super(CrossModalRefinementTransformer, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for rotary embedding.")
        self.num_stages = num_stages
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

        # 位置编码模块
        self.pos_encoder = PositionalEncoding(d_model)

        # 多阶段线性变换（模拟多头注意力中的投影操作）
        self.attention_layers = nn.ModuleList([ 
            nn.Linear(d_model, d_model) for _ in range(num_stages)
        ])

        # 每个阶段对应的层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_stages)
        ])

        # 每个阶段对应的门控融合模块
        self.gated_fusions = nn.ModuleList([
            GatedFusion(d_model) for _ in range(num_stages)
        ])

        # 前馈网络模块
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

        # 最终输出标准化
        self.final_layer_norm = nn.LayerNorm(d_model)

        # 每个阶段的残差连接层
        self.residual_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_stages)
        ])
        self.act_func = nn.ReLU()

        ###############################################
        # 以下为融入 RMT 思想的部分
        ###############################################
        # 是否使用 rotary embedding 与衰减机制
        self.use_retention = True
        # 衰减因子（可调超参数，防止得分过大）
        self.decay_factor = 0.1
        # retention_angle 用于计算每个 token 的旋转角度，形状 (d_model//2,)
        angle = 1.0 / (10000 ** (torch.linspace(0, 1, d_model // 2)))
        self.register_buffer('retention_angle', angle)

    def forward(self, x, y):
        """
        参数：
            x: 第一模态特征，形状 (batch_size, 1, D)
            y: 第二模态特征，形状 (num_omics, 1, D)
        返回：
            h_coattn_final: 融合后的第二模态细化特征，形状 (num_omics, 1, D)
            A_coattn: 最后一阶段的注意力权重，形状 (num_omics, batch_size)
        """
        batch_size, _, D = x.size()
        num_omics = y.size(0)

        # squeeze多余的维度，并添加位置编码
        x = x.squeeze(1)  # (batch_size, D)
        y = y.squeeze(1)  # (num_omics, D)
        x = self.pos_encoder(x.unsqueeze(0)).squeeze(0)
        y = self.pos_encoder(y.unsqueeze(0)).squeeze(0)

        h_coattn = y
        A_coattn = None

        for i in range(self.num_stages):
            # 线性投影
            x_transformed = self.attention_layers[i](x)
            y_transformed = self.attention_layers[i](h_coattn)
            # 层归一化
            x_transformed = self.layer_norms[i](x_transformed)
            y_transformed = self.layer_norms[i](y_transformed)

            ###############################################
            # 应用 rotary embedding（保留机制）：
            # 对于 x：由于每个样本仅有一个 token，视为位置 0（旋转不变）；
            # 对于 y：根据 token 序号计算旋转角度，然后构造 sin 与 cos 参数，
            # 并对 y_transformed 应用 theta_shift。
            ###############################################
            if self.use_retention:
                # 对 x_transformed，每个样本位置视为 0
                batch_positions = torch.zeros(batch_size, device=x_transformed.device).unsqueeze(1)  # (batch_size, 1)
                # retention_angle: (D//2,) -> 扩展后 (1, D//2)
                angles_x = batch_positions * self.retention_angle.unsqueeze(0)  # (batch_size, D//2)
                sin_x = torch.sin(angles_x)  # (batch_size, D//2)
                cos_x = torch.cos(angles_x)  # (batch_size, D//2)
                # 将每个 token的 sin 与 cos 交替重复，构造形状 (batch_size, D)
                sin_full_x = torch.stack([sin_x, sin_x], dim=-1).flatten(1)
                cos_full_x = torch.stack([cos_x, cos_x], dim=-1).flatten(1)
                x_transformed = theta_shift(x_transformed, sin_full_x, cos_full_x)

                # 对 y_transformed，根据 token 序号（0,1,...,num_omics-1）计算旋转参数
                positions = torch.arange(num_omics, device=y_transformed.device).float().unsqueeze(1)  # (num_omics, 1)
                angles_y = positions * self.retention_angle.unsqueeze(0)  # (num_omics, D//2)
                sin_y = torch.sin(angles_y)  # (num_omics, D//2)
                cos_y = torch.cos(angles_y)  # (num_omics, D//2)
                sin_full_y = torch.stack([sin_y, sin_y], dim=-1).flatten(1)  # (num_omics, D)
                cos_full_y = torch.stack([cos_y, cos_y], dim=-1).flatten(1)  # (num_omics, D)
                y_transformed = theta_shift(y_transformed, sin_full_y, cos_full_y)

            # 计算注意力得分
            scores = torch.matmul(y_transformed, x_transformed.T) * self.scale  # (num_omics, batch_size)

            ###############################################
            # 融入衰减机制：对 y 端 token，基于 token 序号施加衰减 bias
            ###############################################
            if self.use_retention:
                # positions: (num_omics, 1)
                decay_bias = positions * self.decay_factor  # (num_omics, 1)
                scores = scores - decay_bias  # 广播到每个 batch 样本

            # softmax归一化得到注意力权重
            A_coattn = F.softmax(scores, dim=-1)  # (num_omics, batch_size)
            # 计算上下文向量
            h_coattn_new = torch.matmul(A_coattn, x_transformed)  # (num_omics, D)
            # 门控融合
            h_coattn = self.gated_fusions[i](h_coattn_new, h_coattn)
            # 前馈网络及残差连接
            h_coattn = h_coattn + self.ffn(h_coattn)
            h_coattn_res = self.residual_layers[i](h_coattn)
            h_coattn = h_coattn + self.act_func(h_coattn_res)
            h_coattn = self.final_layer_norm(h_coattn)

        # 恢复维度 (num_omics, 1, D)
        h_coattn_final = h_coattn.unsqueeze(1)
        return h_coattn_final, A_coattn

###############################################
#       多模态生存分析Transformer模型         #
###############################################
class MSCAT_Surv(nn.Module):
    """
    多模态生存分析Transformer模型
    """
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.2, stage=1):
        super(MSCAT_Surv, self).__init__()
        self.fusion = 'attention'
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes

        # WSI 模型各层尺寸配置
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        # 基因组模型各层尺寸配置
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ###########################################################################
        #                           WSI特征提取分支                              #
        ###########################################################################
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.wsi_net = nn.Sequential(*fc)
        
        ###########################################################################
        #                           基因组特征提取分支                           #
        ###########################################################################
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ###########################################################################
        #                     跨模态细化Transformer 融合模块                      #
        ###########################################################################
        self.coattn = CrossModalRefinementTransformer(d_model=size[1], num_stages=stage)
        
        ###########################################################################
        #                    Path 分支 Transformer 与注意力头                    #
        ###########################################################################
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                        dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(
            nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)
        )
        
        ###########################################################################
        #                    Omic 分支 Transformer 与注意力头                    #
        ###########################################################################
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                        dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(
            nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)
        )
        
        ###########################################################################
        #                           融合策略设置                                  #
        ###########################################################################
        if self.fusion == 'attention':
            self.fusion_attention = FeatureFusionAttention(input_dim=512, hidden_dim=128)
        elif self.fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(256*2, size[2]),
                nn.ReLU(),
                nn.Linear(size[2], size[2]),
                nn.ReLU()
            )
            self.adjust_fusion = nn.Linear(size[2], 256*2)
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            self.adjust_fusion = nn.Linear(256, 256*2)
        else:
            self.mm = None
        
        ###########################################################################
        #                           分类器构建                                  #
        ###########################################################################
        if self.fusion == 'concat':
            classifier_input_dim = 256 * 2
        elif self.fusion == 'bilinear':
            classifier_input_dim = 256
        elif self.fusion == 'attention':
            classifier_input_dim = 256
        else:
            classifier_input_dim = size[2]
        self.classifier = nn.Linear(classifier_input_dim, self.n_classes)

    def forward(self, **kwargs):
        # --------------------- WSI 分支特征提取 ---------------------
        x_path = kwargs['x_path']
        h_path_bag = self.wsi_net(x_path).unsqueeze(1)
        
        # --------------------- 基因组分支特征提取 ---------------------
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        h_omic = [self.sig_networks[idx](sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)
        
        # --------------------- 跨模态细化Transformer 融合 ---------------------
        h_path_coattn, A_coattn = self.coattn(h_path_bag, h_omic_bag)
        
        #######################################################################
        #                        Path 分支处理流程                            #
        #######################################################################
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans_res = h_path_coattn + h_path_trans
        A_path, h_path = self.path_attention_head(h_path_trans_res.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).view(h_path.size(0), -1)
        
        #######################################################################
        #                        Omic 分支处理流程                            #
        #######################################################################
        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans_res = h_omic_bag + h_omic_trans
        A_omic, h_omic = self.omic_attention_head(h_omic_trans_res.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).view(h_omic.size(0), -1)
        
        #######################################################################
        #                        融合策略及分类器                             #
        #######################################################################
        if self.fusion == 'attention':
            h = self.fusion_attention(h_path, h_omic)
        elif self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h_concat = torch.cat([h_path, h_omic], dim=1)
            h_fusion = self.mm(h_concat)
            h_fusion = self.adjust_fusion(h_fusion)
            h = h_concat + h_fusion
        else:
            h = h_path

        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        return hazards, S, Y_hat, attention_scores
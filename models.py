# Author:Hibari
# 2025年11月18日17时33分01秒
# syh19990131@gmail.com
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):  # 初始化GAT层，创建可学习参数
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W矩阵 (F×F'): 特征变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a向量 (2F'×1): 注意力参数向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU: 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # # 步骤1: 线性变换 (N×F) → (N×F')
        N = h.size()[0]

        # 步骤2: 构造注意力输入 - 为所有节点对创建拼接特征
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                             h.repeat(N, 1)], dim=1).view(N, -1,
                                                          2 * self.out_features)  # # 结果: a_input[i,j] = [h_i || h_j]
        # 步骤3: 计算注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 步骤4: 应用邻接矩阵掩码（只保留邻居的注意力）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 步骤5: Softmax归一化
        attention = F.softmax(attention, dim=1)
        # 步骤6: Dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 步骤7: 加权聚合
        h_prime = torch.matmul(attention, h)
        return h_prime


class MultiHeadGAT(nn.Module):
    """多头图注意力"""

    def __init__(self, in_features, out_features, num_heads=8,
                 dropout=0.3):  # 初始化多头图注意力层，创建多个并行的图注意力头。in_features: 输入特征维度;out_features: 输出特征维度;num_heads: 注意力头的数量（默认8个）;dropout: Dropout比率（默认0.3）
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x, adj):
        """
        x: 输入节点特征矩阵，形状为 [N, in_features]，N为节点数
        adj: 邻接矩阵，形状为 [N, N]，表示图的连接关系
        """
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 并行计算+特征拼接
        return x


class EnhancedHypergraphConvolution(nn.Module):
    """超图卷积层"""

    def __init__(self, in_features, out_features, bias=True, dropout=0.3):  # 初始化超图卷积层的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        bias：是否使用偏置（默认True）
        dropout：Dropout比率（默认0.3）
        """
        super(EnhancedHypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 双路径线性变换层
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(in_features, out_features, bias=bias)

        # 批归一化层（稳定训练，加速收敛）
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):  # 执行超图卷积的前向传播
        """
        超图卷积

        参数:
            X: 节点特征矩阵 [N, in_features]
            H: 超图关联矩阵 [N, M]
        """
        # 计算度矩阵
        D_v = torch.sum(H, dim=1, keepdim=True).clamp(min=1)
        D_e = torch.sum(H, dim=0, keepdim=True).clamp(min=1)

        # Path 1: 标准超图卷积
        # 归一化超图关联矩阵
        H_norm = H / torch.sqrt(D_v)
        H_norm = H_norm / torch.sqrt(D_e)
        # 超图卷积操作
        X1 = self.linear1(X)
        X1 = H_norm @ (H_norm.T @ X1)
        X1 = self.bn1(X1)
        X1 = self.dropout(X1)

        # Path 2: 跳跃连接,保留原始特征信息，类似于ResNet中的残差连接
        X2 = self.linear2(X)  # 直接线性变换
        X2 = self.bn2(X2)  # 批归一化
        X2 = self.dropout(X2)  # Dropout

        return F.elu(X1 + X2)


class DualAttentionModule(nn.Module):
    """双重注意力模块"""

    def __init__(self, dim, num_heads=8, dropout=0.3):  # 初始化双重注意力模块的所有组件
        """
        dim：特征维度
        num_heads：多头注意力的头数（默认8）
        dropout：Dropout比率（默认0.3）
        """
        super(DualAttentionModule, self).__init__()

        self.spatial_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                       batch_first=True)  # dim:特征维度,num_head:注意力头数,dropout=dropout,batch_first=True  输入格式：[batch, seq, feature]

        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),  # 降维：压缩信息
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 正则化
            nn.Linear(dim // 4, dim),  # 升维：恢复维度
            nn.Sigmoid()  # 输出0-1权重
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)  # 空间注意力后的归一化
        self.norm2 = nn.LayerNorm(dim)  # 通道注意力后的归一化

    def forward(self, x):
        # x：输入特征矩阵，形状为 [N, dim]，其中N是节点数量
        # 空间注意力
        x_unsqueezed = x.unsqueeze(0)  # 增加batch维度：[N, dim] → [1, N, dim]
        attn_out, _ = self.spatial_attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)  # 多头自注意力
        x = self.norm1(x + attn_out.squeeze(0))  # 残差连接 + 层归一化

        # 通道注意力
        channel_weights = self.channel_attention(x.mean(dim=0, keepdim=True))  # 计算通道权重：对所有节点取平均，得到全局统计信息
        x = self.norm2(x * channel_weights)  # 应用通道权重 + 残差连接 + 层归一化

        return x


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    用于融合来自三个超图的特征
    """

    def __init__(self, dim, num_heads=4, dropout=0.3):
        super(CrossAttentionFusion, self).__init__()

        # 三个超图特征的查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # 门控机制,动态调整融合权重
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

    def forward(self, feat_all, feat_sno, feat_dis):
        """
        融合三个超图的特征
        feat_all: 来自统一超图的特征 [N, dim]
        feat_sno: 来自snoRNA超图的特征 [N, dim]
        feat_dis: 来自disease超图的特征 [N, dim]
        """
        # 堆叠特征用于交叉注意力
        stacked = torch.stack([feat_all, feat_sno, feat_dis], dim=1)  # [N, 3, dim]

        # 交叉注意力
        q = self.q_proj(feat_all).unsqueeze(1)  # [N, 1, dim]
        k = self.k_proj(stacked)  # [N, 3, dim]
        v = self.v_proj(stacked)  # [N, 3, dim]

        attn_out, _ = self.cross_attention(q, k, v)  # [N, 1, dim]
        attn_out = attn_out.squeeze(1)  # [N, dim]

        # 门控融合
        gate_input = torch.cat([feat_all, feat_sno, feat_dis], dim=1)  # [N, 3*dim]
        gate_weight = self.gate(gate_input)  # [N, dim]

        # 残差连接
        output = self.norm(feat_all + gate_weight * self.dropout(attn_out))

        return output


class TripleHypergraphBlock(nn.Module):
    """超图块"""

    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):  # 初始化高级超图块的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        num_heads：多头注意力的头数（默认8）
        dropout：Dropout比率（默认0.3）
        """
        super(TripleHypergraphBlock, self).__init__()

        # 三个超图各自的卷积层
        self.hgc_all = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout)
        self.hgc_sno = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout)
        self.hgc_dis = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout)

        # 交叉注意力融合
        self.fusion = CrossAttentionFusion(out_features, num_heads, dropout)

        # 双重注意力
        self.dual_attention = DualAttentionModule(out_features, num_heads, dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        # 残差投影
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, X, H_all, H_sno, H_dis):
        """
        X: 节点特征 [N, in_features]
        H_all: 统一超图
        H_sno: snoRNA超图
        H_dis: disease超图
        """
        identity = self.residual_proj(X)

        # 在三个超图上分别做卷积
        feat_all = self.hgc_all(X, H_all)
        feat_sno = self.hgc_sno(X, H_sno)
        feat_dis = self.hgc_dis(X, H_dis)

        # 交叉注意力融合
        X = self.fusion(feat_all, feat_sno, feat_dis)
        X = self.norm1(X + identity)

        # 双重注意力
        X_att = self.dual_attention(X)
        X = self.norm2(X + X_att)

        # 前馈网络
        X = X + self.ffn(X)

        return X


class FeatureEnhancementModule(nn.Module):
    """特征增强模块"""

    def __init__(self, in_features, out_features, dropout=0.3):  # 初始化特征增强模块的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        dropout：Dropout比率（默认0.3）
        """
        super(FeatureEnhancementModule, self).__init__()

        self.enhance = nn.Sequential(
            # 扩展
            nn.Linear(in_features, out_features * 2),  # 扩展到2倍
            nn.BatchNorm1d(out_features * 2),  # 批归一化
            nn.ELU(),  # ELU激活
            nn.Dropout(dropout),  # Dropout正则化

            # 压缩
            nn.Linear(out_features * 2, out_features),  # 压缩到目标维度
            nn.BatchNorm1d(out_features),  # 批归一化
            nn.ELU(),  # ELU激活
            nn.Dropout(dropout)  # Dropout正则化
        )

    def forward(self, x):  # 执行特征增强的前向传播
        return self.enhance(x)


class TripleHypergraphNN(nn.Module):
    """深度超图神经网络"""

    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim, assoc_matrix,
                 hidden_dims=[512, 384, 256, 128, 64], num_heads=8, dropout=0.2):  # 初始化深度超图神经网络的所有组件
        """
        num_snorna：snoRNA的数量
        num_disease：disease的数量
        snorna_sim：snoRNA的GIPK相似性矩阵
        disease_sim：disease的GIPK相似性矩阵
        assoc_matrix：当前折的训练关联矩阵，用于构造基于 adj_index 的节点初始特征
        hidden_dims：隐藏层维度列表（默认[512, 384, 256, 128, 64]）
        num_heads：注意力头数（默认8）
        dropout：Dropout比率（默认0.2）
        """
        super(TripleHypergraphNN, self).__init__()

        self.num_snorna = num_snorna
        self.num_disease = num_disease

        # ===== 初始特征: 关联矩阵 + 预先相似性 =====
        assoc_tensor = torch.as_tensor(assoc_matrix, dtype=torch.float32)

        # 预先相似性矩阵（已经在 DataLoader 里融合好了）
        snorna_sim_tensor = torch.as_tensor(snorna_sim, dtype=torch.float32)
        disease_sim_tensor = torch.as_tensor(disease_sim, dtype=torch.float32)

        # 形状安全检查（防止行列顺序出错）
        if snorna_sim_tensor.shape[0] != num_snorna or snorna_sim_tensor.shape[1] != num_snorna:
            raise ValueError(
                f"snoRNA 相似性矩阵形状为 {snorna_sim_tensor.shape}, "
                f"但期望为 ({num_snorna}, {num_snorna})"
            )
        if disease_sim_tensor.shape[0] != num_disease or disease_sim_tensor.shape[1] != num_disease:
            raise ValueError(
                f"disease 相似性矩阵形状为 {disease_sim_tensor.shape}, "
                f"但期望为 ({num_disease}, {num_disease})"
            )
        if assoc_tensor.shape != (num_snorna, num_disease):
            raise ValueError(
                f"关联矩阵形状为 {assoc_tensor.shape}, "
                f"但期望为 ({num_snorna}, {num_disease})"
            )

        # snoRNA 节点初始特征: [ adj_index 的每一行 | SNF 融合后的 sno 相似性行 ]
        #    维度: num_disease + num_snorna
        snorna_feat = torch.cat(
            [assoc_tensor, snorna_sim_tensor],
            dim=1
        )

        # disease 节点初始特征: [ adj_index 的每一列(转置) | disease 相似性行 ]
        #    维度: num_snorna + num_disease
        disease_feat = torch.cat(
            [assoc_tensor.t(), disease_sim_tensor],
            dim=1
        )

        # 可学习的特征嵌入
        self.snorna_features = nn.Parameter(snorna_feat, requires_grad=True)
        self.disease_features = nn.Parameter(disease_feat, requires_grad=True)

        snorna_in_dim = self.snorna_features.size(1)  # num_disease + num_snorna
        disease_in_dim = self.disease_features.size(1)  # num_snorna + num_disease

        # 可学习的特征嵌入
        self.snorna_features = nn.Parameter(snorna_feat, requires_grad=True)
        self.disease_features = nn.Parameter(disease_feat, requires_grad=True)

        snorna_in_dim = self.snorna_features.size(1)  # num_disease
        disease_in_dim = self.disease_features.size(1)  # num_snorna

        # 特征增强 - 将不同维度的特征映射到统一空间
        self.snorna_enhance = FeatureEnhancementModule(snorna_in_dim, hidden_dims[0], dropout)
        self.disease_enhance = FeatureEnhancementModule(disease_in_dim, hidden_dims[0], dropout)

        # 多尺度特征提取
        branch_dims = [hidden_dims[0] // 4, hidden_dims[0] // 4, hidden_dims[0] // 2]

        self.snorna_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(snorna_in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])

        self.disease_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(disease_in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])

        # 三超图卷积块
        self.hg_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hg_blocks.append(
                TripleHypergraphBlock(dims[i], dims[i + 1], num_heads, dropout)
            )

        # 全局池化注意力
        self.global_attention = DualAttentionModule(hidden_dims[-1], num_heads, dropout)

        # 预测头
        final_dim = hidden_dims[-1]
        self.predictor = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim * 2),
            nn.BatchNorm1d(final_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim * 2, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H_all, H_sno, H_dis, drop_edge_rate=0.0):
        """
        H_all: 统一超图 [N, M_all]
        H_sno: snoRNA超图 [N, M_sno]
        H_dis: disease超图 [N, M_dis]
        """
        # DropEdge 数据增强
        if self.training and drop_edge_rate > 0:
            H_all = self._drop_edge(H_all, drop_edge_rate)
            H_sno = self._drop_edge(H_sno, drop_edge_rate)
            H_dis = self._drop_edge(H_dis, drop_edge_rate)

        # 多尺度特征提取
        snorna_features_multi = [scale(self.snorna_features) for scale in self.snorna_multi_scale]
        disease_features_multi = [scale(self.disease_features) for scale in self.disease_multi_scale]

        snorna_feat = torch.cat(snorna_features_multi, dim=1)
        disease_feat = torch.cat(disease_features_multi, dim=1)

        # 特征增强与融合
        snorna_feat = snorna_feat + self.snorna_enhance(self.snorna_features)
        disease_feat = disease_feat + self.disease_enhance(self.disease_features)

        # 拼接所有节点特征
        X = torch.cat([snorna_feat, disease_feat], dim=0)

        # 通过三超图卷积块
        for hg_block in self.hg_blocks:
            X = hg_block(X, H_all, H_sno, H_dis)

        # 全局注意力
        X = self.global_attention(X)

        # 分离特征
        snorna_embed = X[:self.num_snorna]
        disease_embed = X[self.num_snorna:]

        # pair-wise 拼接 + predictor
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)

        combined = torch.cat([snorna_expanded, disease_expanded], dim=2)
        combined = combined.view(-1, combined.size(-1))

        scores = self.predictor(combined)
        scores = scores.view(self.num_snorna, self.num_disease)

        return scores

    def _drop_edge(self, H, rate):
        """DropEdge 数据增强"""
        mask = torch.rand_like(H) > rate
        return H * mask.float()

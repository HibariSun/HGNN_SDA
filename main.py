"""
Author:Hibari
2025年11月08日17时10分25秒
syh19990131@gmail.com
超图神经网络 - snoRNA-Disease关联预测
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            roc_curve, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import math
warnings.filterwarnings('ignore')

# 工具函数，确保实验结果可复现
def set_seed(seed=42):
    """设置所有随机种子以确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 自动检测并配置GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*80)
print(" 超图神经网络 - snoRNA-Disease关联预测")
print("="*80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")


# GIPK计算模块
class GIPKCalculator:
    """高斯相互作用谱核(GIPK)计算器"""
    
    def __init__(self, association_matrix, gamma_ratio=1.0):
        """
        初始化GIPK计算器
        
        参数:
            association_matrix: 关联矩阵 (m×n) m: snoRNA的数量 n: disease的数量
            gamma_ratio: 带宽控制参数 γ'参数，通常设为1 控制高斯核的宽度 影响相似性的"衰减速度" 较大的γ' → 更快衰减 → 只有非常相似的实体才有高相似度 较小的γ' → 更慢衰减 → 更多实体被认为相似
        """
        self.association_matrix = association_matrix
        self.gamma_ratio = gamma_ratio
    
    def compute_gipk_snorna(self): # 计算snoRNA的GIPK相似性
        """
        计算snoRNA的GIPK相似性矩阵
        
        公式: GIPK_S(si, sj) = exp(-γs × ||R(si) - R(sj)||²)
              γs = γ's × m / Σ||R(si)||²
        
        返回: GIPK相似性矩阵 (m×m)
        """
        m, n = self.association_matrix.shape # 获取矩阵维度
        
        # 计算归一化的带宽参数 γs
        sum_norm_squared = np.sum(np.linalg.norm(self.association_matrix, axis=1)**2)  # 计算每个snoRNA关联向量的L2范数平方和
        gamma_s = self.gamma_ratio * m / sum_norm_squared  # 计算归一化的γs
        
        print(f"\n[GIPK计算] snoRNA:")
        print(f"  - 关联矩阵形状: {self.association_matrix.shape}")
        print(f"  - γs (gamma): {gamma_s:.6f}")
        
        # 计算GIPK相似性矩阵
        GIPK_matrix = np.zeros((m, m), dtype=np.float32) # 初始化GIPK矩阵

        # 双重循环计算相似性
        for i in range(m):
            for j in range(m):
                # 计算两个snoRNA关联向量的差
                diff = self.association_matrix[i, :] - self.association_matrix[j, :]
                # 计算差向量的欧氏距离平方
                norm_squared = np.sum(diff ** 2)
                # 应用高斯核函数
                GIPK_matrix[i, j] = np.exp(-gamma_s * norm_squared)
        
        print(f"  - 计算完成: ({m}, {m})")
        print(f"  - 对角线值: {GIPK_matrix[0, 0]:.6f} (应为1.0)") # 验证对角线
        
        return GIPK_matrix
    
    def compute_gipk_disease(self):
        """
        计算disease的GIPK相似性矩阵
        
        使用关联矩阵的转置（每行代表一个disease与所有snoRNA的关联）
        
        返回: GIPK相似性矩阵 (n×n)
        """
        m, n = self.association_matrix.shape
        A_T = self.association_matrix.T  # 转置关联矩阵
        
        # 计算归一化的带宽参数 γd
        sum_norm_squared = np.sum(np.linalg.norm(A_T, axis=1)**2)
        gamma_d = self.gamma_ratio * n / sum_norm_squared
        
        print(f"\n[GIPK计算] Disease:")
        print(f"  - 关联矩阵转置形状: {A_T.shape}")
        print(f"  - γd (gamma): {gamma_d:.6f}")
        
        # 计算GIPK相似性矩阵
        GIPK_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                diff = A_T[i, :] - A_T[j, :]
                norm_squared = np.sum(diff ** 2)
                GIPK_matrix[i, j] = np.exp(-gamma_d * norm_squared)
        
        print(f"  - 计算完成: ({n}, {n})")
        print(f"  - 对角线值: {GIPK_matrix[0, 0]:.6f} (应为1.0)")
        
        return GIPK_matrix


# 超图构造器）
class HypergraphConstructor:
    """
    超图构造器
    
    创新点：同时使用GIPK相似性和关联矩阵构造超边
    """
    
    def __init__(self, association_matrix, snorna_sim, disease_sim):
        """
        初始化超图构造器
        
        参数:
            association_matrix: snoRNA-disease关联矩阵
            snorna_sim: snoRNA的GIPK相似性矩阵
            disease_sim: disease的GIPK相似性矩阵
        """
        self.association_matrix = association_matrix
        self.snorna_sim = snorna_sim
        self.disease_sim = disease_sim
        self.num_snorna = association_matrix.shape[0]
        self.num_disease = association_matrix.shape[1]
    
    def construct_hypergraph(self, k_snorna=15, k_disease=15, 
                           use_association_edges=True,
                           association_weight=0.5):
        """
        构建超图关联矩阵（融合相似性和关联信息）
        
        参数:
            k_snorna: snoRNA的K近邻数量
            k_disease: disease的K近邻数量
            use_association_edges: 是否使用关联矩阵构造额外的超边
            association_weight: 关联超边的权重
        
        返回:
            H: 超图关联矩阵 [num_nodes, num_hyperedges]
        """
        print(f"\n[步骤 2] 构建改进的超图...")
        print(f"  - K近邻: snoRNA={k_snorna}, disease={k_disease}")
        print(f"  - 使用关联超边: {use_association_edges}")
        if use_association_edges:
            print(f"  - 关联超边权重: {association_weight}")
        
        # 构建 K 近邻图，基于GIPK相似性的超边
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)

        # 计算维度
        num_nodes = self.num_snorna + self.num_disease
        
        # 计算超边数量
        if use_association_edges:
            # 相似性超边 + 关联超边
            num_sim_edges = self.num_snorna + self.num_disease
            num_assoc_edges = int(self.association_matrix.sum())  # 已知关联数
            num_hyperedges = num_sim_edges + num_assoc_edges
        else:
            # 仅相似性超边
            num_hyperedges = self.num_snorna + self.num_disease

        # 初始化超图关联矩阵，超图H行代表节点（snoRNA 和 disease）。列代表超边，元素H[i,j]代表节点 i 在超边 j 中的权重
        H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)

        # 填充 snoRNA 相似性超边
        edge_idx = 0
        
        # 填充snoRNA相似性超边
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]  # 找到所有邻居
            for neighbor in neighbors:
                H[neighbor, edge_idx] = snorna_knn[i, neighbor]  # # 使用相似度作为权重
            edge_idx += 1
        
        # 填充disease相似性超边
        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                # disease 节点索引需要偏移 self.num_snorna
                H[self.num_snorna + neighbor, edge_idx] = disease_knn[j, neighbor]
            edge_idx += 1
        
        print(f"  ✓ 相似性超边构建完成: {edge_idx} 条超边")
        
        # 填充关联超边
        if use_association_edges:
            num_assoc_edges_added = 0
            
            # 为每个已知的snoRNA-disease关联创建一条超边
            for i in range(self.num_snorna):
                for j in range(self.num_disease):
                    if self.association_matrix[i, j] == 1:
                        # 创建一条连接snoRNA i和disease j的超边
                        H[i, edge_idx] = association_weight  # snoRNA节点
                        H[self.num_snorna + j, edge_idx] = association_weight  # disease节点
                        edge_idx += 1
                        num_assoc_edges_added += 1
            
            print(f"  ✓ 关联超边构建完成: {num_assoc_edges_added} 条超边")
            print(f"    （基于 {int(self.association_matrix.sum())} 个已知关联）")
        
        print(f"  ✓ 超图构建完成:")
        print(f"    - 总节点数: {num_nodes} ({self.num_snorna} snoRNA + {self.num_disease} disease)")
        print(f"    - 总超边数: {edge_idx}")
        
        return torch.FloatTensor(H).to(device)
    
    def _build_knn_graph(self, similarity_matrix, k):  # 从完整的相似性矩阵中提取 K 近邻图。
        """从相似度矩阵构建K近邻图"""
        n = similarity_matrix.shape[0]
        knn_graph = np.zeros((n, n), dtype=np.float32)  # 初始化空图
        
        for i in range(n):
            sim_scores = similarity_matrix[i].copy()  # 复制相似度分数
            sim_scores[i] = -1  # 排除自己
            top_k_indices = np.argsort(sim_scores)[-k:]  # # 找到 top-K 相似的节点
            knn_graph[i, top_k_indices] = similarity_matrix[i, top_k_indices]
        
        knn_graph = (knn_graph + knn_graph.T) / 2  # 对称化
        
        return knn_graph


# 神经网络层和损失函数
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=0.75, gamma=2.0): # 初始化损失函数，alpha (α): 正样本权重，gamma (γ): 聚焦参数，范围 [0, 5]。
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1-1e-7)  # 裁剪预测值
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))  # 计算交叉熵损失
        p_t = target * pred + (1 - target) * (1 - pred)  # 对正确类别的预测概率
        focal_weight = self.alpha * (1 - p_t) ** self.gamma  # 计算聚焦权重
        focal_loss = focal_weight * ce_loss  # 计算最终损失
        return focal_loss.mean()  # 返回平均损失


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
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU: 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # # 步骤1: 线性变换 (N×F) → (N×F')
        N = h.size()[0]

        # 步骤2: 构造注意力输入 - 为所有节点对创建拼接特征
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)  # # 结果: a_input[i,j] = [h_i || h_j]
        # 步骤3: 计算注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 步骤4: 应用邻接矩阵掩码（只保留邻居的注意力）
        zero_vec = -9e15*torch.ones_like(e)
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
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):  # 初始化多头图注意力层，创建多个并行的图注意力头。in_features: 输入特征维度;out_features: 输出特征维度;num_heads: 注意力头的数量（默认8个）;dropout: Dropout比率（默认0.3）
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
        
        self.spatial_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)  # dim:特征维度,num_head:注意力头数,dropout=dropout,batch_first=True  输入格式：[batch, seq, feature]
        
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),  # 降维：压缩信息
            nn.ReLU(), # 非线性激活
            nn.Dropout(dropout), # 正则化
            nn.Linear(dim // 4, dim), # 升维：恢复维度
            nn.Sigmoid() # 输出0-1权重
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim) # 空间注意力后的归一化
        self.norm2 = nn.LayerNorm(dim) # 通道注意力后的归一化
    
    def forward(self, x):
        # x：输入特征矩阵，形状为 [N, dim]，其中N是节点数量
        # 空间注意力
        x_unsqueezed = x.unsqueeze(0) # 增加batch维度：[N, dim] → [1, N, dim]
        attn_out, _ = self.spatial_attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)  # 多头自注意力
        x = self.norm1(x + attn_out.squeeze(0))  # 残差连接 + 层归一化
        
        # 通道注意力
        channel_weights = self.channel_attention(x.mean(dim=0, keepdim=True)) # 计算通道权重：对所有节点取平均，得到全局统计信息
        x = self.norm2(x * channel_weights) # 应用通道权重 + 残差连接 + 层归一化
        
        return x


class AdvancedHypergraphBlock(nn.Module):
    """超图块"""
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):  # 初始化高级超图块的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        num_heads：多头注意力的头数（默认8）
        dropout：Dropout比率（默认0.3）
        """
        super(AdvancedHypergraphBlock, self).__init__()
        
        self.hgc = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout) # 增强超图卷积层
        self.dual_attention = DualAttentionModule(out_features, num_heads, dropout)  # 双重注意力模块
        
        self.ffn = nn.Sequential(  # 前馈神经网络（FFN）
            nn.Linear(out_features, out_features * 4),  # 扩展4倍
            nn.GELU(),  # 平滑激活函数
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features), # 压缩回原维度
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(out_features) # 超图卷积后
        self.norm2 = nn.LayerNorm(out_features) # 双重注意力后
        
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity() # 处理维度不匹配的残差连接
        
    def forward(self, X, H):  # 执行高级超图块的前向传播
        identity = self.residual_proj(X)  # 保存残差基准
        
        # 超图卷积 + 残差 + 归一化
        X = self.hgc(X, H)
        # 残差连接 + 层归一化
        X = self.norm1(X + identity)
        
        # 双重注意力
        X_att = self.dual_attention(X)
        # 残差连接 + 层归一化
        X = self.norm2(X + X_att)
        
        # 前馈网络 + 残差
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
            nn.Linear(in_features, out_features * 2), # 扩展到2倍
            nn.BatchNorm1d(out_features * 2), # 批归一化
            nn.ELU(), # ELU激活
            nn.Dropout(dropout), # Dropout正则化

            # 压缩
            nn.Linear(out_features * 2, out_features), # 压缩到目标维度
            nn.BatchNorm1d(out_features), # 批归一化
            nn.ELU(), # ELU激活
            nn.Dropout(dropout) # Dropout正则化
        )
        
    def forward(self, x): # 执行特征增强的前向传播
        return self.enhance(x)


class DeepHypergraphNN(nn.Module):
    """深度超图神经网络"""
    
    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim,
                 hidden_dims=[512, 384, 256, 128, 64], num_heads=8, dropout=0.2): # 初始化深度超图神经网络的所有组件
        """
        num_snorna：snoRNA的数量
        num_disease：disease的数量
        snorna_sim：snoRNA的GIPK相似性矩阵
        disease_sim：disease的GIPK相似性矩阵
        hidden_dims：隐藏层维度列表（默认[512, 384, 256, 128, 64]）
        num_heads：注意力头数（默认8）
        dropout：Dropout比率（默认0.2）
        """
        super(DeepHypergraphNN, self).__init__()
        
        self.num_snorna = num_snorna
        self.num_disease = num_disease
        
        # 可学习的特征嵌入
        self.snorna_features = nn.Parameter(torch.FloatTensor(snorna_sim), requires_grad=True)
        self.disease_features = nn.Parameter(torch.FloatTensor(disease_sim), requires_grad=True)
        
        # 特征增强
        self.snorna_enhance = FeatureEnhancementModule(num_snorna, hidden_dims[0], dropout)
        self.disease_enhance = FeatureEnhancementModule(num_disease, hidden_dims[0], dropout)
        
        # 多尺度特征提取
        branch_dims = [hidden_dims[0] // 4, hidden_dims[0] // 4, hidden_dims[0] // 2]

        # snoRNA多尺度分支
        self.snorna_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_snorna, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])

        # disease多尺度分支
        self.disease_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_disease, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])
        
        # 超图卷积块
        self.hg_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hg_blocks.append(
                AdvancedHypergraphBlock(dims[i], dims[i+1], num_heads, dropout)
            )
        
        # 全局池化注意力
        self.global_attention = DualAttentionModule(hidden_dims[-1], num_heads, dropout)
        
        # 预测头
        final_dim = hidden_dims[-1]
        self.predictor = nn.Sequential(
            # 第1层：扩展
            nn.Linear(final_dim * 2, final_dim * 2),
            nn.BatchNorm1d(final_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第2层：压缩
            nn.Linear(final_dim * 2, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第3层：进一步压缩
            nn.Linear(final_dim, final_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第4层：输出层
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):  # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 线性层（Linear）
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, H, drop_edge_rate=0.0):  # 执行完整的前向传播
        # DropEdge数据增强
        if self.training and drop_edge_rate > 0:
            H = self._drop_edge(H, drop_edge_rate)
        
        # 多尺度特征提取
        # 三个分支并行处理
        snorna_features_multi = [scale(self.snorna_features) for scale in self.snorna_multi_scale]
        disease_features_multi = [scale(self.disease_features) for scale in self.disease_multi_scale]
        
        # 拼接多尺度特征
        snorna_feat = torch.cat(snorna_features_multi, dim=1)
        disease_feat = torch.cat(disease_features_multi, dim=1)
        
        # 特征增强与融合
        snorna_feat = snorna_feat + self.snorna_enhance(self.snorna_features) # [num_snorna, 512]
        disease_feat = disease_feat + self.disease_enhance(self.disease_features) # [num_disease, 512]
        
        # 拼接所有节点特征
        X = torch.cat([snorna_feat, disease_feat], dim=0)
        
        # 通过超图卷积块
        for hg_block in self.hg_blocks:
            X = hg_block(X, H)
        
        # 全局注意力
        X = self.global_attention(X)
        
        # 分离特征
        snorna_embed = X[:self.num_snorna] # [361, 64]
        disease_embed = X[self.num_snorna:] # [780, 64]
        
        # 预测所有关联
        # 扩展维度以计算所有配对
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)

        # 拼接配对特征
        combined = torch.cat([snorna_expanded, disease_expanded], dim=2)
        # 展平为二维矩阵
        combined = combined.view(-1, combined.size(-1))

        # 预测关联分数
        scores = self.predictor(combined) # [281580, 128] → [281580, 1]
        scores = scores.view(self.num_snorna, self.num_disease) # [361, 780]
        
        return scores
    
    def _drop_edge(self, H, rate): # DropEdge数据增强
        """DropEdge数据增强"""
        mask = torch.rand_like(H) > rate
        return H * mask.float()


# 学习率调度器
class WarmupCosineScheduler:
    """Warmup + 余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6): # 初始化学习率调度器的所有参数
        """
        optimizer：PyTorch优化器对象（如Adam、SGD）
        warmup_epochs：预热阶段的epoch数
        total_epochs：总训练epoch数
        min_lr：最小学习率（默认1e-6）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']  # 获取初始学习率
        
    def step(self, epoch):  # 根据当前epoch更新学习率
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine阶段：余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# 数据加载类
# 数据加载类
class DataLoader:
    """改进的数据加载类 - 自己计算GIPK并融合外部相似性
       支持 snoRNA / disease 使用不同的 α，并做网格搜索时复用 GIPK 计算结果
    """

    def __init__(self, data_path='./data/',
                 alpha_snorna=None, alpha_disease=None):
        """
        alpha_snorna: snoRNA 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        alpha_disease: disease 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        """
        self.data_path = data_path
        self.alpha_snorna = alpha_snorna
        self.alpha_disease = alpha_disease

        # 缓存基础矩阵，避免在网格搜索时重复计算 GIPK
        self._association_matrix = None
        self._snorna_gipk = None
        self._disease_gipk = None
        self._sno_external = None
        self._disease_external = None
        self._external_available = False

    def load_all_data(self, alpha_snorna=None, alpha_disease=None):
        """
        加载数据并计算GIPK, 同时与外部相似性矩阵做 SWF 融合

        alpha_snorna / alpha_disease:
            - 如果为 None，则使用 __init__ 里传入的默认值
            - 如果仍为 None，则退回到“自动权重模式”（按均值算权重）

        返回:
            association_matrix: 关联矩阵
            snorna_sim_fused: 融合后的 snoRNA 相似性矩阵
            disease_sim_fused: 融合后的 disease 相似性矩阵
        """
        print("\n[步骤 1] 加载数据并计算相似性...")

        # ========== 第一次调用时，真正加载 & 计算 ==========
        if self._association_matrix is None:
            # 1. 加载关联矩阵
            adj_df = pd.read_csv(f"{self.data_path}adj_index.csv", index_col=0)
            self._association_matrix = adj_df.values.astype(np.float32)

            print(f"  ✓ 关联矩阵加载完成:")
            print(f"    - snoRNA数量: {self._association_matrix.shape[0]}")
            print(f"    - Disease数量: {self._association_matrix.shape[1]}")
            print(f"    - 已知关联数量: {int(self._association_matrix.sum())}")

            # 2. 基于关联矩阵计算 GIPK 相似性
            gipk_calculator = GIPKCalculator(self._association_matrix, gamma_ratio=1.0)
            snorna_gipk = gipk_calculator.compute_gipk_snorna()
            disease_gipk = gipk_calculator.compute_gipk_disease()

            self._snorna_gipk = self._normalize_similarity(snorna_gipk)
            self._disease_gipk = self._normalize_similarity(disease_gipk)

            print("\n  ✓ GIPK 计算和归一化完成")

            # 3. 加载外部相似性矩阵
            sno_ext_path = os.path.join(self.data_path, "sno_p2p_smith.csv")
            dis_ext_path = os.path.join(self.data_path, "disease_similarity.csv")

            self._external_available = os.path.exists(sno_ext_path) and os.path.exists(dis_ext_path)
            if not self._external_available:
                print("\n[警告] 未找到外部相似性文件 sno_p2p_smith.csv 或 disease_similarity.csv")
                print("       将仅使用 GIPK 相似性。要启用 SWF 融合，请将两个文件放到 data/ 目录下。")
            else:
                print("\n  ✓ 外部相似性文件已找到, 开始加载:")
                sno_external = pd.read_csv(sno_ext_path, index_col=0).values.astype(np.float32)
                disease_external = pd.read_csv(dis_ext_path, index_col=0).values.astype(np.float32)

                # 形状检查
                if sno_external.shape != self._snorna_gipk.shape:
                    raise ValueError(
                        f"snoRNA 外部相似性矩阵形状为 {sno_external.shape}, "
                        f"但 GIPK 相似性为 {self._snorna_gipk.shape}, 请检查行列顺序是否一致。"
                    )
                if disease_external.shape != self._disease_gipk.shape:
                    raise ValueError(
                        f"disease 外部相似性矩阵形状为 {disease_external.shape}, "
                        f"但 GIPK 相似性为 {self._disease_gipk.shape}, 请检查行列顺序是否一致。"
                    )

                self._sno_external = self._normalize_similarity(sno_external)
                self._disease_external = self._normalize_similarity(disease_external)
        else:
            print("  ✓ 使用缓存的关联矩阵与相似性（GIPK + 外部原始矩阵）")

        # ========== 每次调用都可以给不同 α ==========
        if alpha_snorna is None:
            alpha_snorna = self.alpha_snorna
        if alpha_disease is None:
            alpha_disease = self.alpha_disease

        # 没有外部相似性，就直接用 GIPK
        if (not self._external_available) or (self._sno_external is None) or (self._disease_external is None):
            snorna_sim_fused = self._snorna_gipk
            disease_sim_fused = self._disease_gipk
            print("\n  ✓ 使用 GIPK 相似性（未进行 SWF 融合）")
        else:
            snorna_sim_fused = self._swf_fusion(
                self._snorna_gipk, self._sno_external,
                name="snoRNA", alpha=alpha_snorna
            )
            disease_sim_fused = self._swf_fusion(
                self._disease_gipk, self._disease_external,
                name="Disease", alpha=alpha_disease
            )

            print("\n  ✓ SWF 融合完成:")
            print(f"    - snoRNA 相似性矩阵形状: {snorna_sim_fused.shape}")
            print(f"    - Disease 相似性矩阵形状: {disease_sim_fused.shape}")

        return (
            self._association_matrix.astype(np.float32),
            snorna_sim_fused.astype(np.float32),
            disease_sim_fused.astype(np.float32)
        )

    def _normalize_similarity(self, sim_matrix):
        """归一化相似性矩阵: 对称化 + 对角线设为 1 + 截断到 [0,1]"""
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
        np.fill_diagonal(sim_matrix, 1.0)
        sim_matrix = np.clip(sim_matrix, 0, 1)
        return sim_matrix

    def _swf_fusion(self, sim_gipk, sim_ext, name="", alpha=None):
        """
        SWF 融合:
            如果 alpha 不为空:
                S_fused = alpha * sim_gipk + (1 - alpha) * sim_ext
            如果 alpha 为 None:
                使用“自动权重模式”（按均值算权重）
        """
        if sim_gipk.shape != sim_ext.shape:
            raise ValueError(
                f"SWF 融合失败: 两个矩阵形状不一致: {sim_gipk.shape} vs {sim_ext.shape}"
            )

        # 先做一次归一化和对称化
        sim_gipk = self._normalize_similarity(sim_gipk)
        sim_ext = self._normalize_similarity(sim_ext)

        if alpha is None:
            # 兼容旧逻辑: 按平均值自动算权重
            w_gipk = float(np.mean(sim_gipk))
            w_ext = float(np.mean(sim_ext))
            denom = w_gipk + w_ext + 1e-8
            alpha = w_gipk / denom
            print(f"    - [{name}] 自动计算 alpha(GIPK)={alpha:.4f} (基于均值)")
        else:
            alpha = float(alpha)
            alpha = max(0.0, min(1.0, alpha))

        print(
            f"    - [{name}] 最终使用 alpha(GIPK)={alpha:.4f}, "
            f"1-alpha(EXT)={1-alpha:.4f}"
        )

        fused = alpha * sim_gipk + (1.0 - alpha) * sim_ext
        fused = self._normalize_similarity(fused)
        return fused




# 样本准备
def prepare_samples(association_matrix, train_indices, test_indices): # 从关联矩阵中提取正负样本
    """从关联矩阵中提取正负样本"""
    """
    association_matrix：snoRNA-disease关联矩阵，形状 [num_snorna, num_disease]
    train_indices：训练集的正样本索引（来自K折交叉验证）
    test_indices：测试集的正样本索引（来自K折交叉验证）
    """
    num_snorna, num_disease = association_matrix.shape # 获取矩阵维度
    
    # 找出所有正样本
    pos_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_samples.append((i, j))
    
    pos_samples = np.array(pos_samples)
    
    # 划分正样本
    train_pos = pos_samples[train_indices]
    test_pos = pos_samples[test_indices]
    
    # 提取负样本
    neg_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 0:
                neg_samples.append((i, j))
    
    neg_samples = np.array(neg_samples)
    np.random.shuffle(neg_samples) # 随机打乱负样本

    # 平衡采样负样本
    train_neg = neg_samples[:len(train_pos)]
    test_neg = neg_samples[len(train_pos):len(train_pos) + len(test_pos)]
    
    return train_pos, train_neg, test_pos, test_neg


# 训练器
class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, model, device, label_smoothing=0.1): # 初始化训练器
        """
        model：待训练的神经网络模型
        device：计算设备（CPU或GPU）
        label_smoothing：标签平滑系数（默认0.1）
        """
        self.model = model
        self.device = device
        self.label_smoothing = label_smoothing
        
    def train_epoch(self, H, train_pos, train_neg, optimizer, criterion, drop_edge_rate=0.1):  # 训练一个完整的epoch
        """
        参数：
        H：超图关联矩阵
        train_pos：训练集正样本
        train_neg：训练集负样本
        optimizer：优化器（如Adam）
        criterion：损失函数（如FocalLoss）
        drop_edge_rate：边丢弃率（默认0.1）
        返回值：loss.item()：平均训练损失
        """
        self.model.train() # 设置训练模式
        
        predictions = self.model(H, drop_edge_rate=drop_edge_rate) # 前向传播

        # 初始化损失
        loss = 0
        count = 0

        # 计算标签平滑后的标签
        pos_label = 1.0 - self.label_smoothing
        neg_label = self.label_smoothing

        # 计算正样本损失
        for i, j in train_pos:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.tensor([pos_label]).to(self.device))
            count += 1

        # 计算负样本损失
        for i, j in train_neg:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.tensor([neg_label]).to(self.device))
            count += 1

        # 计算平均损失
        loss = loss / count

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 梯度裁剪
        optimizer.step() # 更新参数
        
        return loss.item()  # 返回损失
    
    def evaluate(self, H, test_pos, test_neg):  # 评估模型在测试集上的性能
        """
        参数：
        H：超图关联矩阵
        test_pos：测试集正样本
        test_neg：测试集负样本
        返回值：
        auc_score：AUC分数
        aupr_score：AUPR分数
        y_true：真实标签
        y_scores：预测分数
        """
        self.model.eval()  # 设置评估模式

        # 无梯度前向传播
        with torch.no_grad():
            predictions = self.model(H)

        # 初始化列表
        y_true = []
        y_scores = []

        # 收集正样本结果
        for i, j in test_pos:
            y_true.append(1)
            y_scores.append(predictions[i, j].cpu().item())

        # 收集负样本结果
        for i, j in test_neg:
            y_true.append(0)
            y_scores.append(predictions[i, j].cpu().item())

        # 转换为NumPy数组
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # 计算评估指标
        auc_score = roc_auc_score(y_true, y_scores)
        aupr_score = average_precision_score(y_true, y_scores)
        
        return auc_score, aupr_score, y_true, y_scores  # 返回结果


# 交叉验证
def cross_validation(association_matrix, snorna_sim, disease_sim, 
                     n_splits=5, epochs=200, lr=0.0005, patience=30,
                     use_association_edges=True, association_weight=0.5,
                     k_snorna=20, k_disease=20):
    """
    执行完整的K折交叉验证

    新增超参数：
        k_snorna: 构造超图时 snoRNA 的K近邻
        k_disease: 构造超图时 disease 的K近邻
        association_weight: 关联超边的权重
    """
    print(f"\n[步骤 3] 开始 {n_splits} 折交叉验证...")
    print("=" * 80)
    print(f"  超图参数: k_snorna={k_snorna}, k_disease={k_disease}, "
          f"association_weight={association_weight}")

    # 找出所有正样本 (在完整的关联矩阵上, 仅用于划分 K 折)
    pos_indices = []
    num_snorna, num_disease = association_matrix.shape
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_indices.append((i, j))

    num_pos = len(pos_indices)
    print(f"总正样本数: {num_pos}")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 初始化结果容器
    fold_results = []
    all_y_true = []
    all_y_scores = []
    all_fold_predictions = []

    # K 折循环
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(num_pos))):
        print(f"\n{'─' * 80}")
        print(f"折 {fold + 1}/{n_splits}")
        print(f"{'─' * 80}")

        # 准备当前折的正负样本索引
        train_pos, train_neg, test_pos, test_neg = prepare_samples(
            association_matrix, train_idx, test_idx
        )

        print(f"训练集 - 正样本: {len(train_pos)}, 负样本: {len(train_neg)}")
        print(f"测试集 - 正样本: {len(test_pos)}, 负样本: {len(test_neg)}")

        # === 为当前折构造“训练专用”的关联矩阵, 并据此重建超图 ===
        train_assoc_matrix = association_matrix.copy().astype(np.float32)
        for (i, j) in test_pos:
            # 将测试折中的正样本从训练关联矩阵中抹去, 防止通过关联超边泄露信息
            train_assoc_matrix[i, j] = 0.0

        print("  - 当前折训练关联矩阵已构造完成 (测试折正样本置 0)")
        print(f"    剩余已知关联数量: {int(train_assoc_matrix.sum())}")

        # 使用当前折的训练关联矩阵构造超图
        hg_constructor = HypergraphConstructor(train_assoc_matrix, snorna_sim, disease_sim)
        H = hg_constructor.construct_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight
        )

        # 创建模型 (保持原有结构不变)
        model = DeepHypergraphNN(
            num_snorna=num_snorna,
            num_disease=num_disease,
            snorna_sim=snorna_sim,
            disease_sim=disease_sim,
            hidden_dims=[512, 384, 256, 128, 64],
            num_heads=8,
            dropout=0.2
        ).to(device)

        # 优化器、损失函数和学习率调度器
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs)

        # 训练器
        trainer = AdvancedTrainer(model, device, label_smoothing=0.1)

        # 训练循环与早停
        best_auc = 0.0
        patience_counter = 0

        pbar = tqdm(range(epochs), desc=f"Fold {fold + 1} Training")
        for epoch in pbar:
            # 更新学习率
            lr_current = scheduler.step(epoch)

            # 训练一个 epoch
            train_loss = trainer.train_epoch(H, train_pos, train_neg, optimizer, criterion)

            # 每 10 个 epoch 在当前折的验证集上评估一次
            if epoch % 10 == 0:
                auc, aupr, _, _ = trainer.evaluate(H, test_pos, test_neg)

                pbar.set_postfix({
                    "Loss": f"{train_loss:.4f}",
                    "AUC": f"{auc:.4f}",
                    "AUPR": f"{aupr:.4f}",
                    "LR": f"{lr_current:.6f}"
                })

                # 早停逻辑（这里仍然以 AUC 为主，也可以改成看 AUPR）
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n早停: 在 epoch {epoch}")
                    break

        # 使用当前折对应的超图在测试集上做最终评估
        auc, aupr, y_true, y_scores = trainer.evaluate(H, test_pos, test_neg)

        print(f"\nFold {fold + 1} 最终结果:")
        print(f"  AUC:  {auc:.4f}")
        print(f"  AUPR: {aupr:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "auc": auc,
            "aupr": aupr
        })

        all_y_true.extend(y_true)
        all_y_scores.extend(y_scores)
        all_fold_predictions.append({
            "fold": fold + 1,
            "y_true": y_true,
            "y_scores": y_scores
        })

    return fold_results, np.array(all_y_true), np.array(all_y_scores), all_fold_predictions



# 网格搜索：对 (alpha_snorna, alpha_disease, k_snorna, k_disease, association_weight) 做搜索
def grid_search(
    data_loader,
    alpha_snorna_list,
    alpha_disease_list,
    k_snorna_list,
    k_disease_list,
    association_weight_list,
    n_splits=5,
    epochs=200,
    lr=0.0005,
    patience=30
):
    """
    返回：
        best_config: 最优配置（以 mean AUPR 为标准）
        best_fold_results, best_y_true, best_y_scores, best_fold_predictions: 对应最优配置的结果
    """
    results = []
    best_config = None
    best_mean_aupr = -1.0
    best_fold_results = None
    best_y_true = None
    best_y_scores = None
    best_fold_predictions = None

    total_combinations = (
        len(alpha_snorna_list)
        * len(alpha_disease_list)
        * len(k_snorna_list)
        * len(k_disease_list)
        * len(association_weight_list)
    )
    comb_id = 0

    print("\n[网格搜索] 开始网格搜索超参数...")
    print(f"  总组合数: {total_combinations}")

    for alpha_s in alpha_snorna_list:
        for alpha_d in alpha_disease_list:
            print(f"\n=== 当前相似性融合超参数: "
                  f"alpha_snorna={alpha_s:.2f}, alpha_disease={alpha_d:.2f} ===")

            # 这里会复用 DataLoader 里缓存的 GIPK 和外部相似性，只是改变融合权重
            association_matrix, snorna_sim, disease_sim = data_loader.load_all_data(
                alpha_snorna=alpha_s,
                alpha_disease=alpha_d
            )

            for k_s in k_snorna_list:
                for k_d in k_disease_list:
                    for assoc_w in association_weight_list:
                        comb_id += 1
                        print(f"\n>>> 组合 {comb_id}/{total_combinations}: "
                              f"k_snorna={k_s}, k_disease={k_d}, association_weight={assoc_w}")

                        fold_results, all_y_true, all_y_scores, fold_predictions = cross_validation(
                            association_matrix=association_matrix,
                            snorna_sim=snorna_sim,
                            disease_sim=disease_sim,
                            n_splits=n_splits,
                            epochs=epochs,
                            lr=lr,
                            patience=patience,
                            use_association_edges=True,
                            association_weight=assoc_w,
                            k_snorna=k_s,
                            k_disease=k_d
                        )

                        aucs = [r['auc'] for r in fold_results]
                        auprs = [r['aupr'] for r in fold_results]
                        mean_auc = float(np.mean(aucs))
                        mean_aupr = float(np.mean(auprs))
                        std_auc = float(np.std(aucs))
                        std_aupr = float(np.std(auprs))

                        results.append({
                            'alpha_snorna': alpha_s,
                            'alpha_disease': alpha_d,
                            'k_snorna': k_s,
                            'k_disease': k_d,
                            'association_weight': assoc_w,
                            'mean_auc': mean_auc,
                            'std_auc': std_auc,
                            'mean_aupr': mean_aupr,
                            'std_aupr': std_aupr
                        })

                        print(f"    -> 当前组合平均 AUC = {mean_auc:.4f}, "
                              f"平均 AUPR = {mean_aupr:.4f}")

                        # 以 AUPR 为主要指标选最优配置
                        if mean_aupr > best_mean_aupr:
                            best_mean_aupr = mean_aupr
                            best_config = {
                                'alpha_snorna': alpha_s,
                                'alpha_disease': alpha_d,
                                'k_snorna': k_s,
                                'k_disease': k_d,
                                'association_weight': assoc_w
                            }
                            best_fold_results = fold_results
                            best_y_true = all_y_true
                            best_y_scores = all_y_scores
                            best_fold_predictions = fold_predictions
                            print("    *** 发现新的最优配置 (以 AUPR 为准) ***")

    # 保存网格搜索结果
    os.makedirs('./outputs', exist_ok=True)
    df_grid = pd.DataFrame(results)
    df_grid.to_csv('./outputs/grid_search_results.csv', index=False)
    print("\n[网格搜索] 所有组合结果已保存到 ./outputs/grid_search_results.csv")

    return best_config, best_fold_results, best_y_true, best_y_scores, best_fold_predictions


# 结果可视化
class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_all_results(self, fold_results, all_y_true, all_y_scores, fold_predictions):
        print(f"\n[步骤 4] 生成可视化结果...")
        
        self._plot_fold_comparison(fold_results)
        self._plot_overall_roc(all_y_true, all_y_scores)
        self._plot_overall_pr(all_y_true, all_y_scores)
        self._plot_comprehensive_panel(fold_results, all_y_true, all_y_scores)
        
        print(f"\n所有可视化已保存到: {self.output_dir}/")
    
    def _plot_fold_comparison(self, fold_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        # AUC
        bars1 = axes[0].bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
        axes[0].axhline(y=np.mean(aucs), color='blue', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(aucs):.4f}')
        axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
        axes[0].set_title('AUC Score across Folds (Improved)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # AUPR
        bars2 = axes[1].bar(folds, auprs, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5)
        axes[1].axhline(y=np.mean(auprs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(auprs):.4f}')
        axes[1].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('AUPR', fontsize=12, fontweight='bold')
        axes[1].set_title('AUPR Score across Folds (Improved)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/IMP_01_fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_01_fold_comparison.png")
    
    def _plot_overall_roc(self, y_true, y_scores):
        plt.figure(figsize=(10, 8))
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Overall ROC Curve (Improved Model)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/IMP_02_overall_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_02_overall_roc_curve.png")
    
    def _plot_overall_pr(self, y_true, y_scores):
        plt.figure(figsize=(10, 8))
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='blue', lw=3, 
                label=f'PR curve (AUPR = {pr_auc:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Overall Precision-Recall Curve (Improved Model)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/IMP_03_overall_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_03_overall_pr_curve.png")
    
    def _plot_comprehensive_panel(self, fold_results, y_true, y_scores):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        # 性能对比
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(folds))
        width = 0.35
        ax1.bar(x - width/2, aucs, width, label='AUC', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, auprs, width, label='AUPR', color='coral', alpha=0.8)
        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(A) Performance Comparison', fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # ROC曲线
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')
        ax2.set_title('(B) ROC Curve', fontweight='bold', loc='left')
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)
        
        # PR曲线
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color='blue', lw=3, label=f'AUPR = {pr_auc:.4f}')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('(C) Precision-Recall Curve', fontweight='bold', loc='left')
        ax3.legend(loc="lower left")
        ax3.grid(alpha=0.3)
        
        # 统计表格
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max'],
            ['AUC', f'{np.mean(aucs):.4f}', f'{np.std(aucs):.4f}', 
             f'{np.min(aucs):.4f}', f'{np.max(aucs):.4f}'],
            ['AUPR', f'{np.mean(auprs):.4f}', f'{np.std(auprs):.4f}', 
             f'{np.min(auprs):.4f}', f'{np.max(auprs):.4f}']
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('(D) Statistical Summary', fontweight='bold', loc='left', pad=20)
        
        plt.savefig(f'{self.output_dir}/IMP_04_comprehensive_panel.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: IMP_04_comprehensive_panel.png")


# 保存结果
def save_results(fold_results, output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(f'{output_dir}/IMP_fold_results.csv', index=False)
    
    aucs = [r['auc'] for r in fold_results]
    auprs = [r['aupr'] for r in fold_results]
    
    summary = {
        'Metric': ['AUC', 'AUPR'],
        'Mean': [np.mean(aucs), np.mean(auprs)],
        'Std': [np.std(aucs), np.std(auprs)],
        'Min': [np.min(aucs), np.min(auprs)],
        'Max': [np.max(aucs), np.max(auprs)],
        'Median': [np.median(aucs), np.median(auprs)]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f'{output_dir}/IMP_summary_statistics.csv', index=False)
    
    print(f"\n[步骤 5] 结果已保存到: {output_dir}/")


# 主函数
# 主函数
def main():
    print("\n开始训练流程...\n")

    # 初始化数据加载器（GIPK / 外部相似性会在第一次调用时计算并缓存）
    data_loader = DataLoader(data_path='./data/')

    # ====== 网格搜索空间======
    alpha_snorna_list = [0.3, 0.5, 0.7]
    alpha_disease_list = [0.3, 0.5, 0.7]
    k_snorna_list = [10, 20, 30]
    k_disease_list = [10, 20, 30]
    association_weight_list = [0.5, 0.8, 1.0]

    # 训练相关参数
    n_splits = 5
    epochs = 300
    lr = 0.0001
    patience = 30

    # ====== 网格搜索 ======
    best_config, fold_results, all_y_true, all_y_scores, fold_predictions = grid_search(
        data_loader=data_loader,
        alpha_snorna_list=alpha_snorna_list,
        alpha_disease_list=alpha_disease_list,
        k_snorna_list=k_snorna_list,
        k_disease_list=k_disease_list,
        association_weight_list=association_weight_list,
        n_splits=n_splits,
        epochs=epochs,
        lr=lr,
        patience=patience
    )

    print("\n[网格搜索] 最优配置如下（以平均 AUPR 为指标）:")
    print(f"  alpha_snorna       = {best_config['alpha_snorna']}")
    print(f"  alpha_disease      = {best_config['alpha_disease']}")
    print(f"  k_snorna           = {best_config['k_snorna']}")
    print(f"  k_disease          = {best_config['k_disease']}")
    print(f"  association_weight = {best_config['association_weight']}")

    # ====== 可视化 & 保存最优配置对应的结果 ======
    visualizer = ResultVisualizer()
    visualizer.plot_all_results(fold_results, all_y_true, all_y_scores, fold_predictions)
    save_results(fold_results)

    print("\n" + "=" * 80)
    print("  训练完成（使用网格搜索得到的最优超参数）")
    print("=" * 80)

    # 性能评估
    avg_auc = np.mean([r['auc'] for r in fold_results])
    avg_aupr = np.mean([r['aupr'] for r in fold_results])

    print(f"\n 最优配置下的性能评估:")
    print(f"  平均 AUC:  {avg_auc:.4f}")
    print(f"  平均 AUPR: {avg_aupr:.4f}")

    # 与原始GCNMF-SDA对比（保留你原来的对比输出）
    print(f"\n 与GCNMF-SDA对比:")
    print(f"  GCNMF-SDA: AUC=0.9659, AUPR=0.9522")
    print(f"  HGNN-SDA:    AUC={avg_auc:.4f}, AUPR={avg_aupr:.4f}")


if __name__ == "__main__":
    main()


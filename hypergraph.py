# Author:Hibari
# 2025年11月18日17时31分50秒
# syh19990131@gmail.com
# hypergraph.py
import numpy as np
import torch
from utils import device
from sklearn.cluster import KMeans


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

    def _build_kmeans_hypergraph(self,
                                 n_clusters=50,
                                 random_state=42,
                                 max_iter=300):
        """
        使用 KMeans 在「所有节点」上做聚类，构造全局超图：
          - 把所有 snoRNA 和 disease 当成统一的一批节点来聚类；
          - 每个簇形成一条超边；
          - 只保留同时包含 snoRNA 和 disease 的簇（保证一条超边里两类节点都出现）。

        特征设计（统一维度 = num_snorna + num_disease）：
          - 对 snoRNA i：
              前 num_snorna 维：snorna_sim[i, :]       （它和所有 sno 的相似度）
              后 num_disease 维：association_matrix[i, :] （它和所有 disease 的关联）
          - 对 disease j：
              前 num_snorna 维：association_matrix[:, j]  （和所有 sno 的关联）
              后 num_disease 维：disease_sim[j, :]        （和所有 disease 的相似度）

        这样 sno / disease 都在同一个特征空间里，自然可以被分到同一个簇。
        """
        num_nodes = self.num_snorna + self.num_disease
        feat_dim = self.num_snorna + self.num_disease

        # === 2.1 构造所有节点的特征矩阵 X ===
        X = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        A = self.association_matrix  # 当前折的训练关联矩阵（不会包含测试折的边）

        # snoRNA 节点：索引 0 ~ num_snorna-1
        for i in range(self.num_snorna):
            # 前半部分：和所有 snoRNA 的相似度
            X[i, :self.num_snorna] = self.snorna_sim[i]
            # 后半部分：和所有 disease 的关联
            X[i, self.num_snorna:] = A[i]

        # disease 节点：索引 num_snorna ~ num_snorna+num_disease-1
        for j in range(self.num_disease):
            node_idx = self.num_snorna + j
            # 前半部分：和所有 snoRNA 的关联（列向量）
            X[node_idx, :self.num_snorna] = A[:, j]
            # 后半部分：和所有 disease 的相似度
            X[node_idx, self.num_snorna:] = self.disease_sim[j]

        # === 2.2 在所有节点上做 KMeans 聚类 ===
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10
        )
        labels = kmeans.fit_predict(X)  # shape: [num_nodes]

        # === 2.3 把每个簇变成一条超边，只保留“混合簇”（既有 sno 也有 disease） ===
        edge_node_lists = []
        for c in range(n_clusters):
            nodes = np.where(labels == c)[0]
            if nodes.size < 2:  # 只过滤太小的簇
                continue
            edge_node_lists.append(nodes)

            has_sno = np.any(nodes < self.num_snorna)
            has_dis = np.any(nodes >= self.num_snorna)
            if not (has_sno and has_dis):
                # 如果这个簇里只有 sno 或只有 disease，就跳过，
                # 强行保证 H_kmeans 的每条超边都“跨两类节点”
                continue

            edge_node_lists.append(nodes)

        num_edges = len(edge_node_lists)
        if num_edges == 0:
            # 极端情况下（簇都被丢掉），返回一个空的超图矩阵
            return torch.zeros((num_nodes, 0), dtype=torch.float32, device=device)

        H = np.zeros((num_nodes, num_edges), dtype=np.float32)
        for e_idx, nodes in enumerate(edge_node_lists):
            H[nodes, e_idx] = 1.0   # 简单地把在该簇里的节点权重都设为 1

        return torch.FloatTensor(H).to(device)

    def _build_neighbor_hypergraph(self):
        """
        使用当前折的训练关联矩阵构造“邻域超图”：

          - 对每个有训练关联的 snoRNA i：
              建一条超边 e_i，包含 sno_i 本人 + 所有与其相连的 disease
          - 对每个有训练关联的 disease j：
              建一条超边 e'_j，包含 disease_j 本人 + 所有与其相连的 snoRNA
          - 对没有任何训练关联的节点，不单独创建超边。

        注意：self.association_matrix 是 train_assoc_matrix（K 折里传进来的），
              测试折的正样本已经被置 0，因此不会泄露测试集信息。
        """
        num_nodes = self.num_snorna + self.num_disease
        A = self.association_matrix
        edge_node_lists = []

        # 以 snoRNA 为中心的邻域超边
        for i in range(self.num_snorna):
            disease_indices = np.where(A[i] > 0)[0]
            if disease_indices.size == 0:
                continue
            nodes = [i] + [self.num_snorna + j for j in disease_indices]
            edge_node_lists.append(nodes)

        # 以 disease 为中心的邻域超边
        for j in range(self.num_disease):
            sno_indices = np.where(A[:, j] > 0)[0]
            if sno_indices.size == 0:
                continue
            nodes = [self.num_snorna + j] + list(sno_indices)
            edge_node_lists.append(nodes)

        num_edges = len(edge_node_lists)
        if num_edges == 0:
            return torch.zeros((num_nodes, 0), dtype=torch.float32, device=device)

        H = np.zeros((num_nodes, num_edges), dtype=np.float32)
        for e_idx, nodes in enumerate(edge_node_lists):
            H[nodes, e_idx] = 1.0

        return torch.FloatTensor(H).to(device)

    def construct_multi_hypergraph(self,
                                   k_snorna=15,
                                   k_disease=15,
                                   use_association_edges=True,
                                   association_weight=0.5,
                                   # ↓↓↓ 以下是 KMeans 独立的参数 ↓↓↓
                                   kmeans_clusters=50,
                                   kmeans_max_iter=300,
                                   kmeans_random_state=42):
        """
        构建三个超图：

          - H_all      : 原来的“相似性 + 关联”超图（KNN + 每条关联一条超边）
          - H_kmeans   : 全局 KMeans 聚类超图，每条超边连接 snoRNA 和 disease
          - H_neighbor : 邻域超图（每个节点的训练邻居集合）

        参数说明：
          k_snorna, k_disease          : 仍然只控制 KNN 部分（H_all 里的相似性超边）
          use_association_edges        : 是否在 H_all 中加入“每条关联一条超边”
          association_weight           : H_all 里关联超边的权重
          kmeans_clusters              : KMeans 聚类的簇数（= 超边数量上限）
          kmeans_max_iter              : KMeans 最大迭代次数
          kmeans_random_state          : KMeans 随机种子

        返回:
          H_all, H_kmeans, H_neighbor
          （为了兼容原来的 TripleHypergraphNN，这里仍然返回为 H_all, H_sno, H_dis，
           只是语义变成：H_all = KNN+关联, H_sno = KMeans, H_dis = 邻域）
        """

        # === 1) 原始的 KNN + 关联超图 ===
        H_all = self.construct_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight
        )

        # === 2) 全局 KMeans 超图（H_kmeans） ===
        H_kmeans = self._build_kmeans_hypergraph(
            n_clusters=kmeans_clusters,
            random_state=kmeans_random_state,
            max_iter=kmeans_max_iter
        )

        # === 3) 邻域超图（H_neighbor） ===
        H_neighbor = self._build_neighbor_hypergraph()

        print(f"  ✓ 多超图构建完成: "
              f"H_all(KNN+关联)={tuple(H_all.shape)}, "
              f"H_kmeans={tuple(H_kmeans.shape)}, "
              f"H_neighbor={tuple(H_neighbor.shape)}")

        # 为了不改动后面模型的接口，这里依然按照 H_all, H_sno, H_dis 的顺序返回
        return H_all, H_kmeans, H_neighbor


# Author:Hibari
# 2025年11月18日17时35分33秒
# syh19990131@gmail.com

# data_utils.py
import os
import numpy as np
import pandas as pd

from gipk import GIPKCalculator


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
        统一的数据加载接口

        返回:
            association_matrix: [num_snorna, num_disease] 的 0/1 关联矩阵
            snorna_sim_fused:   [num_snorna, num_snorna] 的 snoRNA 相似性
            disease_sim_fused:  [num_disease, num_disease] 的 disease 相似性

        当前设计:
            - snoRNA: 先把 sno_p2p_smith + sno_rnaernie_cosine_sim 做 SNF 融合，
                     得到一个 snoRNA 外部相似性矩阵 self._sno_external
            - disease: 直接使用 disease_similarity.csv（你已经融合了 DO + MeSH）

            - 然后用 SWF 做:
                snorna_sim_fused  = alpha_snorna * GIPK + (1 - alpha_snorna) * 外部(SNF)
                disease_sim_fused = alpha_disease * GIPK + (1 - alpha_disease) * 外部
              如果你把 alpha_snorna / alpha_disease 设为 0，就是「完全不用 GIPK」。
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

            # 2. 基于关联矩阵计算 GIPK（你现在如果不想用，只要后面 alpha=0 即可）
            gipk_calculator = GIPKCalculator(self._association_matrix, gamma_ratio=1.0)
            snorna_gipk = gipk_calculator.compute_gipk_snorna()
            disease_gipk = gipk_calculator.compute_gipk_disease()

            self._snorna_gipk = self._normalize_similarity(snorna_gipk)
            self._disease_gipk = self._normalize_similarity(disease_gipk)

            print("\n  ✓ GIPK 计算和归一化完成")

            # 3. 加载外部相似性矩阵：sno 两个 + disease 一个
            sno_smith_path = os.path.join(self.data_path, "sno_p2p_smith.csv")
            sno_rna_path = os.path.join(self.data_path, "sno_rnaernie_cosine_sim.csv")
            dis_ext_path = os.path.join(self.data_path, "disease_similarity.csv")

            self._external_available = (
                    os.path.exists(sno_smith_path)
                    and os.path.exists(sno_rna_path)
                    and os.path.exists(dis_ext_path)
            )

            if not self._external_available:
                print("\n[警告] 未找到以下一个或多个外部相似性文件：")
                print("       - sno_p2p_smith.csv")
                print("       - sno_rnaernie_cosine_sim.csv")
                print("       - disease_similarity.csv")
                print("       将仅使用 GIPK 相似性。")
            else:
                print("\n  ✓ 外部相似性文件已找到, 开始加载并做 SNF 融合:")

                sno_smith = pd.read_csv(sno_smith_path, index_col=0).values.astype(np.float32)
                sno_rna = pd.read_csv(sno_rna_path, index_col=0).values.astype(np.float32)
                disease_external = pd.read_csv(dis_ext_path, index_col=0).values.astype(np.float32)

                # 形状检查
                if sno_smith.shape != self._snorna_gipk.shape or sno_rna.shape != self._snorna_gipk.shape:
                    raise ValueError(
                        f"snoRNA 外部相似性矩阵形状为 {sno_smith.shape} / {sno_rna.shape}, "
                        f"但 GIPK 相似性为 {self._snorna_gipk.shape}, 请检查行列顺序是否一致。"
                    )
                if disease_external.shape != self._disease_gipk.shape:
                    raise ValueError(
                        f"disease 外部相似性矩阵形状为 {disease_external.shape}, "
                        f"但 GIPK 相似性为 {self._disease_gipk.shape}, 请检查行列顺序是否一致。"
                    )

                # 先做一次简单归一化
                sno_smith = self._normalize_similarity(sno_smith)
                sno_rna = self._normalize_similarity(sno_rna)
                disease_external = self._normalize_similarity(disease_external)

                # 用 SNF 融合两个 snoRNA 外部相似性
                self._sno_external = self._snf_two_similarities(
                    sno_smith, sno_rna, K=20, t=20, mu=0.5
                )
                # disease 外部相似性直接使用你融合好的 DO+MeSH
                self._disease_external = disease_external

                print("  ✓ SNF 融合完成: snoRNA 外部相似性 = SNF(smith, RNAErnie)")

        else:
            print("  ✓ 使用缓存的关联矩阵与相似性（GIPK + 外部矩阵）")

        # ========== 每次调用可以传入不同的 alpha ==========
        if alpha_snorna is None:
            alpha_snorna = self.alpha_snorna
        if alpha_disease is None:
            alpha_disease = self.alpha_disease

        # 当前你的需求：想完全不用 GIPK，就在 main.py / grid_search 里把 alpha 设为 0.0
        if (not self._external_available) or (self._sno_external is None) or (self._disease_external is None):
            snorna_sim_fused = self._snorna_gipk
            disease_sim_fused = self._disease_gipk
            print("\n  ✓ 使用 GIPK 相似性（未进行外部融合）")
        else:
            snorna_sim_fused = self._swf_fusion(
                self._snorna_gipk, self._sno_external,
                name="snoRNA", alpha=alpha_snorna
            )
            disease_sim_fused = self._swf_fusion(
                self._disease_gipk, self._disease_external,
                name="Disease", alpha=alpha_disease
            )

            print("\n  ✓ 相似性融合完成:")
            print(f"    - snoRNA 相似性矩阵形状: {snorna_sim_fused.shape}")
            print(f"    - Disease 相似性矩阵形状: {disease_sim_fused.shape}")

        return (
            self._association_matrix.astype(np.float32),
            snorna_sim_fused.astype(np.float32),
            disease_sim_fused.astype(np.float32)
        )

    def _snf_two_similarities(self, S1, S2, K=20, t=20, mu=0.5):
        """
        使用一个简化版的 SNF 算法，把两个相似性矩阵 S1 / S2 融合成一个矩阵。

        参数:
            S1, S2: 形状 [N, N] 的相似性矩阵（建议先做过 _normalize_similarity）
            K: 每个样本保留的近邻数
            t: 迭代次数
            mu: 对弱边的抑制程度 (0<mu<=1, 越小越强化大边)

        返回:
            fused: 形状 [N, N] 的融合后相似性矩阵，已归一化到 [0, 1]，对角线为 1
        """

        def _make_affinity(S):
            """从相似性 S 构造 KNN 的转移矩阵 P"""
            N = S.shape[0]
            # 去掉自环
            S = S - np.diag(np.diag(S))

            W = np.zeros_like(S, dtype=np.float32)
            K_eff = min(K, N - 1)
            for i in range(N):
                # 取每一行最大的 K 个近邻
                idx = np.argsort(S[i])[-K_eff:]
                W[i, idx] = S[i, idx]

            # 对称化
            W = (W + W.T) / 2.0

            # 变成转移矩阵 P
            row_sum = W.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            P = W / (2.0 * row_sum)
            np.fill_diagonal(P, 0.5)
            return P

        # 先构造两个网络的初始转移矩阵
        P1 = _make_affinity(S1)
        P2 = _make_affinity(S2)
        P_list = [P1, P2]

        # 迭代信息扩散
        for _ in range(t):
            new_list = []
            for i in range(2):
                other = P_list[1 - i]  # 只有两个网络时，另一个就是 "other"
                P_bar = other

                # 信息扩散
                P_new = P_list[i] @ P_bar @ P_list[i].T
                if mu != 1.0:
                    P_new = np.power(P_new, mu)

                # 重新构造 KNN + 归一化
                P_new = _make_affinity(P_new)
                new_list.append(P_new)

            P_list = new_list

        fused = (P_list[0] + P_list[1]) / 2.0
        fused = self._normalize_similarity(fused)
        return fused

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
            f"1-alpha(EXT)={1 - alpha:.4f}"
        )

        fused = alpha * sim_gipk + (1.0 - alpha) * sim_ext
        fused = self._normalize_similarity(fused)
        return fused


# 样本准备
def prepare_samples(association_matrix, train_indices, test_indices):  # 从关联矩阵中提取正负样本
    """从关联矩阵中提取正负样本"""
    """
    association_matrix：snoRNA-disease关联矩阵，形状 [num_snorna, num_disease]
    train_indices：训练集的正样本索引（来自K折交叉验证）
    test_indices：测试集的正样本索引（来自K折交叉验证）
    """
    num_snorna, num_disease = association_matrix.shape  # 获取矩阵维度

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
    np.random.shuffle(neg_samples)  # 随机打乱负样本

    # 平衡采样负样本
    train_neg = neg_samples[:len(train_pos)]
    test_neg = neg_samples[len(train_pos):len(train_pos) + len(test_pos)]

    return train_pos, train_neg, test_pos, test_neg

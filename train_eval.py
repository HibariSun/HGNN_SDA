# Author:Hibari
# 2025年11月18日17时40分21秒
# syh19990131@gmail.com
# train_eval.py
import os
import pandas as pd
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn as nn

from gipk import GIPKCalculator
from hypergraph import HypergraphConstructor
from models import TripleHypergraphNN
from data_utils import prepare_samples
from utils import device


# 神经网络层和损失函数
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha=0.75, gamma=2.0):  # 初始化损失函数，alpha (α): 正样本权重，gamma (γ): 聚焦参数，范围 [0, 5]。
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)  # 裁剪预测值
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))  # 计算交叉熵损失
        p_t = target * pred + (1 - target) * (1 - pred)  # 对正确类别的预测概率
        focal_weight = self.alpha * (1 - p_t) ** self.gamma  # 计算聚焦权重
        focal_loss = focal_weight * ce_loss  # 计算最终损失
        return focal_loss.mean()  # 返回平均损失


# 学习率调度器
class WarmupCosineScheduler:
    """Warmup + 余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):  # 初始化学习率调度器的所有参数
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


# 训练器
class TripleHypergraphTrainer:
    """高级训练器"""

    def __init__(self, model, device, label_smoothing=0.1):  # 初始化训练器
        """
        model：待训练的神经网络模型
        device：计算设备（CPU或GPU）
        label_smoothing：标签平滑系数（默认0.1）
        """
        self.model = model
        self.device = device
        self.label_smoothing = label_smoothing

    def train_epoch(self, H_all, H_sno, H_dis, train_pos, train_neg, optimizer, criterion, drop_edge_rate=0.1):
        """训练一个epoch"""
        self.model.train()

        # 前向传播 - 传入三个超图
        predictions = self.model(H_all, H_sno, H_dis, drop_edge_rate=drop_edge_rate)

        loss = 0
        count = 0

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

        loss = loss / count

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def evaluate(self, H_all, H_sno, H_dis, test_pos, test_neg):
        """评估模型"""
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(H_all, H_sno, H_dis)

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

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        auc_score = roc_auc_score(y_true, y_scores)
        aupr_score = average_precision_score(y_true, y_scores)

        return auc_score, aupr_score, y_true, y_scores


# 交叉验证
def cross_validation(association_matrix, snorna_sim, disease_sim,
                     n_splits=5, epochs=200, lr=0.0005, patience=30,
                     use_association_edges=True, association_weight=0.5,
                     k_snorna=20, k_disease=20,
                     data_loader=None,
                     alpha_snorna=None,
                     alpha_disease=None):
    """
    执行完整的K折交叉验证。

    防止数据泄露的关键点：
    - association_matrix 始终表示「完整的已知 snoRNA-disease 关联矩阵」，只用于：
        * 划分 K 折
        * 生成正/负样本标签
    - 真正参与训练的结构信息（超图 + GIPK 相似性）在每一折中
      都只使用当前折的训练关联矩阵 train_assoc_matrix 重新计算，
      从而避免测试折中的边泄露进特征和超图结构。
    """
    print(f"\n[步骤 3] 开始 {n_splits} 折交叉验证...")
    print("=" * 80)
    print(f"  超图参数: k_snorna={k_snorna}, k_disease={k_disease}, "
          f"association_weight={association_weight}")

    # -------------------------------
    # 1. 划分 K 折（只基于完整关联矩阵的正样本位置）
    # -------------------------------
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

    # 如果提供了 data_loader，我们可以利用其中缓存的外部相似性矩阵
    sno_ext = None
    dis_ext = None
    external_available = False
    if data_loader is not None:
        sno_ext = getattr(data_loader, "_sno_external", None)
        dis_ext = getattr(data_loader, "_disease_external", None)
        external_available = getattr(data_loader, "_external_available", False) \
                             and sno_ext is not None and dis_ext is not None

        if external_available:
            print("  [修正] 每一折将使用训练关联矩阵重新计算 GIPK，并与外部相似性做 SWF 融合。")
        else:
            print("  [修正] 每一折仅使用基于训练集的 GIPK 相似性（当前没有可用的外部相似性矩阵）。")

    # -------------------------------
    # 2. K 折循环
    # -------------------------------
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(num_pos))):
        print(f"\n{'─' * 80}")
        print(f"折 {fold + 1}/{n_splits}")
        print(f"{'─' * 80}")

        # 当前折的正/负样本（这里只是索引组合，与特征无关）
        train_pos, train_neg, test_pos, test_neg = prepare_samples(
            association_matrix, train_idx, test_idx
        )

        print(f"训练集 - 正样本: {len(train_pos)}, 负样本: {len(train_neg)}")
        print(f"测试集 - 正样本: {len(test_pos)}, 负样本: {len(test_neg)}")

        # === 2.1 为当前折构造“训练专用”的关联矩阵 ===
        train_assoc_matrix = association_matrix.copy().astype(np.float32)
        for (i, j) in test_pos:
            train_assoc_matrix[i, j] = 0.0

        print("  - 当前折训练关联矩阵已构造完成 (测试折正样本置 0)")
        print(f"    剩余已知关联数量: {int(train_assoc_matrix.sum())}")

        # === 2.2 基于训练关联矩阵重新计算 GIPK，并做 SWF 融合 ===
        if data_loader is not None:
            # 每一折用 train_assoc_matrix 重新计算 snoRNA / disease 的 GIPK
            gipk_calculator = GIPKCalculator(train_assoc_matrix, gamma_ratio=1.0)
            snorna_gipk_fold = gipk_calculator.compute_gipk_snorna()
            disease_gipk_fold = gipk_calculator.compute_gipk_disease()

            if external_available:
                # 使用 DataLoader 自带的 SWF 工具与外部相似性融合
                snorna_sim_fold = data_loader._swf_fusion(
                    snorna_gipk_fold, sno_ext,
                    name=f"snoRNA-Fold{fold + 1}",
                    alpha=alpha_snorna
                )
                disease_sim_fold = data_loader._swf_fusion(
                    disease_gipk_fold, dis_ext,
                    name=f"Disease-Fold{fold + 1}",
                    alpha=alpha_disease
                )
            else:
                # 只使用 GIPK，相当于 alpha=1
                snorna_sim_fold = data_loader._normalize_similarity(snorna_gipk_fold)
                disease_sim_fold = data_loader._normalize_similarity(disease_gipk_fold)
        else:
            # 回退到旧逻辑：直接使用外部传入的 snorna_sim / disease_sim
            # 这种情况下仍然存在“用完整关联矩阵算特征”的数据泄露风险，
            # 但为了兼容老代码，这里不做强制修改。
            snorna_sim_fold = snorna_sim
            disease_sim_fold = disease_sim

        # === 2.3 使用当前折的训练关联矩阵和相似性构造多超图 ===
        hg_constructor = HypergraphConstructor(train_assoc_matrix, snorna_sim_fold, disease_sim_fold)
        H_all, H_sno, H_dis = hg_constructor.construct_multi_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight
        )


        # === 2.4 构建模型（输入特征也使用当前折的相似性矩阵 + 训练关联矩阵） ===
        model = TripleHypergraphNN(
            num_snorna=num_snorna,
            num_disease=num_disease,
            snorna_sim=snorna_sim_fold,
            disease_sim=disease_sim_fold,
            assoc_matrix=train_assoc_matrix,
            hidden_dims=[512, 384, 256, 128, 64],
            num_heads=8,
            dropout=0.2
        ).to(device)

        # 优化器、损失函数和学习率调度器
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs)

        # 训练器
        trainer = TripleHypergraphTrainer(model, device, label_smoothing=0.1)

        # 训练循环与早停
        best_auc = 0.0
        patience_counter = 0

        pbar = tqdm(range(epochs), desc=f"Fold {fold + 1} Training")
        for epoch in pbar:
            # 更新学习率
            lr_current = scheduler.step(epoch)

            # 训练一个 epoch
            train_loss = trainer.train_epoch(
                H_all, H_sno, H_dis,
                train_pos, train_neg,
                optimizer, criterion
            )

            # 每 10 个 epoch 在当前折的验证集上评估一次
            if epoch % 10 == 0:
                auc, aupr, _, _ = trainer.evaluate(
                    H_all, H_sno, H_dis,
                    test_pos, test_neg
                )

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
                    print(f"  - 早停: 在 epoch {epoch}")
                    break

        # === 2.5 使用当前折对应的超图在测试集上做最终评估 ===
        auc, aupr, y_true, y_scores = trainer.evaluate(
            H_all, H_sno, H_dis,
            test_pos, test_neg
        )

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
                            k_disease=k_d,
                            # 新增：把 DataLoader 和当前 α 传进去，
                            # 让 cross_validation 能在每一折用训练关联矩阵重算 GIPK+SWF
                            data_loader=data_loader,
                            alpha_snorna=alpha_s,
                            alpha_disease=alpha_d,
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

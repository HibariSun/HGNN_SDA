"""
Author:Hibari
2025年11月08日17时10分25秒
syh19990131@gmail.com
超图神经网络 - snoRNA-Disease关联预测
"""

# main.py
import numpy as np

from utils import set_seed, device, init_logging, print_device_info
from data_utils import DataLoader
from train_eval import grid_search
from visualization import ResultVisualizer, save_results


# 主函数
def main():
    # 1. 先初始化日志（所有后续 print 都会被记录）
    log_path = init_logging()
    print(f"[日志] 当前运行日志文件: {log_path}")

    # 2. 再设置随机种子 / 打印设备信息
    set_seed(42)
    print_device_info()

    # 3. 后面就是你原来的流程
    print("\n开始训练流程...\n")

    # 初始化数据加载器（GIPK / 外部相似性会在第一次调用时计算并缓存）
    data_loader = DataLoader(data_path='./data/')

    # ====== 网格搜索空间======
    # alpha_snorna_list = [0.3, 0.5, 0.7]
    # alpha_disease_list = [0.3, 0.5, 0.7]
    # k_snorna_list = [10, 20, 30]
    # k_disease_list = [10, 20, 30]
    # association_weight_list = [0.5, 0.8, 1.0]

    alpha_snorna_list = [0.7]
    alpha_disease_list = [0.7]
    k_snorna_list = [20]
    k_disease_list = [30]
    association_weight_list = [0.8]
    # 训练相关参数
    n_splits = 5
    epochs = 300
    lr = 0.00001
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

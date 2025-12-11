"""
从疾病LLM embedding计算余弦相似性矩阵

输入: disease_feat_bgeicl.csv (60×4096 embedding)
输出: disease_similarity.csv (60×60 相似性矩阵)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_disease_similarity(embedding_path, output_path):
    """
    从LLM embedding计算疾病间的余弦相似性
    
    参数:
        embedding_path: disease embedding文件路径
        output_path: 输出相似性矩阵路径
    """
    # 加载embedding
    df = pd.read_csv(embedding_path, index_col=0)
    
    diseases = df.index.tolist()
    embeddings = df.values  # [n_diseases, embedding_dim]
    
    print(f"加载了 {len(diseases)} 个疾病的embedding")
    print(f"Embedding维度: {embeddings.shape[1]}")
    
    # 计算余弦相似性
    print("\n计算余弦相似性矩阵...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 转为DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=diseases,
        columns=diseases
    )
    
    # 统计信息
    print(f"\n相似性矩阵统计:")
    print(f"  - 形状: {similarity_df.shape}")
    print(f"  - 最小值: {similarity_matrix.min():.4f}")
    print(f"  - 最大值: {similarity_matrix.max():.4f}")
    print(f"  - 均值: {similarity_matrix.mean():.4f}")
    print(f"  - 对角线均值: {np.diag(similarity_matrix).mean():.4f} (应该=1.0)")
    
    # 保存
    similarity_df.to_csv(output_path)
    print(f"\n✓ 相似性矩阵已保存至: {output_path}")
    
    return similarity_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算疾病间的余弦相似性')
    parser.add_argument('-i', '--input', default='disease_feat_bgeicl.csv',
                        help='输入embedding文件')
    parser.add_argument('-o', '--output', default='disease_similarity.csv',
                        help='输出相似性矩阵文件')
    
    args = parser.parse_args()
    
    compute_disease_similarity(args.input, args.output)

"""
将GL4SDA的sd_association.csv转换为HGNN-SDA格式的adj_index.csv

输入格式 (sd_association.csv):
    ,Symbol,DO_ID,score
    0,SNORA11E,DOID:0050745,0.328976
    ...

输出格式 (adj_index.csv):
    ,DOID:xxx,DOID:yyy,...
    snoRNA1,1.0,0.0,...
    snoRNA2,0.0,1.0,...
"""

import pandas as pd
import numpy as np


def convert_sd_to_adj(input_path, output_path, use_binary=True):
    """
    将sd_association.csv转换为adj_index.csv
    
    参数:
        input_path: 输入文件路径 (sd_association.csv)
        output_path: 输出文件路径 (adj_index.csv)
        use_binary: True使用0/1二值，False使用原始score
    """
    # 加载数据
    sd_df = pd.read_csv(input_path, index_col=0)
    print(f"加载了 {len(sd_df)} 条关联记录")
    print(f"  - snoRNA数量: {sd_df['Symbol'].nunique()}")
    print(f"  - 疾病数量: {sd_df['DO_ID'].nunique()}")
    
    # 获取所有唯一的snoRNA和疾病(DOID)
    snornas = sorted(sd_df['Symbol'].unique())
    diseases = sorted(sd_df['DO_ID'].unique())
    
    print(f"\n创建邻接矩阵: {len(snornas)} snoRNAs × {len(diseases)} diseases")
    
    # 初始化零矩阵
    adj_matrix = pd.DataFrame(
        np.zeros((len(snornas), len(diseases)), dtype=np.float32),
        index=snornas,
        columns=diseases
    )
    
    # 填充关联
    for _, row in sd_df.iterrows():
        snorna = row['Symbol']
        doid = row['DO_ID']
        score = row['score']
        
        if use_binary:
            adj_matrix.loc[snorna, doid] = 1.0
        else:
            # 如果同一对有多条记录，取最大值
            adj_matrix.loc[snorna, doid] = max(adj_matrix.loc[snorna, doid], score)
    
    # 统计
    n_associations = int((adj_matrix > 0).sum().sum())
    print(f"  - 总关联数: {n_associations}")
    print(f"  - 稀疏度: {1 - n_associations / (len(snornas) * len(diseases)):.4f}")
    
    # 保存
    adj_matrix.to_csv(output_path)
    print(f"\n✓ 已保存至: {output_path}")
    print(f"  - 形状: {adj_matrix.shape}")
    
    return adj_matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GL4SDA sd_association.csv → adj_index.csv')
    parser.add_argument('-i', '--input', default='sd_association.csv', help='输入文件')
    parser.add_argument('-o', '--output', default='adj_index.csv', help='输出文件')
    parser.add_argument('--use-score', action='store_true', help='使用原始score而非二值')
    
    args = parser.parse_args()
    
    convert_sd_to_adj(args.input, args.output, use_binary=not args.use_score)

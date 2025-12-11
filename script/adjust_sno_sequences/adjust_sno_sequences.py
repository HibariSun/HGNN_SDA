# Author:Hibari
# 2025年12月11日21时52分58秒
# syh19990131@gmail.com
import pandas as pd

# 加载
adj_df = pd.read_csv('adj_index.csv', index_col=0)
seq_df = pd.read_csv('sno_sequences.csv')

# 按adj_index的顺序重排
target_order = list(adj_df.index)
seq_df = seq_df.set_index('sno_id').loc[target_order].reset_index()

# 保存
seq_df.to_csv('sno_sequences.csv', index=False)
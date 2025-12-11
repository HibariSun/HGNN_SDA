# Author:Hibari
# 2025年11月27日23时47分39秒
# syh19990131@gmail.com
import os
import numpy as np
import pandas as pd

def main():
    # 1. 读入 embedding
    emb_path = os.path.join("data", "sno_rnaernie_emb.npy")
    emb = np.load(emb_path)   # shape: [num_sno, dim]
    print("Embedding shape:", emb.shape)

    # 2. L2 归一化：每一行是一个 snoRNA
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_norm = emb / norm

    # 3. 余弦相似度矩阵 = 归一化后的点积
    #    sim[i,j] = cos(emb[i], emb[j])
    sim = emb_norm @ emb_norm.T   # shape: [num_sno, num_sno]
    print("Similarity shape:", sim.shape)

    # 4. 保存为 .npy
    sim_npy_path = os.path.join("data", "sno_rnaernie_cosine_sim.npy")
    np.save(sim_npy_path, sim)
    print(f"Saved npy to {sim_npy_path}")

    # 5. 保存为 .csv
    sim_csv_path = os.path.join("data", "sno_rnaernie_cosine_sim.csv")
    df = pd.DataFrame(sim)
    df.to_csv(sim_csv_path, index=False, header=False)
    print(f"Saved csv to {sim_csv_path}")

if __name__ == "__main__":
    main()

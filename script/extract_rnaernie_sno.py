# Author:Hibari
# 2025年11月27日23时03分27秒
# syh19990131@gmail.com

import os
import re
import numpy as np
import pandas as pd
import torch
from multimolecule import RnaTokenizer, RnaErnieModel

def clean_seq(seq: str) -> str:
    """轻量清洗: 去空白, 大写, T->U, 只保留 A/C/G/U/N"""
    seq = re.sub(r"\s+", "", str(seq))  # 去掉空白
    seq = seq.upper()
    seq = seq.replace("T", "U")
    allowed = set("ACGUN")
    seq = "".join(ch for ch in seq if ch in allowed)
    return seq

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 读入 sno_sequences.csv
    seq_path = os.path.join("data", "sno_sequences.csv")
    df = pd.read_csv(seq_path)

    if "sequence" not in df.columns:
        raise ValueError("sno_sequences.csv 里找不到 'sequence' 这一列")

    # 如果没有 sno_id，你也可以用 index 做占位
    if "sno_id" not in df.columns:
        df["sno_id"] = [f"sno_{i}" for i in range(len(df))]

    # 2. 清洗序列
    df["sequence_clean"] = df["sequence"].apply(clean_seq)
    # 过滤掉空序列的
    df = df[df["sequence_clean"].str.len() > 0].reset_index(drop=True)

    sno_ids = df["sno_id"].tolist()
    seqs = df["sequence_clean"].tolist()

    print(f"Total snoRNA with valid sequences: {len(seqs)}")

    # 3. 加载 RNAErnie
    local_rnaernie_path = "/root/autodl-tmp/models/rnaernie"  # 按你实际解压路径填

    tokenizer = RnaTokenizer.from_pretrained(local_rnaernie_path)
    model = RnaErnieModel.from_pretrained(local_rnaernie_path).to(device)

    model.eval()

    batch_size = 32
    emb_list = []

    # 4. 分批提取 CLS 向量
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # [B, L, D]
            cls_emb = hidden[:, 0, :]           # [B, D]

            emb_list.append(cls_emb.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"Processed {min(i + batch_size, len(seqs))}/{len(seqs)} sequences")

    emb = np.concatenate(emb_list, axis=0)  # [num_sno, D]
    print("Embedding shape:", emb.shape)

    # 5. 保存结果
    out_emb_path = os.path.join("data", "sno_rnaernie_emb.npy")
    np.save(out_emb_path, emb)
    print(f"Saved embeddings to {out_emb_path}")

    # 保存 sno_id 顺序，方便以后对齐
    out_id_path = os.path.join("data", "sno_rnaernie_ids.txt")
    with open(out_id_path, "w") as f:
        for sid in sno_ids:
            f.write(str(sid) + "\n")
    print(f"Saved sno id list to {out_id_path}")

if __name__ == "__main__":
    main()

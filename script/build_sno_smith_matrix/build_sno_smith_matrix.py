# Author:Hibari
# 2025年11月27日22时34分41秒
# syh19990131@gmail.com
import pandas as pd
import numpy as np
import parasail
from math import sqrt

# 配置区
INPUT_SEQ_CSV = "sno_sequences.csv"  # 输入：你的序列表
OUTPUT_SIM_CSV = "my_sno_p2p_smith_cov.csv"  # 输出：生成的相似性矩阵（带 coverage）

SNO_ID_COL = "sno_id"  # snoRNA ID 的列名
SEQ_COL = "sequence"  # 序列的列名

# Smith-Waterman 参数（可以与原论文保持一致）
GAP_OPEN = 5
GAP_EXTEND = 2
SCORING_MATRIX = parasail.nuc44  # 用 DNA 风格的核酸矩阵

# coverage 相关的超参数
USE_COVERAGE_THRESHOLD = True  # 是否启用 coverage 阈值
COVERAGE_MIN = 0.3  # 覆盖度低于这个值的相似度可以压成 0（你可以自己调）


# 辅助函数
def normalize_seq(s: str) -> str:
    """
    把 RNA/DNA 序列标准化成 parasail.nuc44 能接受的形式：
    - 去掉前后空格
    - 大写
    - U -> T （把 RNA 当 DNA 来算）
    """
    s = s.strip().upper()
    s = s.replace("U", "T")
    return s


def parse_cigar_coverage(cigar_str: str, len_i: int, len_j: int) -> float:
    """
    根据 CIGAR 字符串估计 coverage。

    简单做法：
    - 按 SAM 规范解析 CIGAR：如 "10M1I5M2D..."
    - 对于每个操作 (op):
        - 如果 op in 'M=XID'，说明这一段在对齐里占了一定的“列数”
        - 列数 += 这一段的长度 n
    - coverage = aligned_columns / max(len_i, len_j)

    注意：这是一个合理且常见的近似，足够用来抑制“很短的局部高分”。
    """
    if len_i == 0 and len_j == 0:
        return 0.0

    aligned_cols = 0
    num = ""

    for ch in cigar_str:
        if ch.isdigit():
            num += ch
        else:
            if not num:
                continue
            length = int(num)
            num = ""

            # M / = / X : 消耗两条序列
            # I : 消耗 query（这里可以理解为 s_i）
            # D : 消耗 ref（这里可以理解为 s_j）
            # 这些都占对齐的一列
            if ch in "M=XID":
                aligned_cols += length
            # 其他操作（S, H, P, N）这里可以忽略

    max_len = max(len_i, len_j)
    if max_len <= 0:
        return 0.0

    cov = aligned_cols / max_len
    if cov < 0.0:
        cov = 0.0
    elif cov > 1.0:
        cov = 1.0
    return cov


# 1. 读取序列
df = pd.read_csv(INPUT_SEQ_CSV)

if SNO_ID_COL not in df.columns or SEQ_COL not in df.columns:
    raise ValueError(f"找不到列 {SNO_ID_COL} 或 {SEQ_COL}，请检查 {INPUT_SEQ_CSV} 的列名")

sno_ids = df[SNO_ID_COL].astype(str).tolist()
raw_seqs = df[SEQ_COL].astype(str).tolist()

seqs = [normalize_seq(s) for s in raw_seqs]
lengths = [len(s) for s in seqs]

n = len(sno_ids)
print(f"共读取 {n} 条 snoRNA 序列。")

# 2. 自比对得分 SW(s_i, s_i)
self_scores = np.zeros(n, dtype=float)

print("正在计算 self-score（SW(s_i, s_i））...")

for i in range(n):
    seq = seqs[i]
    if len(seq) == 0:
        self_scores[i] = 1e-6
        continue
    res = parasail.sw(seq, seq, GAP_OPEN, GAP_EXTEND, SCORING_MATRIX)
    score_ii = res.score
    if score_ii <= 0:
        score_ii = 1e-6
    self_scores[i] = score_ii

# 3. 计算两两相似性：归一化 SW × coverage
sim_mat = np.zeros((n, n), dtype=float)

print("正在计算两两 Smith-Waterman 相似性 + coverage（会比之前稍慢一些）...")

for i in range(n):
    seq_i = seqs[i]
    len_i = lengths[i]
    score_ii = self_scores[i]

    if (i + 1) % 10 == 0 or i == n - 1:
        print(f"  进度：{i + 1}/{n}")

    for j in range(i, n):
        if i == j:
            # 自己和自己，直接设成 1
            sim_mat[i, j] = 1.0
            continue

        seq_j = seqs[j]
        len_j = lengths[j]
        score_jj = self_scores[j]

        if len_i == 0 or len_j == 0:
            sim_ij = 0.0
        else:
            # 用 traceback 版本拿到 CIGAR，用来算 coverage
            res = parasail.sw_trace(seq_i, seq_j, GAP_OPEN, GAP_EXTEND, SCORING_MATRIX)
            score_ij = res.score

            # 归一化
            denom = sqrt(score_ii * score_jj) if score_ii > 0 and score_jj > 0 else 1e-6
            swnorm = score_ij / denom

            # coverage
            cigar = res.cigar
            cigar_bytes = cigar.decode  # 返回 bytes，如 b"10M1I5M"
            cigar_str = cigar_bytes.decode('utf-8') if isinstance(cigar_bytes, bytes) else str(cigar_bytes)
            cov = parse_cigar_coverage(cigar_str, len_i, len_j)

            if USE_COVERAGE_THRESHOLD and cov < COVERAGE_MIN:
                sim_ij = 0.0
            else:
                sim_ij = swnorm * cov

            # 理论上 0-1，保险裁剪一下
            if sim_ij < 0:
                sim_ij = 0.0
            elif sim_ij > 1:
                sim_ij = 1.0

        sim_mat[i, j] = sim_ij
        sim_mat[j, i] = sim_ij  # 对称

# 对角线强制设为 1（以防万一）
np.fill_diagonal(sim_mat, 1.0)

# 4. 保存成 CSV
sim_df = pd.DataFrame(sim_mat, index=sno_ids, columns=sno_ids)
sim_df.to_csv(OUTPUT_SIM_CSV)

print(f"\n带 coverage 的相似性矩阵已保存到：{OUTPUT_SIM_CSV}")
print("形状：", sim_df.shape)
print("示例：前 3x3 矩阵：")
print(sim_df.iloc[:3, :3])
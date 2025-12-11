"""
将FASTA格式的snoRNA序列文件转换为CSV格式

输入格式 (snorna_sequences.fas):
    >SNORA11E
    GGGGUGUGCUCAGAGCAGGGGGCCUGAAGAAUGGCUCCUCUGUUUACAACACACCAACAGGAAGCUGGGGUCAUCGUGAUGAGGGGCACAAACUUGUGGCCUCCCUACAGACAAAUGCCCUACAUGU
    >SNORA13
    AGCCUUUGUGUUGCCCAUUCACUUUGGAAACUAGUGAAUGUGGUGUCAAAAAGGCGUAAAUUAAACGCUUUGCAGCCUUUUCCUGCCCUUAAAUUUGAUACCUUUGGUGUAGGAGCUGCAUAAGUAACAGUU

输出格式 (sno_sequences.csv):
    sno_id,sequence
    SNORA11E,GGGGTGTGCTCAGAGCAGGGG...
"""

import argparse


def parse_fasta(fasta_path):
    """解析FASTA文件"""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # 保存上一条序列
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # 开始新序列
                current_id = line[1:]  # 去掉 '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一条序列
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def convert_fasta_to_csv(input_path, output_path, rna_to_dna=True):
    """
    将FASTA转换为CSV
    
    参数:
        input_path: 输入FASTA文件
        output_path: 输出CSV文件
        rna_to_dna: 是否将U转换为T (RNA→DNA)
    """
    # 解析FASTA
    sequences = parse_fasta(input_path)
    print(f"读取了 {len(sequences)} 条序列")
    
    # 写入CSV
    with open(output_path, 'w') as f:
        f.write('sno_id,sequence\n')
        
        for sno_id, seq in sequences.items():
            # 可选：将U转换为T
            if rna_to_dna:
                seq = seq.replace('U', 'T').replace('u', 't')
            
            f.write(f'{sno_id},{seq}\n')
    
    print(f"✓ 已保存至: {output_path}")
    
    # 统计信息
    seq_lens = [len(s) for s in sequences.values()]
    print(f"\n序列统计:")
    print(f"  - 数量: {len(sequences)}")
    print(f"  - 最短: {min(seq_lens)} bp")
    print(f"  - 最长: {max(seq_lens)} bp")
    print(f"  - 平均: {sum(seq_lens)/len(seq_lens):.1f} bp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FASTA转CSV')
    parser.add_argument('-i', '--input', required=True, help='输入FASTA文件')
    parser.add_argument('-o', '--output', default='sno_sequences.csv', help='输出CSV文件')
    parser.add_argument('--keep-rna', action='store_true', help='保留U不转换为T')
    
    args = parser.parse_args()
    
    convert_fasta_to_csv(args.input, args.output, rna_to_dna=not args.keep_rna)

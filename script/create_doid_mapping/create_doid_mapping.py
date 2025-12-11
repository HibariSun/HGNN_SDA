"""
从GL4SDA的disease_description.txt提取DOID到疾病名称的映射
包含手动修正的映射
"""

import pandas as pd

# 手动修正的DOID到疾病名称映射（基于Disease Ontology官方名称）
MANUAL_MAPPING = {
    "DOID:0050745": "Diffuse Large B-Cell Lymphoma",
    "DOID:0050866": "Papillary Thyroid Carcinoma",
    "DOID:0050894": "Ameloblastoma",
    "DOID:0060041": "Autism Spectrum Disorder",
    "DOID:0060318": "Acute Promyelocytic Leukemia",
    "DOID:0060669": "Cerebral Cavernous Malformation",
    "DOID:0080600": "COVID-19",
    "DOID:10283": "Prostate Cancer",
    "DOID:10286": "Prostate Carcinoma",
    "DOID:1040": "Chronic Lymphocytic Leukemia",
    "DOID:10534": "Stomach Cancer",
    "DOID:10652": "Alzheimer Disease",
    "DOID:10941": "Intracranial Aneurysm",
    "DOID:11054": "Bladder Cancer",
    "DOID:1115": "Sarcoma",
    "DOID:11165": "Hepatitis C",
    "DOID:11476": "Osteoporosis",
    "DOID:1168": "Familial Hyperlipidemia",
    "DOID:11983": "Prader-Willi Syndrome",
    "DOID:1222": "Liver Cirrhosis",
    "DOID:1287": "Cardiovascular System Disease",
    "DOID:13223": "Uterine Fibroid",
    "DOID:1324": "Lung Cancer",
    "DOID:1612": "Breast Cancer",
    "DOID:162": "Cancer",
    "DOID:1793": "Pancreatic Cancer",
    "DOID:1883": "Hepatitis B",
    "DOID:1909": "Melanoma",
    "DOID:219": "Colon Cancer",
    "DOID:234": "Colon Carcinoma",
    "DOID:2366": "Acute Myeloid Leukemia",
    "DOID:2377": "Multiple Sclerosis",
    "DOID:2394": "Ovarian Cancer",
    "DOID:299": "Adenocarcinoma",
    "DOID:3068": "Glioblastoma",
    "DOID:3069": "Astrocytoma",
    "DOID:3070": "Malignant Glioma",
    "DOID:3347": "Osteosarcoma",
    "DOID:3498": "Pancreatic Ductal Adenocarcinoma",
    "DOID:3905": "Lung Carcinoma",
    "DOID:3908": "Non-Small Cell Lung Carcinoma",
    "DOID:3910": "Lung Adenocarcinoma",
    "DOID:3969": "Follicular Thyroid Carcinoma",
    "DOID:399": "Tuberculosis",
    "DOID:4007": "Bladder Carcinoma",
    "DOID:4362": "Cervical Cancer",
    "DOID:4467": "Clear Cell Renal Cell Carcinoma",
    "DOID:5041": "Esophageal Cancer",
    "DOID:5082": "Liver Disease",
    "DOID:5517": "Stomach Carcinoma",
    "DOID:5520": "Head and Neck Squamous Cell Carcinoma",
    "DOID:684": "Hepatocellular Carcinoma",
    "DOID:707": "B-Cell Lymphoma",
    "DOID:7148": "Rheumatoid Arthritis",
    "DOID:7474": "Malignant Pleural Mesothelioma",
    "DOID:8398": "Osteoarthritis",
    "DOID:9008": "Psoriatic Arthritis",
    "DOID:9074": "Systemic Lupus Erythematosus",
    "DOID:9119": "Acute Myeloid Leukemia",
    "DOID:9256": "Colorectal Cancer",
}


def create_doid_mapping(sd_association_path, output_path):
    """
    创建DOID到疾病名称的映射CSV
    
    参数:
        sd_association_path: sd_association.csv的路径（用于获取实际使用的DOID列表）
        output_path: 输出CSV文件路径
    """
    # 读取sd_association获取实际使用的DOID
    sd_df = pd.read_csv(sd_association_path, index_col=0)
    doids = sorted(sd_df['DO_ID'].unique())
    
    print(f"数据集中共有 {len(doids)} 种疾病")
    
    # 创建映射表
    mapping_data = []
    for doid in doids:
        name = MANUAL_MAPPING.get(doid, doid)  # 如果没有映射，使用DOID本身
        mapping_data.append({'DOID': doid, 'disease_name': name})
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # 保存
    mapping_df.to_csv(output_path, index=False)
    print(f"\n✓ 映射表已保存至: {output_path}")
    
    # 显示全部内容
    print("\n=== DOID到疾病名称映射 ===")
    for _, row in mapping_df.iterrows():
        print(f"  {row['DOID']:15} -> {row['disease_name']}")
    
    return mapping_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成DOID到疾病名称的映射CSV')
    parser.add_argument('-i', '--input', default='sd_association.csv', 
                        help='输入文件 (sd_association.csv)')
    parser.add_argument('-o', '--output', default='doid_disease_mapping.csv', 
                        help='输出文件')
    
    args = parser.parse_args()
    
    create_doid_mapping(args.input, args.output)

import pandas as pd
import random
import numpy as np

def parse_fasta_like_csv(file_path):
    sequences = []
    current_sequence = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                if current_sequence:
                    header = current_sequence[0]
                    gene_info = header.split('|')
                    sequences.append({
                        'gene_id': gene_info[0],
                        'transcript_id': gene_info[1],
                        'gene_name': gene_info[2],
                        'sequence': ''.join(current_sequence[1:])
                    })
                    current_sequence = []
                current_sequence.append(line.strip()[1:])
            else:
                current_sequence.append(line.strip())
    
    if current_sequence:
        header = current_sequence[0]
        gene_info = header.split('|')
        sequences.append({
            'gene_id': gene_info[0],
            'transcript_id': gene_info[1],
            'gene_name': gene_info[2],
            'sequence': ''.join(current_sequence[1:])
        })
    
    return pd.DataFrame(sequences)

def process_3utr_sequence(sequence):
    if len(sequence) > 5000:
        return sequence[:500]
    elif len(sequence) < 500:
        return sequence + 'N' * (500 - len(sequence))
    return sequence[:500]

def create_mti_pair(mrna_seq, mirna_seq):
    return mrna_seq + 'N' * 6 + mirna_seq

def normalize_mirna_id(mirna_id):
    """标准化miRNA ID格式"""
    # 移除'hsa-'前缀
    mirna_id = mirna_id.replace('hsa-', '')
    # 转换为大写
    mirna_id = mirna_id.upper()
    # 移除所有连字符
    mirna_id = mirna_id.replace('-', '')
    # 替换 miR 为 MIR
    mirna_id = mirna_id.replace('MIR', 'MIR')
    # 处理3p/5p后缀
    if '3P' in mirna_id:
        mirna_id = mirna_id.replace('3P', '3P')
    if '5P' in mirna_id:
        mirna_id = mirna_id.replace('5P', '5P')
    return mirna_id

def load_and_process_data():
    print("Reading input files...")
    mti_df = pd.read_csv('hsa_MTI.csv')
    mature_df = pd.read_csv('mature.csv')
    utr_df = parse_fasta_like_csv('ensembl_biomart_human_3utr.csv')
    
    # 打印一些基因名称示例
    print("\nSample gene names from UTR data:")
    print(utr_df['gene_name'].head())
    print("\nSample target genes from MTI data:")
    print(mti_df['Target Gene'].head())
    
    # 创建基因名称到序列的映射
    utr_dict = {}
    gene_name_mapping = {}
    for _, row in utr_df.iterrows():
        # 存储原始名称和大写名称
        gene_name = row['gene_name']
        gene_name_upper = gene_name.upper()
        utr_dict[gene_name] = row['sequence']
        utr_dict[gene_name_upper] = row['sequence']
        
        # 存储基因名称的变体
        gene_name_mapping[gene_name] = gene_name
        gene_name_mapping[gene_name_upper] = gene_name
        # 处理可能的别名（去除括号内容）
        base_name = gene_name.split('(')[0].strip()
        gene_name_mapping[base_name] = gene_name
        gene_name_mapping[base_name.upper()] = gene_name
    
    # 处理3'UTR序列
    print("\nProcessing 3'UTR sequences...")
    for gene_name in list(utr_dict.keys()):
        utr_dict[gene_name] = process_3utr_sequence(utr_dict[gene_name])
    
    # 创建miRNA ID到序列的映射
    mirna_dict = {}
    for _, row in mature_df.iterrows():
        if not str(row['Sequence_ID']).startswith('hsa-'):
            continue
        original_id = str(row['Sequence_ID']).split()[0]
        normalized_id = normalize_mirna_id(original_id)
        mirna_dict[normalized_id] = row['Sequence']
        mirna_dict[original_id] = row['Sequence']
        # 存储没有3p/5p后缀的版本
        base_id = normalized_id
        for suffix in ['3P', '5P']:
            if suffix in normalized_id:
                base_id = normalized_id.replace(suffix, '')
                mirna_dict[base_id] = row['Sequence']
    
    # 打印一些基因名称示例
    print("\nSample gene names from UTR data:")
    print(utr_df['gene_name'].head())
    print("\nSample target genes from MTI data:")
    print(mti_df['Target Gene'].head())
    
    # 创建ID映射字典
    id_mapping = {}
    for _, row in mti_df.iterrows():
        original_id = row['miRNA']
        normalized_id = normalize_mirna_id(original_id)
        if normalized_id not in id_mapping:
            id_mapping[normalized_id] = set()
        id_mapping[normalized_id].add(original_id)
    
    print("\nSample ID mappings:")
    sample_mappings = list(id_mapping.items())[:5]
    for norm_id, orig_ids in sample_mappings:
        print(f"{norm_id}: {orig_ids}")
    
    # 打印一些标准化后的ID示例
    print("\nSample normalized IDs:")
    sample_ids = list(mirna_dict.keys())[:5]
    for id in sample_ids:
        print(id)
    
    print("\nProcessing 3'UTR sequences...")
    for gene_name in utr_dict:
        utr_dict[gene_name] = process_3utr_sequence(utr_dict[gene_name])
    
    print("\nGenerating positive pairs...")
    positive_pairs = []
    processed_count = 0
    
    # Print some statistics
    print(f"Number of unique UTR genes: {len(utr_dict)}")
    print(f"Number of unique miRNAs: {len(mirna_dict)}")
    
    for _, row in mti_df.iterrows():
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count}/{len(mti_df)} MTI entries")
            
        try:
            gene_name = row['Target Gene'].split('(')[0].strip().upper()
            mirna_id = normalize_mirna_id(row['miRNA'])
            
            if gene_name in utr_dict and mirna_id in mirna_dict:
                mrna_seq = utr_dict[gene_name]
                mirna_seq = mirna_dict[mirna_id]
                pair = create_mti_pair(mrna_seq, mirna_seq)
                positive_pairs.append({
                    'sequence': pair,
                    'label': 1,
                    'gene_id': gene_name,
                    'mirna_id': row['miRNA']
                })
            else:
                if gene_name not in utr_dict:
                    print(f"Gene variants tried: {[gene_name, gene_name.split('.')[0], gene_name.split('(')[0].strip()]}")
                if mirna_id not in mirna_dict:
                    print(f"miRNA variants tried: {[mirna_id, original_id, base_id]}")
                    
        except Exception as e:
            print(f"Error processing entry {row['Target Gene']}: {str(e)}")
            continue
    
    print(f"\nSuccessfully generated {len(positive_pairs)} positive pairs")
    
    if len(positive_pairs) == 0:
        print("No positive pairs were generated. Stopping process.")
        return pd.DataFrame()
    
    print("\nGenerating negative pairs...")
    n_negative = len(positive_pairs)
    used_pairs = set((row['Target Gene'].split('(')[0].strip().upper(), 
                      normalize_mirna_id(row['miRNA'])) 
                     for _, row in mti_df.iterrows())
    
    negative_pairs = []
    utr_items = list(utr_dict.items())
    mirna_items = list(mirna_dict.items())
    
    while len(negative_pairs) < n_negative:
        utr_item = random.choice(utr_items)
        mirna_item = random.choice(mirna_items)
        
        if (utr_item[0], mirna_item[0]) not in used_pairs:
            pair = create_mti_pair(utr_item[1], mirna_item[1])
            negative_pairs.append({
                'sequence': pair,
                'label': 0,
                'gene_id': utr_item[0],
                'mirna_id': mirna_item[0]
            })
    
    print("Finalizing dataset...")
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    final_df = pd.DataFrame(all_pairs)
    
    output_file = 'mti_pairs.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
    
    return final_df

if __name__ == "__main__":
    print("Starting MTI pairs processing...")
    try:
        result_df = load_and_process_data()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 
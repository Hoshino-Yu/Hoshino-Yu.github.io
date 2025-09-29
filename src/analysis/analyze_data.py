#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
from collections import defaultdict

def analyze_all_datasets():
    """分析所有数据集"""
    
    # 定义所有数据文件
    data_files = {
        '企业基本信息': '../../data/raw/01_企业基本信息.xlsx',
        '信贷申请数据': '../../data/raw/02_信贷申请数据.xlsx', 
        '财务报表数据': '../../data/raw/03_财务报表数据.xlsx',
        '税务记录': '../../data/raw/04_税务记录.xlsx',
        '银行流水': '../../data/raw/05_银行流水.xlsx',
        '企业主信用数据': '../../data/raw/06_企业主信用数据.xlsx',
        '行业经营状况': '../../data/raw/07_行业经营状况.xlsx',
        '企业征信报告': '../../data/raw/08_企业征信报告.xlsx',
        '上下游合作情况': '../../data/raw/09_上下游合作情况.xlsx',
        '企业经营风险评估': '../../data/raw/10_企业经营风险评估.xlsx',
        '合并信贷数据': '../../data/raw/中小企业合并信贷数据.xlsx'
    }
    
    all_columns = defaultdict(list)  # 记录所有列名及其出现的文件
    text_columns = {}  # 记录文本型列
    numeric_columns = {}  # 记录数值型列
    dataset_info = {}  # 记录每个数据集的详细信息
    
    print("=== 分析所有数据集 ===\n")
    
    for name, filename in data_files.items():
        if not os.path.exists(filename):
            print(f"⚠️  文件不存在: {filename}")
            continue
            
        try:
            df = pd.read_excel(filename)
            print(f"📊 {name} ({filename})")
            print(f"   形状: {df.shape}")
            print(f"   列数: {df.shape[1]}")
            
            # 记录数据集信息
            dataset_info[name] = {
                'filename': filename,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'text_columns': [],
                'numeric_columns': [],
                'categorical_columns': []
            }
            
            # 分析每一列
            for col in df.columns:
                all_columns[col].append(name)
                
                # 判断列类型
                if df[col].dtype == 'object':
                    # 文本型列
                    unique_values = df[col].nunique()
                    sample_values = df[col].dropna().unique()[:5]
                    
                    dataset_info[name]['text_columns'].append({
                        'column': col,
                        'unique_count': unique_values,
                        'sample_values': list(sample_values),
                        'missing_rate': df[col].isnull().mean()
                    })
                    
                    if col not in text_columns:
                        text_columns[col] = []
                    text_columns[col].append({
                        'dataset': name,
                        'unique_count': unique_values,
                        'sample_values': list(sample_values)
                    })
                    
                else:
                    # 数值型列
                    dataset_info[name]['numeric_columns'].append({
                        'column': col,
                        'dtype': str(df[col].dtype),
                        'min': df[col].min() if not df[col].isnull().all() else None,
                        'max': df[col].max() if not df[col].isnull().all() else None,
                        'mean': df[col].mean() if not df[col].isnull().all() else None,
                        'missing_rate': df[col].isnull().mean()
                    })
                    
                    if col not in numeric_columns:
                        numeric_columns[col] = []
                    numeric_columns[col].append({
                        'dataset': name,
                        'dtype': str(df[col].dtype),
                        'stats': {
                            'min': df[col].min() if not df[col].isnull().all() else None,
                            'max': df[col].max() if not df[col].isnull().all() else None,
                            'mean': df[col].mean() if not df[col].isnull().all() else None
                        }
                    })
            
            print(f"   文本型列: {len(dataset_info[name]['text_columns'])}")
            print(f"   数值型列: {len(dataset_info[name]['numeric_columns'])}")
            print()
            
        except Exception as e:
            print(f"❌ 读取 {filename} 时出错: {e}\n")
    
    # 分析重复列
    print("=== 重复列分析 ===")
    duplicate_columns = {col: files for col, files in all_columns.items() if len(files) > 1}
    
    if duplicate_columns:
        for col, files in duplicate_columns.items():
            print(f"📋 '{col}' 出现在: {', '.join(files)}")
    else:
        print("✅ 未发现重复列名")
    
    print(f"\n总计发现 {len(duplicate_columns)} 个重复列名\n")
    
    # 分析文本型数据
    print("=== 文本型数据分析 ===")
    for col, info_list in text_columns.items():
        print(f"📝 {col}:")
        for info in info_list:
            print(f"   - {info['dataset']}: {info['unique_count']} 个唯一值")
            print(f"     样本值: {info['sample_values']}")
    
    print(f"\n总计 {len(text_columns)} 个文本型特征\n")
    
    # 识别可能的标识符列
    print("=== 标识符列识别 ===")
    id_keywords = ['id', 'ID', '编号', '名称', '姓名', '企业名', '法人']
    potential_id_columns = []
    
    for col in all_columns.keys():
        if any(keyword in col for keyword in id_keywords):
            potential_id_columns.append(col)
    
    if potential_id_columns:
        print("🏷️  可能的标识符列:")
        for col in potential_id_columns:
            files = all_columns[col]
            print(f"   - {col} (出现在: {', '.join(files)})")
    else:
        print("✅ 未发现明显的标识符列")
    
    print(f"\n总计 {len(potential_id_columns)} 个可能的标识符列\n")
    
    # 保存分析结果
    analysis_result = {
        'dataset_info': dataset_info,
        'duplicate_columns': duplicate_columns,
        'text_columns': text_columns,
        'numeric_columns': numeric_columns,
        'potential_id_columns': potential_id_columns
    }
    
    return analysis_result

if __name__ == "__main__":
    result = analyze_all_datasets()
    print("=== 数据集分析完成 ===")
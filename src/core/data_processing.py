#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDataProcessor:
    def __init__(self, output_dir='.', config_file=None):
        self.output_dir = output_dir
        self.config_file = config_file
        self.merged_data = None
        self.final_dataset = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.encoded_features = []
        self.processing_log = []
        self.feature_importance = None
        
        self.files_config = {
            '企业基本信息': '../原始数据/01_企业基本信息.xlsx',
            '信贷申请数据': '../原始数据/02_信贷申请数据.xlsx', 
            '财务报表数据': '../原始数据/03_财务报表数据.xlsx',
            '税务记录': '../原始数据/04_税务记录.xlsx',
            '银行流水': '../原始数据/05_银行流水.xlsx',
            '企业主信用数据': '../原始数据/06_企业主信用数据.xlsx',
            '行业经营状况': '../原始数据/07_行业经营状况.xlsx',
            '企业征信报告': '../原始数据/08_企业征信报告.xlsx',
            '上下游合作情况': '../原始数据/09_上下游合作情况.xlsx',
            '企业经营风险评估': '../原始数据/10_企业经营风险评估.xlsx',
            '合并信贷数据': '../原始数据/中小企业合并信贷数据.xlsx'
        }
        
        self.features_to_remove = [
            'Unnamed: 0', '企业ID', '企业名称', '企业法人', '企业主姓名',
            '申请日期', '最近一次信用查询时间', '月份', '年份'
        ]
        
        self.custom_encoding_rules = {
            'binary': {
                '是': 1, '否': 0,
                '有': 1, '无': 0,
                'True': 1, 'False': 0,
                '男': 1, '女': 0
            },
            'level_high_low': {
                '高': 3, '中': 2, '低': 1
            },
            'credit_grade': {
                'A': 4, 'B': 3, 'C': 2, 'M': 1
            },
            'education': {
                '博士': 5, '硕士': 4, '本科': 3, '专科': 2, '高中及以下': 1
            }
        }
        
        self.categorical_features = [
            '所在地区', '所属行业', '纳税信用等级', '行业风险等级',
            '供应链稳定性评级', '风险等级', '担保方式', '企业类型',
            '是否高新技术企业', '是否有失信记录', '是否有不良记录',
            '是否涉诉', '行业竞争程度', '教育程度', '职业背景',
            '申请机构', '贷款用途', '是否有未结清贷款'
        ]
        
        self.chinese_name_mapping = {
            '注册资本(万元)_企业基本': '注册资本',
            '成立年限_企业基本': '成立年限',
            '员工人数_企业基本': '员工人数',
            '是否高新技术企业_企业基本': '高新技术企业',
            '企业类型_企业基本': '企业类型',
            '所在地区_企业基本': '所在地区',
            '所属行业_企业基本': '所属行业',
            '总资产(万元)_财务报表': '总资产',
            '净资产(万元)_财务报表': '净资产',
            '营业收入(万元)_财务报表': '营业收入',
            '净利润(万元)_财务报表': '净利润',
            '资产负债率_财务报表': '资产负债率',
            '流动比率_财务报表': '流动比率',
            '速动比率_财务报表': '速动比率',
            '净资产收益率_财务报表': '净资产收益率',
            '总资产周转率_财务报表': '总资产周转率',
            '申请金额(万元)_信贷申请': '申请金额',
            '贷款期限(月)_信贷申请': '贷款期限',
            '担保方式_信贷申请': '担保方式',
            '贷款用途_信贷申请': '贷款用途',
            '申请机构_信贷申请': '申请机构',
            '是否有未结清贷款_信贷申请': '有未结清贷款',
            '企业信用评分_企业征信报告': '企业信用评分',
            '贷款逾期次数_企业征信报告': '贷款逾期次数',
            '最长逾期天数_企业征信报告': '最长逾期天数',
            '是否有不良记录_企业征信报告': '有不良记录',
            '是否涉诉_企业征信报告': '涉诉情况',
            '纳税信用等级_税务记录': '纳税信用等级',
            '个人月收入(万元)_企业主信用': '企业主月收入',
            '个人资产(万元)_企业主信用': '企业主资产',
            '教育程度_企业主信用': '教育程度',
            '职业背景_企业主信用': '职业背景',
            '是否有失信记录_企业主信用': '有失信记录',
            '企业主信用评分_企业主信用': '企业主信用评分',
            '行业增长率_行业经营状况': '行业增长率',
            '行业竞争程度_行业经营状况': '行业竞争程度',
            '行业风险等级_行业经营状况': '行业风险等级',
            '市场风险_企业经营风险评估': '市场风险',
            '经营风险_企业经营风险评估': '经营风险',
            '财务风险_企业经营风险评估': '财务风险',
            '风险等级_企业经营风险评估': '风险等级',
            '月均收入(万元)_银行流水': '月均收入',
            '月均支出(万元)_银行流水': '月均支出',
            '账户余额(万元)_银行流水': '账户余额',
            '流水稳定性_银行流水': '流水稳定性',
            '供应链稳定性评级_上下游合作情况': '供应链稳定性',
            '主要客户数量_上下游合作情况': '主要客户数量',
            '主要供应商数量_上下游合作情况': '主要供应商数量',
            '是否有逾期': '是否逾期'
        }
        
        self._log_step("数据处理器初始化完成")
    
    def _log_step(self, message, details=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'details': details
        }
        self.processing_log.append(log_entry)
        print(f"[{timestamp}] {message}")
        if details:
            print(f"  详情: {details}")
    
    def load_and_aggregate_data(self):
        self._log_step("=== 开始数据加载和聚合 ===")
        
        main_file = self.files_config['合并信贷数据']
        if os.path.exists(main_file):
            self.merged_data = pd.read_excel(main_file)
            self._log_step(f"加载主数据集: {main_file}", f"形状: {self.merged_data.shape}")
            self.merged_data.columns = [col.replace('\ufeff', '') for col in self.merged_data.columns]
        else:
            raise FileNotFoundError(f"主数据文件不存在: {main_file}")
        
        loaded_files = []
        for name, filename in self.files_config.items():
            if name == '合并信贷数据':
                continue
                
            if os.path.exists(filename):
                try:
                    df = pd.read_excel(filename)
                    self._log_step(f"加载数据集: {filename}")
                    
                    id_col = self._find_id_column(df)
                    if id_col:
                        unique_ids = df[id_col].nunique()
                        total_rows = len(df)
                        
                        if total_rows > unique_ids:
                            df_aggregated = self._aggregate_multiple_records(df, id_col, name)
                            self._log_step(f"聚合多条记录", f"{total_rows} → {len(df_aggregated)} 行")
                        else:
                            df_aggregated = df
                        
                        suffix = f"_{name.replace('数据', '').replace('信息', '')}"
                        df_renamed = df_aggregated.rename(columns={col: f"{col}{suffix}" if col != id_col else col 
                                                          for col in df_aggregated.columns})
                        
                        self.merged_data = pd.merge(self.merged_data, df_renamed, 
                                                  left_on='企业ID', right_on=id_col, 
                                                  how='left', suffixes=('', suffix))
                        loaded_files.append(filename)
                        
                except Exception as e:
                    self._log_step(f"加载失败: {filename}", str(e))
            else:
                self._log_step(f"文件不存在: {filename}")
        
        self._log_step("数据加载完成", f"最终形状: {self.merged_data.shape}, 成功加载: {len(loaded_files)} 个文件")
        return self.merged_data
    
    def _find_id_column(self, df):
        possible_id_cols = ['企业ID', 'ID', 'id', '编号']
        for col in df.columns:
            if any(id_name in col for id_name in possible_id_cols):
                return col
        return None
    
    def _aggregate_multiple_records(self, df, id_col, dataset_name):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
        if id_col in text_cols:
            text_cols.remove(id_col)
        
        agg_dict = {}
        
        for col in numeric_cols:
            if any(keyword in col for keyword in ['金额', '收入', '支出', '余额', '资产', '负债', '利润']):
                if '流水' in dataset_name or '银行' in dataset_name:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'
            elif any(keyword in col for keyword in ['率', '比例', '百分比']):
                agg_dict[col] = 'mean'
            elif any(keyword in col for keyword in ['次数', '笔数', '数量']):
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'mean'
        
        for col in text_cols:
            agg_dict[col] = 'first'
        
        df_agg = df.groupby(id_col).agg(agg_dict).reset_index()
        return df_agg
    
    def clean_features(self):
        self._log_step("=== 特征清理 ===")
        
        initial_cols = self.merged_data.shape[1]
        
        cols_to_drop = []
        for col in self.features_to_remove:
            matching_cols = [c for c in self.merged_data.columns if col in c]
            cols_to_drop.extend(matching_cols)
        
        cols_to_drop = list(set(cols_to_drop))
        
        if cols_to_drop:
            self.merged_data = self.merged_data.drop(columns=cols_to_drop)
            self._log_step(f"删除无用特征", f"删除了 {len(cols_to_drop)} 个特征")
        
        duplicate_cols = []
        for i, col1 in enumerate(self.merged_data.columns):
            for col2 in self.merged_data.columns[i+1:]:
                if self.merged_data[col1].equals(self.merged_data[col2]):
                    duplicate_cols.append(col2)
        
        if duplicate_cols:
            self.merged_data = self.merged_data.drop(columns=duplicate_cols)
            self._log_step(f"删除重复列", f"删除了 {len(duplicate_cols)} 个重复列")
        
        constant_cols = []
        for col in self.merged_data.columns:
            if self.merged_data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            self.merged_data = self.merged_data.drop(columns=constant_cols)
            self._log_step(f"删除常数列", f"删除了 {len(constant_cols)} 个常数列")
        
        final_cols = self.merged_data.shape[1]
        self._log_step("特征清理完成", f"{initial_cols} → {final_cols} 列")
        
        return self.merged_data
    
    def handle_missing_values(self):
        self._log_step("=== 处理缺失值 ===")
        
        missing_summary = self.merged_data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            self._log_step(f"发现缺失值", f"{len(missing_cols)} 列有缺失值")
            for col, count in missing_cols.items():
                if self.merged_data[col].dtype == 'object':
                    mode_val = self.merged_data[col].mode()
                    if len(mode_val) > 0:
                        self.merged_data[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.merged_data[col].fillna('未知', inplace=True)
                else:
                    median_val = self.merged_data[col].median()
                    self.merged_data[col].fillna(median_val, inplace=True)
        else:
            self._log_step("缺失值检查", "无缺失值")
        
        return self.merged_data
    
    def encode_categorical_features(self):
        self._log_step("=== 分类特征标签编码 ===")
        
        encoded_count = 0
        self.encoded_features = []
        encoding_details = {}
        
        for feature_name in self.categorical_features:
            matching_cols = [col for col in self.merged_data.columns 
                           if feature_name in col and self.merged_data[col].dtype == 'object']
            
            for col in matching_cols:
                unique_values = self.merged_data[col].unique()
                encoded_values = self._apply_custom_encoding(col, unique_values)
                
                if encoded_values:
                    self.merged_data[col] = self.merged_data[col].map(encoded_values)
                    encoding_details[col] = {'type': 'custom', 'mapping': encoded_values}
                else:
                    le = LabelEncoder()
                    self.merged_data[col] = le.fit_transform(self.merged_data[col])
                    self.label_encoders[col] = le
                    encoding_details[col] = {'type': 'label', 'classes': le.classes_.tolist()}
                
                self.encoded_features.append(col)
                encoded_count += 1
        
        self._log_step("分类特征编码完成", f"编码了 {encoded_count} 个分类特征")
        return self.merged_data
    
    def _apply_custom_encoding(self, col_name, unique_values):
        if len(unique_values) == 2:
            for val in unique_values:
                if val in self.custom_encoding_rules['binary']:
                    encoding_map = {}
                    for val in unique_values:
                        if val in self.custom_encoding_rules['binary']:
                            encoding_map[val] = self.custom_encoding_rules['binary'][val]
                        else:
                            existing_vals = [v for v in unique_values if v in self.custom_encoding_rules['binary']]
                            if existing_vals:
                                existing_code = self.custom_encoding_rules['binary'][existing_vals[0]]
                                encoding_map[val] = 1 - existing_code
                    return encoding_map
        
        if '信用' in col_name or '等级' in col_name:
            if all(val in self.custom_encoding_rules['credit_grade'] for val in unique_values):
                return {val: self.custom_encoding_rules['credit_grade'][val] for val in unique_values}
        
        if '教育' in col_name:
            if all(val in self.custom_encoding_rules['education'] for val in unique_values):
                return {val: self.custom_encoding_rules['education'][val] for val in unique_values}
        
        if all(val in self.custom_encoding_rules['level_high_low'] for val in unique_values):
            return {val: self.custom_encoding_rules['level_high_low'][val] for val in unique_values}
        
        return None
    
    def feature_selection(self, importance_threshold=0.001):
        self._log_step("=== 特征选择 ===")
        
        target_col = '是否有逾期'
        if target_col not in self.merged_data.columns:
            self._log_step("警告", "未找到目标变量，跳过特征选择")
            return self.merged_data
        
        X = self.merged_data.drop(columns=[target_col])
        y = self.merged_data[target_col]
        
        original_features = X.shape[1]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = self.feature_importance[
            self.feature_importance['importance'] > importance_threshold
        ]['feature'].tolist()
        
        self.merged_data = self.merged_data[selected_features + [target_col]]
        
        self._log_step("特征选择完成", 
                      f"阈值: {importance_threshold}, {original_features} → {len(selected_features)} 特征")
        
        return self.merged_data
    
    def optimize_standardization(self):
        self._log_step("=== 优化标准化处理 ===")
        
        target_col = '是否有逾期'
        if target_col in self.merged_data.columns:
            features_to_standardize = []
            encoded_features_found = []
            
            for col in self.merged_data.columns:
                if col == target_col:
                    continue
                
                is_encoded = any(encoded_col in col for encoded_col in self.encoded_features)
                
                if not is_encoded and self.merged_data[col].dtype in ['int64', 'float64']:
                    features_to_standardize.append(col)
                else:
                    encoded_features_found.append(col)
            
            if features_to_standardize:
                self.merged_data[features_to_standardize] = self.scaler.fit_transform(
                    self.merged_data[features_to_standardize])
                self._log_step("连续特征标准化", f"标准化了 {len(features_to_standardize)} 个连续特征")
            
            if encoded_features_found:
                self._log_step("编码特征保持原值", f"{len(encoded_features_found)} 个编码特征未标准化")
        
        return self.merged_data
    
    def simplify_chinese_names(self):
        self._log_step("=== 简化中文列名 ===")
        
        new_column_mapping = {}
        
        for old_col in self.merged_data.columns:
            if old_col in self.chinese_name_mapping:
                new_column_mapping[old_col] = self.chinese_name_mapping[old_col]
            else:
                new_col = old_col
                
                suffixes_to_remove = [
                    '_企业基本信息', '_财务报表数据', '_信贷申请数据', '_税务记录',
                    '_银行流水', '_企业主信用数据', '_行业经营状况', '_企业征信报告',
                    '_上下游合作情况', '_企业经营风险评估', '_企业基本', '_财务报表',
                    '_信贷申请', '_企业主信用', '_行业经营', '_企业征信', '_上下游合作',
                    '_企业经营风险', '_风险评估'
                ]
                
                for suffix in suffixes_to_remove:
                    if new_col.endswith(suffix):
                        new_col = new_col.replace(suffix, '')
                        break
                
                if '(万元)' in new_col:
                    new_col = new_col.replace('(万元)', '')
                if '(月)' in new_col:
                    new_col = new_col.replace('(月)', '')
                
                new_column_mapping[old_col] = new_col
        
        used_names = set()
        final_mapping = {}
        for old_col, new_col in new_column_mapping.items():
            if new_col in used_names:
                counter = 1
                while f"{new_col}_{counter}" in used_names:
                    counter += 1
                new_col = f"{new_col}_{counter}"
            used_names.add(new_col)
            final_mapping[old_col] = new_col
        
        self.merged_data = self.merged_data.rename(columns=final_mapping)
        
        self._log_step("中文列名简化完成", f"简化了 {len(final_mapping)} 个列名")
        return self.merged_data
    
    def validate_data_quality(self):
        self._log_step("=== 数据质量验证 ===")
        
        quality_report = {
            'basic_info': {
                'total_rows': len(self.merged_data),
                'total_columns': len(self.merged_data.columns),
                'missing_values': self.merged_data.isnull().sum().sum(),
                'duplicate_rows': self.merged_data.duplicated().sum(),
                'infinite_values': np.isinf(self.merged_data.select_dtypes(include=[np.number])).sum().sum()
            },
            'target_distribution': {}
        }
        
        target_col = '是否有逾期'
        if target_col in self.merged_data.columns:
            quality_report['target_distribution'] = self.merged_data[target_col].value_counts().to_dict()
        
        self._log_step("数据质量验证完成", 
                      f"企业: {quality_report['basic_info']['total_rows']}, 特征: {quality_report['basic_info']['total_columns']-1}")
        
        return quality_report
    
    def save_processed_data(self, filename='../输出结果/processed_data_final.xlsx'):
        self._log_step("=== 保存处理后的数据 ===")
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            self.merged_data.to_excel(output_path, index=False)
            self.final_dataset = self.merged_data.copy()
            
            self._log_step("数据保存完成", 
                          f"文件: {output_path}, 形状: {self.merged_data.shape}")
            
            return output_path
            
        except Exception as e:
            self._log_step("数据保存失败", str(e))
            raise e
    
    def generate_documentation(self):
        self._log_step("=== 生成处理文档 ===")
        
        quality_report = self.validate_data_quality()
        doc_content = self._create_comprehensive_documentation(quality_report)
        
        doc_filename = os.path.join(self.output_dir, '../文档/数据处理说明文档.md')
        with open(doc_filename, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        log_filename = os.path.join(self.output_dir, 'processing_log.json')
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
        
        if self.feature_importance is not None:
            importance_filename = os.path.join(self.output_dir, '../输出结果/feature_importance.xlsx')
            self.feature_importance.to_excel(importance_filename, index=False)
        
        self._log_step("文档生成完成", f"主文档: {doc_filename}")
        return doc_filename
    
    def _create_comprehensive_documentation(self, quality_report):
        target_col = '是否有逾期'
        target_info = "未找到目标变量"
        if target_col in self.merged_data.columns:
            target_info = f"目标变量 '{target_col}' 有 {self.merged_data[target_col].nunique()} 个唯一值"
        
        encoded_examples = []
        for feature in self.encoded_features[:5]:
            if feature in self.merged_data.columns:
                encoded_examples.append(f"- {feature}: {self.merged_data[feature].nunique()} 个类别")
        
        top_features = []
        if self.feature_importance is not None:
            for _, row in self.feature_importance.head(10).iterrows():
                top_features.append(f"| {row['feature']} | {row['importance']:.4f} |")
        
        doc_content = f"""# 中小企业信贷风险评估 - 数据处理说明文档


**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  


## 📊 数据集基本信息

### 原始数据源
本项目整合了以下10个数据源：

| 序号 | 数据源 | 文件名 | 主要内容 |
|------|--------|--------|----------|
| 1 | 企业基本信息 | 01_企业基本信息.xlsx | 注册资本、成立年限、员工数等 |
| 2 | 信贷申请数据 | 02_信贷申请数据.xlsx | 申请金额、期限、担保方式等 |
| 3 | 财务报表数据 | 03_财务报表数据.xlsx | 资产负债、收入利润、财务比率等 |
| 4 | 税务记录 | 04_税务记录.xlsx | 纳税信用等级、税务合规情况等 |
| 5 | 银行流水 | 05_银行流水.xlsx | 月均收支、账户余额、流水稳定性等 |
| 6 | 企业主信用数据 | 06_企业主信用数据.xlsx | 个人收入资产、教育背景、信用记录等 |
| 7 | 行业经营状况 | 07_行业经营状况.xlsx | 行业增长率、竞争程度、风险等级等 |
| 8 | 企业征信报告 | 08_企业征信报告.xlsx | 信用评分、逾期记录、不良记录等 |
| 9 | 上下游合作情况 | 09_上下游合作情况.xlsx | 供应链稳定性、客户供应商数量等 |
| 10 | 企业经营风险评估 | 10_企业经营风险评估.xlsx | 市场风险、经营风险、财务风险等 |


| **企业数量** | {quality_report['basic_info']['total_rows']} |
| **特征数量** | {quality_report['basic_info']['total_columns']-1} |
| **总列数** | {quality_report['basic_info']['total_columns']} |
| **目标变量** | 贷款逾期次数 |
| **数据文件大小** | 约 0.11 MB |


| **缺失值** | {quality_report['basic_info']['missing_values']} |
| **重复行** | {quality_report['basic_info']['duplicate_rows']} |
| **无穷值** | {quality_report['basic_info']['infinite_values']} |

{target_info}

{chr(10).join(encoded_examples)}


{chr(10).join(top_features)}


**文档生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return doc_content
    
    def run_complete_pipeline(self, save_filename='../输出结果/processed_data_final.xlsx'):
        print("🚀 开始完整数据预处理管道")
        print("="*80)
        
        try:
            self.load_and_aggregate_data()
            self.clean_features()
            self.handle_missing_values()
            self.encode_categorical_features()
            self.feature_selection()
            self.optimize_standardization()
            self.simplify_chinese_names()
            quality_report = self.validate_data_quality()
            data_file = self.save_processed_data(save_filename)
            doc_file = self.generate_documentation()
            
            print("\n" + "="*80)
            print("🎉 完整数据预处理管道执行完成！")
            print(f"📊 最终数据集: {data_file}")
            print(f"📋 说明文档: {doc_file}")
            print(f"📈 数据规模: {self.merged_data.shape[0]} 企业 × {self.merged_data.shape[1]-1} 特征")
            
            return data_file, doc_file
            
        except Exception as e:
            self._log_step("处理过程出现错误", str(e))
            raise e

def main():
    print("中小企业信贷风险评估 - 数据预处理系统")
    print("版本: 3.0 (中文优化版)")
    print("="*80)
    
    processor = ComprehensiveDataProcessor()
    data_file, doc_file = processor.run_complete_pipeline()
    
    print(f"\n✅ 处理完成！")
    print(f"📊 数据文件: {data_file}")
    print(f"📋 文档文件: {doc_file}")
    
    return data_file, doc_file

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SME信贷风险评估系统 - 数据预处理模块
包含多维度数据源整合、高级数据清洗、特征工程等功能
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings('ignore')

class AdvancedDataPreprocessor:
    """
    高级数据预处理器
    支持多维度数据源整合、智能数据清洗、特征工程等功能
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.processed_data = None
        self.feature_metadata = {}
        self.processing_log = []
        self.scalers = {}
        self.imputers = {}
        
        # 设置日志
        self._setup_logging()
        
        # 数据源配置
        self.data_source_config = {
            '企业基本信息': {
                'path': '../原始数据/01_企业基本信息.xlsx',
                'key_column': '企业ID',
                'priority': 1,
                'required': True
            },
            '财务报表数据': {
                'path': '../原始数据/03_财务报表数据.xlsx',
                'key_column': '企业ID',
                'priority': 2,
                'required': True
            },
            '信贷申请数据': {
                'path': '../原始数据/02_信贷申请数据.xlsx',
                'key_column': '企业ID',
                'priority': 3,
                'required': True
            },
            '企业征信报告': {
                'path': '../原始数据/08_企业征信报告.xlsx',
                'key_column': '企业ID',
                'priority': 4,
                'required': True
            },
            '税务记录': {
                'path': '../原始数据/04_税务记录.xlsx',
                'key_column': '企业ID',
                'priority': 5,
                'required': False
            },
            '银行流水': {
                'path': '../原始数据/05_银行流水.xlsx',
                'key_column': '企业ID',
                'priority': 6,
                'required': False
            },
            '企业主信用数据': {
                'path': '../原始数据/06_企业主信用数据.xlsx',
                'key_column': '企业ID',
                'priority': 7,
                'required': False
            },
            '行业经营状况': {
                'path': '../原始数据/07_行业经营状况.xlsx',
                'key_column': '企业ID',
                'priority': 8,
                'required': False
            },
            '上下游合作情况': {
                'path': '../原始数据/09_上下游合作情况.xlsx',
                'key_column': '企业ID',
                'priority': 9,
                'required': False
            },
            '企业经营风险评估': {
                'path': '../原始数据/10_企业经营风险评估.xlsx',
                'key_column': '企业ID',
                'priority': 10,
                'required': False
            }
        }
        
        # 特征类型定义
        self.feature_types = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'ordinal': [],
            'datetime': [],
            'text': []
        }
        
        # 编码规则
        self.encoding_rules = {
            'binary_mapping': {
                '是': 1, '否': 0, '有': 1, '无': 0,
                'True': 1, 'False': 0, '男': 1, '女': 0,
                'Y': 1, 'N': 0, 'yes': 1, 'no': 0
            },
            'ordinal_mapping': {
                '风险等级': {'低': 1, '中': 2, '高': 3},
                '信用等级': {'A': 4, 'B': 3, 'C': 2, 'D': 1},
                '教育程度': {'博士': 5, '硕士': 4, '本科': 3, '专科': 2, '高中及以下': 1},
                '企业规模': {'大型': 4, '中型': 3, '小型': 2, '微型': 1}
            }
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../输出结果/preprocessing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_multi_source_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载多维度数据源
        
        Returns:
            包含所有数据源的字典
        """
        self.logger.info("开始加载多维度数据源...")
        loaded_sources = {}
        
        for source_name, config in self.data_source_config.items():
            try:
                if os.path.exists(config['path']):
                    df = pd.read_excel(config['path'])
                    loaded_sources[source_name] = df
                    self.logger.info(f"成功加载 {source_name}: {df.shape}")
                    
                    # 记录数据源信息
                    self.feature_metadata[source_name] = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'missing_ratio': df.isnull().sum() / len(df)
                    }
                else:
                    if config['required']:
                        raise FileNotFoundError(f"必需的数据源文件不存在: {config['path']}")
                    else:
                        self.logger.warning(f"可选数据源文件不存在: {config['path']}")
                        
            except Exception as e:
                self.logger.error(f"加载数据源 {source_name} 失败: {str(e)}")
                if config['required']:
                    raise
        
        self.data_sources = loaded_sources
        return loaded_sources
    
    def intelligent_data_integration(self) -> pd.DataFrame:
        """
        智能数据整合
        基于企业ID进行多表关联，处理重复字段和冲突数据
        
        Returns:
            整合后的数据框
        """
        self.logger.info("开始智能数据整合...")
        
        if not self.data_sources:
            raise ValueError("请先加载数据源")
        
        # 按优先级排序数据源
        sorted_sources = sorted(
            self.data_sources.items(),
            key=lambda x: self.data_source_config[x[0]]['priority']
        )
        
        # 从最高优先级数据源开始
        base_name, base_df = sorted_sources[0]
        integrated_df = base_df.copy()
        key_column = self.data_source_config[base_name]['key_column']
        
        self.logger.info(f"以 {base_name} 作为基础数据源")
        
        # 逐步整合其他数据源
        for source_name, source_df in sorted_sources[1:]:
            try:
                source_key = self.data_source_config[source_name]['key_column']
                
                # 处理列名冲突
                source_df_renamed = self._resolve_column_conflicts(
                    source_df, integrated_df, source_name
                )
                
                # 执行左连接
                integrated_df = pd.merge(
                    integrated_df, source_df_renamed,
                    left_on=key_column, right_on=source_key,
                    how='left', suffixes=('', f'_{source_name}')
                )
                
                self.logger.info(f"整合 {source_name}: {integrated_df.shape}")
                
            except Exception as e:
                self.logger.error(f"整合数据源 {source_name} 失败: {str(e)}")
                continue
        
        self.processed_data = integrated_df
        self.logger.info(f"数据整合完成，最终形状: {integrated_df.shape}")
        
        return integrated_df
    
    def _resolve_column_conflicts(self, source_df: pd.DataFrame, 
                                target_df: pd.DataFrame, 
                                source_name: str) -> pd.DataFrame:
        """
        解决列名冲突
        
        Args:
            source_df: 源数据框
            target_df: 目标数据框
            source_name: 数据源名称
            
        Returns:
            重命名后的数据框
        """
        source_df_copy = source_df.copy()
        
        # 找出冲突的列名
        conflicting_cols = set(source_df.columns) & set(target_df.columns)
        key_column = self.data_source_config[source_name]['key_column']
        
        # 排除关键列
        conflicting_cols.discard(key_column)
        
        # 重命名冲突列
        rename_dict = {}
        for col in conflicting_cols:
            new_name = f"{col}_{source_name}"
            rename_dict[col] = new_name
            self.logger.info(f"列名冲突解决: {col} -> {new_name}")
        
        if rename_dict:
            source_df_copy = source_df_copy.rename(columns=rename_dict)
        
        return source_df_copy
    
    def advanced_data_cleaning(self) -> pd.DataFrame:
        """
        高级数据清洗
        包括异常值检测、数据类型优化、一致性检查等
        
        Returns:
            清洗后的数据框
        """
        self.logger.info("开始高级数据清洗...")
        
        if self.processed_data is None:
            raise ValueError("请先进行数据整合")
        
        df = self.processed_data.copy()
        
        # 1. 数据类型优化
        df = self._optimize_data_types(df)
        
        # 2. 异常值检测和处理
        df = self._handle_outliers(df)
        
        # 3. 数据一致性检查
        df = self._consistency_check(df)
        
        # 4. 重复数据处理
        df = self._handle_duplicates(df)
        
        # 5. 特征类型分类
        self._classify_feature_types(df)
        
        self.processed_data = df
        self.logger.info(f"数据清洗完成，最终形状: {df.shape}")
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型"""
        self.logger.info("优化数据类型...")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为数值类型
                try:
                    # 清理数值字符串
                    cleaned_series = df[col].astype(str).str.replace(',', '').str.replace('，', '')
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # 如果转换成功率高于80%，则采用数值类型
                    if numeric_series.notna().sum() / len(df) > 0.8:
                        df[col] = numeric_series
                        self.logger.info(f"列 {col} 转换为数值类型")
                        continue
                except:
                    pass
                
                # 尝试转换为日期类型
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    if date_series.notna().sum() / len(df) > 0.8:
                        df[col] = date_series
                        self.logger.info(f"列 {col} 转换为日期类型")
                        continue
                except:
                    pass
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        self.logger.info("检测和处理异常值...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].notna().sum() > 0:
                # 使用IQR方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if len(outliers) > 0:
                    outlier_ratio = len(outliers) / df[col].notna().sum()
                    
                    if outlier_ratio < 0.05:  # 异常值比例小于5%，进行处理
                        # 使用边界值替换异常值
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        df.loc[df[col] > upper_bound, col] = upper_bound
                        
                        self.logger.info(f"列 {col} 处理了 {len(outliers)} 个异常值")
        
        return df
    
    def _consistency_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据一致性检查"""
        self.logger.info("进行数据一致性检查...")
        
        # 检查逻辑一致性
        # 例如：资产负债率应该在0-100%之间
        if '资产负债率' in df.columns:
            df.loc[df['资产负债率'] > 100, '资产负债率'] = 100
            df.loc[df['资产负债率'] < 0, '资产负债率'] = 0
        
        # 检查日期一致性
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            # 移除未来日期
            future_mask = df[col] > datetime.now()
            if future_mask.sum() > 0:
                df.loc[future_mask, col] = pd.NaT
                self.logger.warning(f"列 {col} 移除了 {future_mask.sum()} 个未来日期")
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理重复数据"""
        self.logger.info("处理重复数据...")
        
        initial_shape = df.shape
        
        # 基于企业ID去重，保留第一条记录
        if '企业ID' in df.columns:
            df = df.drop_duplicates(subset=['企业ID'], keep='first')
        else:
            df = df.drop_duplicates()
        
        removed_count = initial_shape[0] - df.shape[0]
        if removed_count > 0:
            self.logger.info(f"移除了 {removed_count} 条重复记录")
        
        return df
    
    def _classify_feature_types(self, df: pd.DataFrame):
        """分类特征类型"""
        self.logger.info("分类特征类型...")
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                unique_count = df[col].nunique()
                if unique_count == 2:
                    self.feature_types['binary'].append(col)
                elif unique_count <= 10 and df[col].dtype == 'int64':
                    self.feature_types['ordinal'].append(col)
                else:
                    self.feature_types['numerical'].append(col)
            elif df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count <= 20:
                    self.feature_types['categorical'].append(col)
                else:
                    self.feature_types['text'].append(col)
            elif df[col].dtype.name.startswith('datetime'):
                self.feature_types['datetime'].append(col)
        
        self.logger.info(f"特征类型分类完成: {dict(self.feature_types)}")
    
    def intelligent_missing_value_handling(self) -> pd.DataFrame:
        """
        智能缺失值处理
        根据特征类型和缺失模式选择最佳填充策略
        
        Returns:
            处理后的数据框
        """
        self.logger.info("开始智能缺失值处理...")
        
        if self.processed_data is None:
            raise ValueError("请先进行数据清洗")
        
        df = self.processed_data.copy()
        
        # 分析缺失模式
        missing_info = self._analyze_missing_patterns(df)
        
        # 数值特征缺失值处理
        numerical_cols = self.feature_types['numerical'] + self.feature_types['binary'] + self.feature_types['ordinal']
        if numerical_cols:
            df = self._handle_numerical_missing(df, numerical_cols)
        
        # 分类特征缺失值处理
        categorical_cols = self.feature_types['categorical']
        if categorical_cols:
            df = self._handle_categorical_missing(df, categorical_cols)
        
        # 日期特征缺失值处理
        datetime_cols = self.feature_types['datetime']
        if datetime_cols:
            df = self._handle_datetime_missing(df, datetime_cols)
        
        self.processed_data = df
        self.logger.info("缺失值处理完成")
        
        return df
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """分析缺失模式"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            
            missing_info[col] = {
                'count': missing_count,
                'ratio': missing_ratio,
                'strategy': self._determine_imputation_strategy(col, missing_ratio)
            }
        
        return missing_info
    
    def _determine_imputation_strategy(self, column: str, missing_ratio: float) -> str:
        """确定填充策略"""
        if missing_ratio == 0:
            return 'no_missing'
        elif missing_ratio > 0.7:
            return 'drop_column'
        elif missing_ratio > 0.3:
            return 'knn_imputation'
        else:
            return 'simple_imputation'
    
    def _handle_numerical_missing(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """处理数值特征缺失值"""
        for col in cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                missing_ratio = df[col].isnull().sum() / len(df)
                
                if missing_ratio > 0.3:
                    # 使用KNN填充
                    imputer = KNNImputer(n_neighbors=5)
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                else:
                    # 使用中位数填充
                    imputer = SimpleImputer(strategy='median')
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                
                self.logger.info(f"数值列 {col} 缺失值处理完成")
        
        return df
    
    def _handle_categorical_missing(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """处理分类特征缺失值"""
        for col in cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                # 使用众数填充
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                
                self.logger.info(f"分类列 {col} 使用众数 '{mode_value}' 填充缺失值")
        
        return df
    
    def _handle_datetime_missing(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """处理日期特征缺失值"""
        for col in cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                # 使用中位数日期填充
                median_date = df[col].median()
                df[col] = df[col].fillna(median_date)
                
                self.logger.info(f"日期列 {col} 使用中位数日期填充缺失值")
        
        return df
    
    def save_processed_data(self, output_path: str = '../输出结果/processed_data_enhanced.xlsx'):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        if self.processed_data is not None:
            self.processed_data.to_excel(output_path, index=False)
            self.logger.info(f"处理后的数据已保存到: {output_path}")
            
            # 保存元数据
            metadata_path = output_path.replace('.xlsx', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'feature_types': self.feature_types,
                    'feature_metadata': self.feature_metadata,
                    'processing_log': self.processing_log
                }, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"元数据已保存到: {metadata_path}")
        else:
            raise ValueError("没有可保存的处理后数据")
    
    def get_processing_summary(self) -> Dict:
        """
        获取处理摘要
        
        Returns:
            处理摘要信息
        """
        if self.processed_data is None:
            return {}
        
        return {
            'data_shape': self.processed_data.shape,
            'feature_types': {k: len(v) for k, v in self.feature_types.items()},
            'missing_values': self.processed_data.isnull().sum().to_dict(),
            'data_sources_loaded': len(self.data_sources),
            'processing_steps': len(self.processing_log)
        }


def main():
    """主函数 - 演示数据预处理流程"""
    try:
        # 创建预处理器实例
        preprocessor = AdvancedDataPreprocessor()
        
        # 加载多维度数据源
        data_sources = preprocessor.load_multi_source_data()
        print(f"成功加载 {len(data_sources)} 个数据源")
        
        # 智能数据整合
        integrated_data = preprocessor.intelligent_data_integration()
        print(f"数据整合完成，形状: {integrated_data.shape}")
        
        # 高级数据清洗
        cleaned_data = preprocessor.advanced_data_cleaning()
        print(f"数据清洗完成，形状: {cleaned_data.shape}")
        
        # 智能缺失值处理
        final_data = preprocessor.intelligent_missing_value_handling()
        print(f"缺失值处理完成，形状: {final_data.shape}")
        
        # 保存处理后的数据
        preprocessor.save_processed_data()
        
        # 打印处理摘要
        summary = preprocessor.get_processing_summary()
        print("\n处理摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n数据预处理完成！")
        
    except Exception as e:
        print(f"数据预处理过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
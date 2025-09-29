#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SME信贷风险评估系统 - 特征工程模块
实现智能特征提取、特征变换、特征选择和特征组合
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import itertools

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    高级特征工程器
    支持自动化特征生成、特征选择、特征变换和特征组合
    """
    
    def __init__(self, output_dir: str = '../输出结果/feature_engineering'):
        """
        初始化特征工程器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.feature_transformers = {}
        self.feature_selectors = {}
        self.generated_features = {}
        self.feature_importance_scores = {}
        
        # 设置日志
        self._setup_logging()
        
        # 特征工程配置
        self.config = {
            'numerical_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'text_features': [],
            'target_column': 'target',
            'feature_selection_methods': ['variance', 'correlation', 'mutual_info', 'rfe'],
            'scaling_methods': ['standard', 'minmax', 'robust'],
            'encoding_methods': ['onehot', 'label', 'target'],
            'polynomial_degree': 2,
            'interaction_threshold': 0.1,
            'correlation_threshold': 0.95
        }
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'feature_engineering.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def auto_feature_engineering(self, df: pd.DataFrame, target_column: str = 'target') -> pd.DataFrame:
        """
        自动化特征工程流程
        
        Args:
            df: 输入数据框
            target_column: 目标列名
            
        Returns:
            特征工程后的数据框
        """
        self.logger.info("开始自动化特征工程流程")
        
        # 更新配置
        self.config['target_column'] = target_column
        
        # 1. 特征类型识别
        feature_types = self._identify_feature_types(df)
        self.config.update(feature_types)
        
        # 2. 基础特征清理
        df_cleaned = self._basic_feature_cleaning(df)
        
        # 3. 生成新特征
        df_with_new_features = self._generate_new_features(df_cleaned)
        
        # 4. 特征变换
        df_transformed = self._apply_feature_transformations(df_with_new_features)
        
        # 5. 特征选择
        df_selected = self._feature_selection(df_transformed, target_column)
        
        # 6. 特征缩放
        df_scaled = self._feature_scaling(df_selected, target_column)
        
        # 7. 生成特征工程报告
        self._generate_feature_engineering_report(df, df_scaled)
        
        self.logger.info("自动化特征工程流程完成")
        
        return df_scaled
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Dict:
        """识别特征类型"""
        self.logger.info("识别特征类型")
        
        numerical_features = []
        categorical_features = []
        datetime_features = []
        text_features = []
        
        for column in df.columns:
            if column == self.config['target_column']:
                continue
                
            dtype = df[column].dtype
            unique_count = df[column].nunique()
            total_count = len(df[column])
            
            # 数值型特征
            if dtype in ['int64', 'float64']:
                if unique_count > 10 and unique_count / total_count > 0.05:
                    numerical_features.append(column)
                else:
                    categorical_features.append(column)
            
            # 日期时间特征
            elif dtype == 'datetime64[ns]' or 'date' in column.lower() or 'time' in column.lower():
                datetime_features.append(column)
            
            # 文本特征
            elif dtype == 'object':
                avg_length = df[column].astype(str).str.len().mean()
                if avg_length > 20:  # 平均长度大于20认为是文本
                    text_features.append(column)
                else:
                    categorical_features.append(column)
        
        feature_types = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'datetime_features': datetime_features,
            'text_features': text_features
        }
        
        self.logger.info(f"特征类型识别完成: {feature_types}")
        
        return feature_types
    
    def _basic_feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础特征清理"""
        self.logger.info("执行基础特征清理")
        
        df_cleaned = df.copy()
        
        # 处理缺失值
        for column in df_cleaned.columns:
            if column == self.config['target_column']:
                continue
                
            missing_ratio = df_cleaned[column].isnull().sum() / len(df_cleaned)
            
            if missing_ratio > 0.8:
                # 缺失值过多，删除特征
                df_cleaned.drop(column, axis=1, inplace=True)
                self.logger.info(f"删除缺失值过多的特征: {column} (缺失率: {missing_ratio:.2%})")
            
            elif missing_ratio > 0:
                if column in self.config['numerical_features']:
                    # 数值型特征用中位数填充
                    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                else:
                    # 分类特征用众数填充
                    mode_value = df_cleaned[column].mode()
                    if len(mode_value) > 0:
                        df_cleaned[column].fillna(mode_value[0], inplace=True)
                    else:
                        df_cleaned[column].fillna('Unknown', inplace=True)
        
        # 更新特征类型列表（移除已删除的特征）
        remaining_columns = set(df_cleaned.columns)
        self.config['numerical_features'] = [f for f in self.config['numerical_features'] if f in remaining_columns]
        self.config['categorical_features'] = [f for f in self.config['categorical_features'] if f in remaining_columns]
        self.config['datetime_features'] = [f for f in self.config['datetime_features'] if f in remaining_columns]
        self.config['text_features'] = [f for f in self.config['text_features'] if f in remaining_columns]
        
        return df_cleaned
    
    def _generate_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成新特征"""
        self.logger.info("生成新特征")
        
        df_new = df.copy()
        new_features_info = {}
        
        # 1. 数值特征的数学变换
        df_new, math_features = self._generate_mathematical_features(df_new)
        new_features_info['mathematical_features'] = math_features
        
        # 2. 统计特征
        df_new, stat_features = self._generate_statistical_features(df_new)
        new_features_info['statistical_features'] = stat_features
        
        # 3. 交互特征
        df_new, interaction_features = self._generate_interaction_features(df_new)
        new_features_info['interaction_features'] = interaction_features
        
        # 4. 时间特征
        if self.config['datetime_features']:
            df_new, time_features = self._generate_time_features(df_new)
            new_features_info['time_features'] = time_features
        
        # 5. 聚合特征
        df_new, agg_features = self._generate_aggregation_features(df_new)
        new_features_info['aggregation_features'] = agg_features
        
        # 6. 比率特征
        df_new, ratio_features = self._generate_ratio_features(df_new)
        new_features_info['ratio_features'] = ratio_features
        
        # 7. 分箱特征
        df_new, binning_features = self._generate_binning_features(df_new)
        new_features_info['binning_features'] = binning_features
        
        self.generated_features = new_features_info
        
        self.logger.info(f"新特征生成完成，共生成 {len(df_new.columns) - len(df.columns)} 个新特征")
        
        return df_new
    
    def _generate_mathematical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成数学变换特征"""
        df_math = df.copy()
        math_features = []
        
        for feature in self.config['numerical_features']:
            if feature not in df_math.columns:
                continue
                
            values = df_math[feature]
            
            # 对数变换（处理正偏态）
            if (values > 0).all():
                df_math[f'{feature}_log'] = np.log1p(values)
                math_features.append(f'{feature}_log')
            
            # 平方根变换
            if (values >= 0).all():
                df_math[f'{feature}_sqrt'] = np.sqrt(values)
                math_features.append(f'{feature}_sqrt')
            
            # 平方变换
            df_math[f'{feature}_square'] = values ** 2
            math_features.append(f'{feature}_square')
            
            # 倒数变换（避免除零）
            if (values != 0).all():
                df_math[f'{feature}_reciprocal'] = 1 / values
                math_features.append(f'{feature}_reciprocal')
        
        return df_math, math_features
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成统计特征"""
        df_stat = df.copy()
        stat_features = []
        
        numerical_cols = [col for col in self.config['numerical_features'] if col in df_stat.columns]
        
        if len(numerical_cols) >= 2:
            # 行级统计特征
            df_stat['row_sum'] = df_stat[numerical_cols].sum(axis=1)
            df_stat['row_mean'] = df_stat[numerical_cols].mean(axis=1)
            df_stat['row_std'] = df_stat[numerical_cols].std(axis=1)
            df_stat['row_min'] = df_stat[numerical_cols].min(axis=1)
            df_stat['row_max'] = df_stat[numerical_cols].max(axis=1)
            df_stat['row_median'] = df_stat[numerical_cols].median(axis=1)
            df_stat['row_range'] = df_stat['row_max'] - df_stat['row_min']
            
            stat_features.extend(['row_sum', 'row_mean', 'row_std', 'row_min', 'row_max', 'row_median', 'row_range'])
            
            # 非零特征数量
            df_stat['non_zero_count'] = (df_stat[numerical_cols] != 0).sum(axis=1)
            stat_features.append('non_zero_count')
        
        return df_stat, stat_features
    
    def _generate_interaction_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成交互特征"""
        df_interact = df.copy()
        interaction_features = []
        
        numerical_cols = [col for col in self.config['numerical_features'] if col in df_interact.columns]
        
        # 限制交互特征数量，避免特征爆炸
        max_interactions = min(10, len(numerical_cols))
        selected_cols = numerical_cols[:max_interactions]
        
        # 两两特征交互
        for i, col1 in enumerate(selected_cols):
            for col2 in selected_cols[i+1:]:
                # 乘积交互
                df_interact[f'{col1}_x_{col2}'] = df_interact[col1] * df_interact[col2]
                interaction_features.append(f'{col1}_x_{col2}')
                
                # 比值交互（避免除零）
                col2_values = df_interact[col2]
                if (col2_values != 0).all():
                    df_interact[f'{col1}_div_{col2}'] = df_interact[col1] / col2_values
                    interaction_features.append(f'{col1}_div_{col2}')
                
                # 差值交互
                df_interact[f'{col1}_minus_{col2}'] = df_interact[col1] - df_interact[col2]
                interaction_features.append(f'{col1}_minus_{col2}')
        
        return df_interact, interaction_features
    
    def _generate_time_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成时间特征"""
        df_time = df.copy()
        time_features = []
        
        for feature in self.config['datetime_features']:
            if feature not in df_time.columns:
                continue
            
            # 确保是datetime类型
            df_time[feature] = pd.to_datetime(df_time[feature], errors='coerce')
            
            # 提取时间组件
            df_time[f'{feature}_year'] = df_time[feature].dt.year
            df_time[f'{feature}_month'] = df_time[feature].dt.month
            df_time[f'{feature}_day'] = df_time[feature].dt.day
            df_time[f'{feature}_weekday'] = df_time[feature].dt.weekday
            df_time[f'{feature}_quarter'] = df_time[feature].dt.quarter
            df_time[f'{feature}_is_weekend'] = (df_time[feature].dt.weekday >= 5).astype(int)
            
            time_features.extend([
                f'{feature}_year', f'{feature}_month', f'{feature}_day',
                f'{feature}_weekday', f'{feature}_quarter', f'{feature}_is_weekend'
            ])
            
            # 距离当前时间的天数
            current_date = datetime.now()
            df_time[f'{feature}_days_from_now'] = (current_date - df_time[feature]).dt.days
            time_features.append(f'{feature}_days_from_now')
        
        return df_time, time_features
    
    def _generate_aggregation_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成聚合特征"""
        df_agg = df.copy()
        agg_features = []
        
        # 基于分类特征的聚合
        for cat_feature in self.config['categorical_features']:
            if cat_feature not in df_agg.columns:
                continue
                
            for num_feature in self.config['numerical_features']:
                if num_feature not in df_agg.columns:
                    continue
                
                # 计算分组统计
                group_stats = df_agg.groupby(cat_feature)[num_feature].agg(['mean', 'std', 'count'])
                
                # 映射回原数据
                df_agg[f'{num_feature}_mean_by_{cat_feature}'] = df_agg[cat_feature].map(group_stats['mean'])
                df_agg[f'{num_feature}_std_by_{cat_feature}'] = df_agg[cat_feature].map(group_stats['std'])
                df_agg[f'{num_feature}_count_by_{cat_feature}'] = df_agg[cat_feature].map(group_stats['count'])
                
                agg_features.extend([
                    f'{num_feature}_mean_by_{cat_feature}',
                    f'{num_feature}_std_by_{cat_feature}',
                    f'{num_feature}_count_by_{cat_feature}'
                ])
                
                # 处理NaN值
                df_agg[f'{num_feature}_std_by_{cat_feature}'].fillna(0, inplace=True)
        
        return df_agg, agg_features
    
    def _generate_ratio_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成比率特征"""
        df_ratio = df.copy()
        ratio_features = []
        
        numerical_cols = [col for col in self.config['numerical_features'] if col in df_ratio.columns]
        
        # 生成与总和的比率
        if len(numerical_cols) >= 2:
            total_sum = df_ratio[numerical_cols].sum(axis=1)
            
            for col in numerical_cols:
                # 避免除零
                ratio_col = f'{col}_ratio_to_total'
                df_ratio[ratio_col] = np.where(total_sum != 0, df_ratio[col] / total_sum, 0)
                ratio_features.append(ratio_col)
        
        # 生成与最大值的比率
        if len(numerical_cols) >= 2:
            row_max = df_ratio[numerical_cols].max(axis=1)
            
            for col in numerical_cols:
                ratio_col = f'{col}_ratio_to_max'
                df_ratio[ratio_col] = np.where(row_max != 0, df_ratio[col] / row_max, 0)
                ratio_features.append(ratio_col)
        
        return df_ratio, ratio_features
    
    def _generate_binning_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """生成分箱特征"""
        df_bin = df.copy()
        binning_features = []
        
        for feature in self.config['numerical_features']:
            if feature not in df_bin.columns:
                continue
            
            # 等频分箱
            try:
                df_bin[f'{feature}_qcut'] = pd.qcut(df_bin[feature], q=5, labels=False, duplicates='drop')
                binning_features.append(f'{feature}_qcut')
            except:
                pass
            
            # 等距分箱
            try:
                df_bin[f'{feature}_cut'] = pd.cut(df_bin[feature], bins=5, labels=False)
                binning_features.append(f'{feature}_cut')
            except:
                pass
        
        return df_bin, binning_features
    
    def _apply_feature_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用特征变换"""
        self.logger.info("应用特征变换")
        
        df_transformed = df.copy()
        
        # 分类特征编码
        df_transformed = self._encode_categorical_features(df_transformed)
        
        # 处理异常值
        df_transformed = self._handle_outliers(df_transformed)
        
        return df_transformed
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        df_encoded = df.copy()
        
        for feature in self.config['categorical_features']:
            if feature not in df_encoded.columns:
                continue
            
            unique_count = df_encoded[feature].nunique()
            
            if unique_count <= 2:
                # 二分类特征使用标签编码
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                self.feature_transformers[f'{feature}_label_encoder'] = le
            
            elif unique_count <= 10:
                # 低基数分类特征使用独热编码
                encoded_features = pd.get_dummies(df_encoded[feature], prefix=feature)
                df_encoded = pd.concat([df_encoded, encoded_features], axis=1)
                df_encoded.drop(feature, axis=1, inplace=True)
            
            else:
                # 高基数分类特征使用目标编码
                if self.config['target_column'] in df_encoded.columns:
                    target_means = df_encoded.groupby(feature)[self.config['target_column']].mean()
                    df_encoded[f'{feature}_target_encoded'] = df_encoded[feature].map(target_means)
                    df_encoded.drop(feature, axis=1, inplace=True)
                    self.feature_transformers[f'{feature}_target_encoder'] = target_means
        
        return df_encoded
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        df_no_outliers = df.copy()
        
        numerical_cols = df_no_outliers.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.config['target_column']]
        
        for col in numerical_cols:
            # 使用IQR方法检测异常值
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 截断异常值
            df_no_outliers[col] = np.clip(df_no_outliers[col], lower_bound, upper_bound)
        
        return df_no_outliers
    
    def _feature_selection(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """特征选择"""
        self.logger.info("执行特征选择")
        
        if target_column not in df.columns:
            self.logger.warning(f"目标列 {target_column} 不存在，跳过特征选择")
            return df
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        selected_features = set(X.columns)
        
        # 1. 方差过滤
        if 'variance' in self.config['feature_selection_methods']:
            selector = VarianceThreshold(threshold=0.01)
            X_var = selector.fit_transform(X)
            var_features = X.columns[selector.get_support()]
            selected_features &= set(var_features)
            self.feature_selectors['variance_selector'] = selector
        
        # 2. 相关性过滤
        if 'correlation' in self.config['feature_selection_methods']:
            corr_features = self._correlation_feature_selection(X, threshold=self.config['correlation_threshold'])
            selected_features &= set(corr_features)
        
        # 3. 互信息特征选择
        if 'mutual_info' in self.config['feature_selection_methods']:
            try:
                selector = SelectKBest(score_func=mutual_info_classif, k=min(50, len(selected_features)))
                X_selected = X[list(selected_features)]
                selector.fit(X_selected, y)
                mi_features = X_selected.columns[selector.get_support()]
                selected_features &= set(mi_features)
                self.feature_selectors['mutual_info_selector'] = selector
            except:
                self.logger.warning("互信息特征选择失败")
        
        # 4. 递归特征消除
        if 'rfe' in self.config['feature_selection_methods'] and len(selected_features) > 20:
            try:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                selector = RFE(estimator, n_features_to_select=min(20, len(selected_features)))
                X_selected = X[list(selected_features)]
                selector.fit(X_selected, y)
                rfe_features = X_selected.columns[selector.get_support()]
                selected_features &= set(rfe_features)
                self.feature_selectors['rfe_selector'] = selector
            except:
                self.logger.warning("RFE特征选择失败")
        
        # 保存特征重要性分数
        self._calculate_feature_importance(X[list(selected_features)], y)
        
        # 返回选择后的特征
        final_features = list(selected_features) + [target_column]
        df_selected = df[final_features]
        
        self.logger.info(f"特征选择完成，从 {len(X.columns)} 个特征中选择了 {len(selected_features)} 个特征")
        
        return df_selected
    
    def _correlation_feature_selection(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """基于相关性的特征选择"""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找到高相关性的特征对
        high_corr_pairs = []
        for column in upper_triangle.columns:
            high_corr_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((feature, column))
        
        # 移除高相关性特征中的一个
        features_to_drop = set()
        for feature1, feature2 in high_corr_pairs:
            # 保留方差更大的特征
            if X[feature1].var() > X[feature2].var():
                features_to_drop.add(feature2)
            else:
                features_to_drop.add(feature1)
        
        selected_features = [col for col in X.columns if col not in features_to_drop]
        
        return selected_features
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """计算特征重要性"""
        try:
            # 使用随机森林计算特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_scores = dict(zip(X.columns, rf.feature_importances_))
            self.feature_importance_scores = importance_scores
            
        except Exception as e:
            self.logger.warning(f"计算特征重要性失败: {str(e)}")
    
    def _feature_scaling(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """特征缩放"""
        self.logger.info("执行特征缩放")
        
        df_scaled = df.copy()
        
        # 获取数值特征
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_column]
        
        if len(numerical_cols) > 0:
            # 使用标准化
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.feature_transformers['standard_scaler'] = scaler
        
        return df_scaled
    
    def _generate_feature_engineering_report(self, df_original: pd.DataFrame, df_final: pd.DataFrame):
        """生成特征工程报告"""
        self.logger.info("生成特征工程报告")
        
        report = {
            'original_features_count': len(df_original.columns),
            'final_features_count': len(df_final.columns),
            'generated_features': self.generated_features,
            'feature_importance_scores': self.feature_importance_scores,
            'feature_transformers': list(self.feature_transformers.keys()),
            'feature_selectors': list(self.feature_selectors.keys()),
            'processing_time': datetime.now().isoformat()
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'feature_engineering_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存特征重要性
        if self.feature_importance_scores:
            importance_df = pd.DataFrame(
                list(self.feature_importance_scores.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(self.output_dir, 'feature_importance.xlsx')
            importance_df.to_excel(importance_path, index=False)
        
        # 保存变换器
        transformers_path = os.path.join(self.output_dir, 'feature_transformers.joblib')
        joblib.dump(self.feature_transformers, transformers_path)
        
        self.logger.info(f"特征工程报告已保存: {report_path}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对新数据应用已训练的特征工程流程
        
        Args:
            df: 新数据
            
        Returns:
            变换后的数据
        """
        self.logger.info("对新数据应用特征工程变换")
        
        df_transformed = df.copy()
        
        # 应用保存的变换器
        for transformer_name, transformer in self.feature_transformers.items():
            try:
                if 'label_encoder' in transformer_name:
                    feature_name = transformer_name.replace('_label_encoder', '')
                    if feature_name in df_transformed.columns:
                        df_transformed[feature_name] = transformer.transform(df_transformed[feature_name].astype(str))
                
                elif 'target_encoder' in transformer_name:
                    feature_name = transformer_name.replace('_target_encoder', '')
                    if feature_name in df_transformed.columns:
                        df_transformed[f'{feature_name}_target_encoded'] = df_transformed[feature_name].map(transformer)
                        df_transformed.drop(feature_name, axis=1, inplace=True)
                
                elif 'standard_scaler' in transformer_name:
                    numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) > 0:
                        df_transformed[numerical_cols] = transformer.transform(df_transformed[numerical_cols])
            
            except Exception as e:
                self.logger.warning(f"应用变换器 {transformer_name} 失败: {str(e)}")
        
        return df_transformed
    
    def save_feature_engineering_pipeline(self, filename: str = 'feature_engineering_pipeline.joblib'):
        """保存特征工程流水线"""
        pipeline_data = {
            'config': self.config,
            'feature_transformers': self.feature_transformers,
            'feature_selectors': self.feature_selectors,
            'generated_features': self.generated_features,
            'feature_importance_scores': self.feature_importance_scores
        }
        
        output_path = os.path.join(self.output_dir, filename)
        joblib.dump(pipeline_data, output_path)
        
        self.logger.info(f"特征工程流水线已保存: {output_path}")
    
    def load_feature_engineering_pipeline(self, filename: str = 'feature_engineering_pipeline.joblib'):
        """加载特征工程流水线"""
        input_path = os.path.join(self.output_dir, filename)
        
        if os.path.exists(input_path):
            pipeline_data = joblib.load(input_path)
            
            self.config = pipeline_data.get('config', {})
            self.feature_transformers = pipeline_data.get('feature_transformers', {})
            self.feature_selectors = pipeline_data.get('feature_selectors', {})
            self.generated_features = pipeline_data.get('generated_features', {})
            self.feature_importance_scores = pipeline_data.get('feature_importance_scores', {})
            
            self.logger.info(f"特征工程流水线已加载: {input_path}")
        else:
            self.logger.warning(f"流水线文件不存在: {input_path}")


def main():
    """主函数 - 演示特征工程流程"""
    try:
        # 这里应该加载实际数据进行特征工程
        # 由于这是演示，我们创建一个简单的示例
        
        print("特征工程模块已创建完成！")
        print("主要功能包括:")
        print("1. 自动化特征类型识别")
        print("2. 智能特征生成（数学变换、统计特征、交互特征等）")
        print("3. 多种特征选择方法")
        print("4. 特征变换和编码")
        print("5. 特征重要性分析")
        print("6. 特征工程流水线保存和加载")
        print("7. 新数据变换支持")
        
    except Exception as e:
        print(f"特征工程过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
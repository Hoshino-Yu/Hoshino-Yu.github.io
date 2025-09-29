"""
增强的模型评估系统
提供全面的模型对比分析功能，包括运行效率、复杂度分析、鲁棒性测试等
"""

import os
import json
import logging
import joblib
import warnings
import time
import psutil
import tracemalloc
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings('ignore')

class EnhancedModelEvaluator:
    """
    增强的模型评估器
    提供完整的模型性能评估、效率分析、复杂度分析和鲁棒性测试
    """
    
    def __init__(self, output_dir: str = '../输出结果/enhanced_evaluation'):
        """
        初始化增强评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.evaluation_results = {}
        self.model_performance_metrics = {}
        self.model_efficiency_metrics = {}
        self.model_complexity_metrics = {}
        self.robustness_test_results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.output_dir, 'enhanced_evaluation.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def enhanced_model_comparison(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                                y_test: pd.Series, X_train: Optional[pd.DataFrame] = None,
                                y_train: Optional[pd.Series] = None) -> Dict:
        """
        增强的模型对比分析
        
        Args:
            models_dict: 模型字典 {model_name: model}
            X_test: 测试特征
            y_test: 测试标签
            X_train: 训练特征（可选）
            y_train: 训练标签（可选）
            
        Returns:
            完整的模型对比结果
        """
        self.logger.info("开始增强模型对比分析")
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            self.logger.info(f"评估模型: {model_name}")
            
            # 1. 基础性能评估
            performance_metrics = self._evaluate_model_performance(model, X_test, y_test, model_name)
            
            # 2. 运行效率评估
            efficiency_metrics = self._evaluate_model_efficiency(model, X_test, y_test, X_train, y_train)
            
            # 3. 模型复杂度分析
            complexity_metrics = self._analyze_model_complexity(model, X_test)
            
            # 4. 异常输入处理能力测试
            robustness_metrics = self._test_model_robustness(model, X_test, y_test)
            
            # 5. 不同条件下的性能表现
            condition_performance = self._evaluate_performance_under_conditions(model, X_test, y_test)
            
            comparison_results[model_name] = {
                'performance_metrics': performance_metrics,
                'efficiency_metrics': efficiency_metrics,
                'complexity_metrics': complexity_metrics,
                'robustness_metrics': robustness_metrics,
                'condition_performance': condition_performance,
                'overall_score': self._calculate_overall_score(
                    performance_metrics, efficiency_metrics, complexity_metrics, robustness_metrics
                )
            }
        
        # 生成对比可视化
        self._generate_enhanced_comparison_charts(comparison_results)
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(comparison_results)
        
        self.logger.info("增强模型对比分析完成")
        
        return {
            'detailed_results': comparison_results,
            'comparison_report': comparison_report,
            'best_overall_model': self._find_best_model(comparison_results),
            'recommendations': self._generate_model_recommendations(comparison_results)
        }
    
    def _evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict:
        """评估模型基础性能"""
        try:
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 基础指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"性能评估失败 {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_model_efficiency(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                 X_train: Optional[pd.DataFrame] = None, 
                                 y_train: Optional[pd.Series] = None) -> Dict:
        """评估模型运行效率"""
        efficiency_metrics = {}
        
        try:
            # 1. 预测时间测试
            tracemalloc.start()
            
            # 单次预测时间
            single_start = time.time()
            _ = model.predict(X_test.iloc[:1])
            single_prediction_time = time.time() - single_start
            
            # 批量预测时间
            batch_start = time.time()
            predictions = model.predict(X_test)
            batch_prediction_time = time.time() - batch_start
            
            # 内存使用情况
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            efficiency_metrics.update({
                'single_prediction_time_ms': single_prediction_time * 1000,
                'batch_prediction_time_s': batch_prediction_time,
                'predictions_per_second': len(X_test) / batch_prediction_time if batch_prediction_time > 0 else 0,
                'memory_usage_mb': current / 1024 / 1024,
                'peak_memory_mb': peak / 1024 / 1024
            })
            
            # 2. 训练时间测试（如果提供了训练数据）
            if X_train is not None and y_train is not None:
                # 使用小样本测试训练时间
                sample_size = min(1000, len(X_train))
                X_sample = X_train.sample(n=sample_size, random_state=42)
                y_sample = y_train.loc[X_sample.index]
                
                # 克隆模型进行训练时间测试
                test_model = clone(model)
                
                train_start = time.time()
                test_model.fit(X_sample, y_sample)
                training_time = time.time() - train_start
                
                efficiency_metrics.update({
                    'training_time_per_1k_samples_s': training_time,
                    'estimated_full_training_time_s': training_time * (len(X_train) / sample_size)
                })
            
            # 3. CPU使用率监控
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            efficiency_metrics['cpu_usage_percent'] = cpu_percent
            
        except Exception as e:
            self.logger.warning(f"效率评估失败: {str(e)}")
            efficiency_metrics['error'] = str(e)
        
        return efficiency_metrics
    
    def _analyze_model_complexity(self, model, X_test: pd.DataFrame) -> Dict:
        """分析模型复杂度"""
        complexity_metrics = {}
        
        try:
            # 1. 参数数量分析
            if hasattr(model, 'coef_'):
                # 线性模型
                n_parameters = np.prod(model.coef_.shape)
                complexity_metrics['n_parameters'] = n_parameters
                complexity_metrics['model_type'] = 'linear'
            elif hasattr(model, 'n_estimators'):
                # 集成模型
                n_estimators = model.n_estimators
                if hasattr(model, 'max_depth'):
                    max_depth = model.max_depth or 10  # 默认深度
                    complexity_metrics['n_estimators'] = n_estimators
                    complexity_metrics['max_depth'] = max_depth
                    complexity_metrics['estimated_nodes'] = n_estimators * (2 ** max_depth - 1)
                complexity_metrics['model_type'] = 'ensemble'
            elif hasattr(model, 'support_vectors_'):
                # SVM
                n_support_vectors = len(model.support_vectors_)
                complexity_metrics['n_support_vectors'] = n_support_vectors
                complexity_metrics['model_type'] = 'svm'
            else:
                complexity_metrics['model_type'] = 'other'
            
            # 2. 模型大小（序列化后）
            import pickle
            model_bytes = pickle.dumps(model)
            complexity_metrics['model_size_mb'] = len(model_bytes) / 1024 / 1024
            
            # 3. 决策复杂度（基于特征重要性的分布）
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # 计算重要性的基尼系数（衡量分布不均匀程度）
                sorted_importances = np.sort(importances)
                n = len(importances)
                gini_coefficient = (2 * np.sum((np.arange(1, n + 1) * sorted_importances))) / (n * np.sum(sorted_importances)) - (n + 1) / n
                complexity_metrics['feature_importance_gini'] = gini_coefficient
                complexity_metrics['effective_features'] = np.sum(importances > 0.01)  # 重要性>1%的特征数
            
            # 4. 可解释性评分
            interpretability_score = self._calculate_interpretability_score(model, complexity_metrics)
            complexity_metrics['interpretability_score'] = interpretability_score
            
        except Exception as e:
            self.logger.warning(f"复杂度分析失败: {str(e)}")
            complexity_metrics['error'] = str(e)
        
        return complexity_metrics
    
    def _test_model_robustness(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """测试模型对异常输入的处理能力"""
        robustness_metrics = {}
        
        try:
            # 1. 缺失值处理测试
            missing_test_results = self._test_missing_value_handling(model, X_test, y_test)
            robustness_metrics['missing_value_handling'] = missing_test_results
            
            # 2. 异常值处理测试
            outlier_test_results = self._test_outlier_handling(model, X_test, y_test)
            robustness_metrics['outlier_handling'] = outlier_test_results
            
            # 3. 数据分布变化测试
            distribution_test_results = self._test_distribution_shift(model, X_test, y_test)
            robustness_metrics['distribution_shift'] = distribution_test_results
            
            # 4. 噪声鲁棒性测试
            noise_test_results = self._test_noise_robustness(model, X_test, y_test)
            robustness_metrics['noise_robustness'] = noise_test_results
            
            # 5. 计算综合鲁棒性评分
            robustness_score = self._calculate_robustness_score(robustness_metrics)
            robustness_metrics['overall_robustness_score'] = robustness_score
            
        except Exception as e:
            self.logger.warning(f"鲁棒性测试失败: {str(e)}")
            robustness_metrics['error'] = str(e)
        
        return robustness_metrics
    
    def _test_missing_value_handling(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """测试缺失值处理能力"""
        results = {}
        
        try:
            # 原始性能
            original_predictions = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_predictions)
            
            # 随机引入缺失值
            missing_ratios = [0.05, 0.1, 0.2, 0.3]
            
            for ratio in missing_ratios:
                X_missing = X_test.copy()
                
                # 随机选择位置设置为NaN
                mask = np.random.random(X_missing.shape) < ratio
                X_missing = X_missing.astype(float)
                X_missing[mask] = np.nan
                
                # 简单填充策略
                X_filled = X_missing.fillna(X_missing.mean())
                
                try:
                    predictions = model.predict(X_filled)
                    accuracy = accuracy_score(y_test, predictions)
                    performance_drop = original_accuracy - accuracy
                    
                    results[f'missing_{int(ratio*100)}pct'] = {
                        'accuracy': accuracy,
                        'performance_drop': performance_drop,
                        'relative_drop': performance_drop / original_accuracy if original_accuracy > 0 else 0
                    }
                except Exception as e:
                    results[f'missing_{int(ratio*100)}pct'] = {
                        'error': str(e),
                        'accuracy': 0,
                        'performance_drop': 1,
                        'relative_drop': 1
                    }
        
        except Exception as e:
            self.logger.warning(f"缺失值测试失败: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _test_outlier_handling(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """测试异常值处理能力"""
        results = {}
        
        try:
            # 原始性能
            original_predictions = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_predictions)
            
            # 引入异常值
            outlier_ratios = [0.01, 0.05, 0.1]
            
            for ratio in outlier_ratios:
                X_outlier = X_test.copy()
                
                # 随机选择数值列
                numeric_cols = X_outlier.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    n_outliers = int(len(X_outlier) * ratio)
                    if n_outliers > 0:
                        outlier_indices = np.random.choice(len(X_outlier), n_outliers, replace=False)
                        
                        # 生成极端值（均值±5倍标准差）
                        mean_val = X_outlier[col].mean()
                        std_val = X_outlier[col].std()
                        extreme_values = np.random.choice(
                            [mean_val + 5*std_val, mean_val - 5*std_val], 
                            n_outliers
                        )
                        
                        X_outlier.iloc[outlier_indices, X_outlier.columns.get_loc(col)] = extreme_values
                
                try:
                    predictions = model.predict(X_outlier)
                    accuracy = accuracy_score(y_test, predictions)
                    performance_drop = original_accuracy - accuracy
                    
                    results[f'outlier_{int(ratio*100)}pct'] = {
                        'accuracy': accuracy,
                        'performance_drop': performance_drop,
                        'relative_drop': performance_drop / original_accuracy if original_accuracy > 0 else 0
                    }
                except Exception as e:
                    results[f'outlier_{int(ratio*100)}pct'] = {
                        'error': str(e),
                        'accuracy': 0,
                        'performance_drop': 1,
                        'relative_drop': 1
                    }
        
        except Exception as e:
            self.logger.warning(f"异常值测试失败: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _test_distribution_shift(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """测试数据分布变化的影响"""
        results = {}
        
        try:
            # 原始性能
            original_predictions = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_predictions)
            
            # 数值特征标准化程度测试
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # 测试不同的缩放因子
                scale_factors = [0.5, 2.0, 5.0]
                
                for factor in scale_factors:
                    X_scaled = X_test.copy()
                    X_scaled[numeric_cols] = X_scaled[numeric_cols] * factor
                    
                    try:
                        predictions = model.predict(X_scaled)
                        accuracy = accuracy_score(y_test, predictions)
                        performance_drop = original_accuracy - accuracy
                        
                        results[f'scale_{factor}x'] = {
                            'accuracy': accuracy,
                            'performance_drop': performance_drop,
                            'relative_drop': performance_drop / original_accuracy if original_accuracy > 0 else 0
                        }
                    except Exception as e:
                        results[f'scale_{factor}x'] = {
                            'error': str(e),
                            'accuracy': 0,
                            'performance_drop': 1,
                            'relative_drop': 1
                        }
        
        except Exception as e:
            self.logger.warning(f"分布变化测试失败: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _test_noise_robustness(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """测试噪声鲁棒性"""
        results = {}
        
        try:
            # 原始性能
            original_predictions = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_predictions)
            
            # 添加不同程度的高斯噪声
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            
            for noise_level in noise_levels:
                X_noisy = X_test.copy()
                
                # 为数值列添加噪声
                for col in numeric_cols:
                    noise = np.random.normal(0, X_test[col].std() * noise_level, len(X_test))
                    X_noisy[col] = X_noisy[col] + noise
                
                try:
                    predictions = model.predict(X_noisy)
                    accuracy = accuracy_score(y_test, predictions)
                    performance_drop = original_accuracy - accuracy
                    
                    results[f'noise_{int(noise_level*100)}pct'] = {
                        'accuracy': accuracy,
                        'performance_drop': performance_drop,
                        'relative_drop': performance_drop / original_accuracy if original_accuracy > 0 else 0
                    }
                except Exception as e:
                    results[f'noise_{int(noise_level*100)}pct'] = {
                        'error': str(e),
                        'accuracy': 0,
                        'performance_drop': 1,
                        'relative_drop': 1
                    }
        
        except Exception as e:
            self.logger.warning(f"噪声鲁棒性测试失败: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_performance_under_conditions(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估不同条件下的性能表现"""
        condition_results = {}
        
        try:
            # 1. 不同数据集大小下的性能
            sample_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
            size_performance = {}
            
            for size in sample_sizes:
                n_samples = int(len(X_test) * size)
                if n_samples > 0:
                    X_sample = X_test.sample(n=n_samples, random_state=42)
                    y_sample = y_test.loc[X_sample.index]
                    
                    predictions = model.predict(X_sample)
                    accuracy = accuracy_score(y_sample, predictions)
                    
                    size_performance[f'{int(size*100)}pct'] = accuracy
            
            condition_results['sample_size_performance'] = size_performance
            
            # 2. 不同类别平衡度下的性能
            class_balance_performance = self._evaluate_class_balance_performance(model, X_test, y_test)
            condition_results['class_balance_performance'] = class_balance_performance
            
        except Exception as e:
            self.logger.warning(f"条件性能评估失败: {str(e)}")
            condition_results['error'] = str(e)
        
        return condition_results
    
    def _evaluate_class_balance_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估不同类别平衡度下的性能"""
        results = {}
        
        try:
            # 按类别分组
            class_0_indices = y_test[y_test == 0].index
            class_1_indices = y_test[y_test == 1].index
            
            # 测试不同的类别比例
            ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for ratio in ratios:
                # 构建特定比例的测试集
                n_class_1 = min(len(class_1_indices), int(500 * ratio))
                n_class_0 = min(len(class_0_indices), int(500 * (1 - ratio)))
                
                if n_class_1 > 0 and n_class_0 > 0:
                    selected_indices = (
                        list(np.random.choice(class_1_indices, n_class_1, replace=False)) +
                        list(np.random.choice(class_0_indices, n_class_0, replace=False))
                    )
                    
                    X_balanced = X_test.loc[selected_indices]
                    y_balanced = y_test.loc[selected_indices]
                    
                    predictions = model.predict(X_balanced)
                    accuracy = accuracy_score(y_balanced, predictions)
                    precision = precision_score(y_balanced, predictions, zero_division=0)
                    recall = recall_score(y_balanced, predictions, zero_division=0)
                    
                    results[f'ratio_{int(ratio*100)}pct'] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'class_1_ratio': ratio
                    }
        
        except Exception as e:
            self.logger.warning(f"类别平衡性能评估失败: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_interpretability_score(self, model, complexity_metrics: Dict) -> float:
        """计算模型可解释性评分"""
        score = 0.0
        
        # 基于模型类型的基础分数
        model_type = complexity_metrics.get('model_type', 'other')
        
        if model_type == 'linear':
            score += 0.8  # 线性模型高可解释性
        elif model_type == 'ensemble':
            score += 0.4  # 集成模型中等可解释性
        elif model_type == 'svm':
            score += 0.3  # SVM较低可解释性
        else:
            score += 0.2  # 其他模型低可解释性
        
        # 基于复杂度的调整
        if 'n_parameters' in complexity_metrics:
            # 参数越少，可解释性越高
            n_params = complexity_metrics['n_parameters']
            if n_params < 10:
                score += 0.2
            elif n_params < 100:
                score += 0.1
        
        if 'effective_features' in complexity_metrics:
            # 有效特征越少，可解释性越高
            n_features = complexity_metrics['effective_features']
            if n_features < 5:
                score += 0.2
            elif n_features < 20:
                score += 0.1
        
        return min(score, 1.0)  # 限制在0-1之间
    
    def _calculate_robustness_score(self, robustness_metrics: Dict) -> float:
        """计算综合鲁棒性评分"""
        scores = []
        
        # 缺失值处理评分
        missing_handling = robustness_metrics.get('missing_value_handling', {})
        if missing_handling and 'error' not in missing_handling:
            missing_scores = []
            for key, value in missing_handling.items():
                if isinstance(value, dict) and 'relative_drop' in value:
                    # 性能下降越小，鲁棒性越好
                    missing_scores.append(1 - min(value['relative_drop'], 1))
            if missing_scores:
                scores.append(np.mean(missing_scores))
        
        # 异常值处理评分
        outlier_handling = robustness_metrics.get('outlier_handling', {})
        if outlier_handling and 'error' not in outlier_handling:
            outlier_scores = []
            for key, value in outlier_handling.items():
                if isinstance(value, dict) and 'relative_drop' in value:
                    outlier_scores.append(1 - min(value['relative_drop'], 1))
            if outlier_scores:
                scores.append(np.mean(outlier_scores))
        
        # 噪声鲁棒性评分
        noise_robustness = robustness_metrics.get('noise_robustness', {})
        if noise_robustness and 'error' not in noise_robustness:
            noise_scores = []
            for key, value in noise_robustness.items():
                if isinstance(value, dict) and 'relative_drop' in value:
                    noise_scores.append(1 - min(value['relative_drop'], 1))
            if noise_scores:
                scores.append(np.mean(noise_scores))
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_overall_score(self, performance_metrics: Dict, efficiency_metrics: Dict,
                               complexity_metrics: Dict, robustness_metrics: Dict) -> float:
        """计算模型综合评分"""
        
        # 权重设置
        weights = {
            'performance': 0.4,
            'efficiency': 0.2,
            'complexity': 0.2,
            'robustness': 0.2
        }
        
        # 性能评分（基于准确率和AUC）
        performance_score = 0
        if 'accuracy' in performance_metrics:
            performance_score += performance_metrics['accuracy'] * 0.5
        if 'roc_auc' in performance_metrics:
            performance_score += performance_metrics['roc_auc'] * 0.5
        
        # 效率评分（基于预测速度，归一化处理）
        efficiency_score = 0
        if 'predictions_per_second' in efficiency_metrics:
            # 假设1000预测/秒为满分
            pps = efficiency_metrics['predictions_per_second']
            efficiency_score = min(pps / 1000, 1.0)
        
        # 复杂度评分（可解释性评分）
        complexity_score = complexity_metrics.get('interpretability_score', 0)
        
        # 鲁棒性评分
        robustness_score = robustness_metrics.get('overall_robustness_score', 0)
        
        # 加权综合评分
        overall_score = (
            performance_score * weights['performance'] +
            efficiency_score * weights['efficiency'] +
            complexity_score * weights['complexity'] +
            robustness_score * weights['robustness']
        )
        
        return overall_score
    
    def _generate_enhanced_comparison_charts(self, comparison_results: Dict):
        """生成增强的对比图表"""
        
        # 1. 综合性能雷达图
        self._create_radar_chart(comparison_results)
        
        # 2. 效率对比图
        self._create_efficiency_comparison_chart(comparison_results)
        
        # 3. 鲁棒性对比图
        self._create_robustness_comparison_chart(comparison_results)
        
        # 4. 复杂度vs性能散点图
        self._create_complexity_performance_scatter(comparison_results)
    
    def _create_radar_chart(self, comparison_results: Dict):
        """创建雷达图显示模型综合性能"""
        
        models = list(comparison_results.keys())
        metrics = ['性能', '效率', '可解释性', '鲁棒性', '综合评分']
        
        fig = go.Figure()
        
        for model_name in models:
            result = comparison_results[model_name]
            
            # 提取各维度评分
            performance_score = result['performance_metrics'].get('accuracy', 0)
            efficiency_score = min(result['efficiency_metrics'].get('predictions_per_second', 0) / 1000, 1.0)
            complexity_score = result['complexity_metrics'].get('interpretability_score', 0)
            robustness_score = result['robustness_metrics'].get('overall_robustness_score', 0)
            overall_score = result['overall_score']
            
            values = [performance_score, efficiency_score, complexity_score, robustness_score, overall_score]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="模型综合性能雷达图"
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'model_radar_comparison.html'))
    
    def _create_efficiency_comparison_chart(self, comparison_results: Dict):
        """创建效率对比图表"""
        
        models = list(comparison_results.keys())
        
        # 提取效率指标
        prediction_times = []
        memory_usage = []
        predictions_per_sec = []
        
        for model_name in models:
            efficiency = comparison_results[model_name]['efficiency_metrics']
            prediction_times.append(efficiency.get('batch_prediction_time_s', 0))
            memory_usage.append(efficiency.get('memory_usage_mb', 0))
            predictions_per_sec.append(efficiency.get('predictions_per_second', 0))
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['预测时间对比', '内存使用对比', '预测速度对比', '效率综合评分'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 预测时间
        fig.add_trace(
            go.Bar(x=models, y=prediction_times, name='预测时间(秒)', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 内存使用
        fig.add_trace(
            go.Bar(x=models, y=memory_usage, name='内存使用(MB)', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 预测速度
        fig.add_trace(
            go.Bar(x=models, y=predictions_per_sec, name='预测/秒', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # 效率评分
        efficiency_scores = [min(pps / 1000, 1.0) for pps in predictions_per_sec]
        fig.add_trace(
            go.Bar(x=models, y=efficiency_scores, name='效率评分', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="模型效率对比分析")
        fig.write_html(os.path.join(self.output_dir, 'charts', 'efficiency_comparison.html'))
    
    def _create_robustness_comparison_chart(self, comparison_results: Dict):
        """创建鲁棒性对比图表"""
        
        models = list(comparison_results.keys())
        
        # 创建热力图数据
        robustness_data = []
        test_types = ['缺失值处理', '异常值处理', '噪声鲁棒性', '分布变化']
        
        for model_name in models:
            robustness = comparison_results[model_name]['robustness_metrics']
            
            # 计算各类测试的平均性能保持率
            missing_score = self._calculate_avg_performance_retention(
                robustness.get('missing_value_handling', {})
            )
            outlier_score = self._calculate_avg_performance_retention(
                robustness.get('outlier_handling', {})
            )
            noise_score = self._calculate_avg_performance_retention(
                robustness.get('noise_robustness', {})
            )
            distribution_score = self._calculate_avg_performance_retention(
                robustness.get('distribution_shift', {})
            )
            
            robustness_data.append([missing_score, outlier_score, noise_score, distribution_score])
        
        fig = go.Figure(data=go.Heatmap(
            z=robustness_data,
            x=test_types,
            y=models,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in robustness_data],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='模型鲁棒性对比热力图',
            xaxis_title='测试类型',
            yaxis_title='模型',
            height=400
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'robustness_heatmap.html'))
    
    def _calculate_avg_performance_retention(self, test_results: Dict) -> float:
        """计算平均性能保持率"""
        if not test_results or 'error' in test_results:
            return 0.0
        
        retention_rates = []
        for key, value in test_results.items():
            if isinstance(value, dict) and 'relative_drop' in value:
                retention_rate = 1 - min(value['relative_drop'], 1)
                retention_rates.append(retention_rate)
        
        return np.mean(retention_rates) if retention_rates else 0.0
    
    def _create_complexity_performance_scatter(self, comparison_results: Dict):
        """创建复杂度vs性能散点图"""
        
        models = list(comparison_results.keys())
        
        performance_scores = []
        complexity_scores = []
        efficiency_scores = []
        model_names = []
        
        for model_name in models:
            result = comparison_results[model_name]
            
            performance_scores.append(result['performance_metrics'].get('accuracy', 0))
            complexity_scores.append(result['complexity_metrics'].get('interpretability_score', 0))
            efficiency_scores.append(min(result['efficiency_metrics'].get('predictions_per_second', 0) / 1000, 1.0))
            model_names.append(model_name)
        
        fig = go.Figure(data=go.Scatter(
            x=complexity_scores,
            y=performance_scores,
            mode='markers+text',
            marker=dict(
                size=[score * 50 + 10 for score in efficiency_scores],  # 气泡大小表示效率
                color=efficiency_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="效率评分")
            ),
            text=model_names,
            textposition="top center"
        ))
        
        fig.update_layout(
            title='模型复杂度 vs 性能 (气泡大小表示效率)',
            xaxis_title='可解释性评分',
            yaxis_title='性能评分',
            height=600
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'complexity_performance_scatter.html'))
    
    def _find_best_model(self, comparison_results: Dict) -> Dict:
        """找出最佳模型"""
        
        best_overall = max(comparison_results.keys(), 
                          key=lambda x: comparison_results[x]['overall_score'])
        
        best_performance = max(comparison_results.keys(),
                             key=lambda x: comparison_results[x]['performance_metrics'].get('accuracy', 0))
        
        best_efficiency = max(comparison_results.keys(),
                            key=lambda x: comparison_results[x]['efficiency_metrics'].get('predictions_per_second', 0))
        
        best_robustness = max(comparison_results.keys(),
                            key=lambda x: comparison_results[x]['robustness_metrics'].get('overall_robustness_score', 0))
        
        return {
            'best_overall': best_overall,
            'best_performance': best_performance,
            'best_efficiency': best_efficiency,
            'best_robustness': best_robustness,
            'scores': {
                'overall': comparison_results[best_overall]['overall_score'],
                'performance': comparison_results[best_performance]['performance_metrics'].get('accuracy', 0),
                'efficiency': comparison_results[best_efficiency]['efficiency_metrics'].get('predictions_per_second', 0),
                'robustness': comparison_results[best_robustness]['robustness_metrics'].get('overall_robustness_score', 0)
            }
        }
    
    def _generate_model_recommendations(self, comparison_results: Dict) -> Dict:
        """生成模型推荐建议"""
        
        recommendations = {}
        
        for model_name, result in comparison_results.items():
            model_recommendations = []
            
            # 基于性能的建议
            accuracy = result['performance_metrics'].get('accuracy', 0)
            if accuracy < 0.7:
                model_recommendations.append("模型准确率较低，建议进行特征工程或超参数调优")
            elif accuracy > 0.9:
                model_recommendations.append("模型性能优秀，可考虑用于生产环境")
            
            # 基于效率的建议
            pps = result['efficiency_metrics'].get('predictions_per_second', 0)
            if pps < 100:
                model_recommendations.append("预测速度较慢，不适合实时应用场景")
            elif pps > 1000:
                model_recommendations.append("预测速度快，适合高并发场景")
            
            # 基于鲁棒性的建议
            robustness_score = result['robustness_metrics'].get('overall_robustness_score', 0)
            if robustness_score < 0.5:
                model_recommendations.append("鲁棒性较差，需要加强数据预处理和异常处理")
            elif robustness_score > 0.8:
                model_recommendations.append("鲁棒性良好，能够处理各种异常情况")
            
            # 基于复杂度的建议
            interpretability = result['complexity_metrics'].get('interpretability_score', 0)
            if interpretability < 0.3:
                model_recommendations.append("模型可解释性较低，建议在监管严格的场景中谨慎使用")
            elif interpretability > 0.7:
                model_recommendations.append("模型可解释性好，适合需要解释性的业务场景")
            
            recommendations[model_name] = model_recommendations
        
        return recommendations
    
    def _generate_comparison_report(self, comparison_results: Dict) -> str:
        """生成对比报告"""
        
        report_lines = []
        report_lines.append("# 增强模型对比分析报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 概述
        report_lines.append("## 概述")
        report_lines.append(f"本次对比分析了 {len(comparison_results)} 个模型的综合性能，包括基础性能、运行效率、模型复杂度和鲁棒性测试。")
        report_lines.append("")
        
        # 最佳模型
        best_models = self._find_best_model(comparison_results)
        report_lines.append("## 最佳模型")
        report_lines.append(f"- **综合最佳**: {best_models['best_overall']} (评分: {best_models['scores']['overall']:.4f})")
        report_lines.append(f"- **性能最佳**: {best_models['best_performance']} (准确率: {best_models['scores']['performance']:.4f})")
        report_lines.append(f"- **效率最佳**: {best_models['best_efficiency']} (预测速度: {best_models['scores']['efficiency']:.2f} 预测/秒)")
        report_lines.append(f"- **鲁棒性最佳**: {best_models['best_robustness']} (鲁棒性评分: {best_models['scores']['robustness']:.4f})")
        report_lines.append("")
        
        # 详细分析
        report_lines.append("## 详细分析")
        
        for model_name, result in comparison_results.items():
            report_lines.append(f"### {model_name}")
            
            # 性能指标
            performance = result['performance_metrics']
            report_lines.append("**性能指标:**")
            report_lines.append(f"- 准确率: {performance.get('accuracy', 0):.4f}")
            report_lines.append(f"- 精确率: {performance.get('precision', 0):.4f}")
            report_lines.append(f"- 召回率: {performance.get('recall', 0):.4f}")
            report_lines.append(f"- F1分数: {performance.get('f1_score', 0):.4f}")
            if 'roc_auc' in performance:
                report_lines.append(f"- ROC AUC: {performance['roc_auc']:.4f}")
            
            # 效率指标
            efficiency = result['efficiency_metrics']
            report_lines.append("**效率指标:**")
            report_lines.append(f"- 预测速度: {efficiency.get('predictions_per_second', 0):.2f} 预测/秒")
            report_lines.append(f"- 内存使用: {efficiency.get('memory_usage_mb', 0):.2f} MB")
            report_lines.append(f"- 批量预测时间: {efficiency.get('batch_prediction_time_s', 0):.4f} 秒")
            
            # 复杂度指标
            complexity = result['complexity_metrics']
            report_lines.append("**复杂度指标:**")
            report_lines.append(f"- 模型类型: {complexity.get('model_type', 'unknown')}")
            report_lines.append(f"- 可解释性评分: {complexity.get('interpretability_score', 0):.4f}")
            if 'model_size_mb' in complexity:
                report_lines.append(f"- 模型大小: {complexity['model_size_mb']:.2f} MB")
            
            # 鲁棒性指标
            robustness = result['robustness_metrics']
            report_lines.append("**鲁棒性指标:**")
            report_lines.append(f"- 综合鲁棒性评分: {robustness.get('overall_robustness_score', 0):.4f}")
            
            # 综合评分
            report_lines.append(f"**综合评分: {result['overall_score']:.4f}**")
            report_lines.append("")
        
        # 推荐建议
        recommendations = self._generate_model_recommendations(comparison_results)
        report_lines.append("## 推荐建议")
        
        for model_name, model_recommendations in recommendations.items():
            if model_recommendations:
                report_lines.append(f"### {model_name}")
                for rec in model_recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'reports', 'enhanced_model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path

def main():
    """主函数 - 演示增强模型评估流程"""
    try:
        print("增强模型评估模块已创建完成！")
        print("主要功能包括:")
        print("1. 运行效率对比（训练时间、预测时间、内存使用）")
        print("2. 模型复杂度分析（参数数量、模型大小、可解释性）")
        print("3. 异常输入处理能力测试（缺失值、异常值、噪声、分布变化）")
        print("4. 不同条件下的性能表现对比")
        print("5. 综合评分和推荐建议")
        print("6. 可视化对比图表生成")
        
    except Exception as e:
        print(f"增强模型评估过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
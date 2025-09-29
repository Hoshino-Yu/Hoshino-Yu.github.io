"""
交叉验证系统
实现K折交叉验证、时间序列交叉验证、分层交叉验证和模型泛化能力评估
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer, classification_report
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

class CrossValidationSystem:
    """
    交叉验证系统
    提供多种交叉验证方法和泛化能力评估
    """
    
    def __init__(self, output_dir: str = '../输出结果/cross_validation'):
        """
        初始化交叉验证系统
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.cv_results = {}
        self.generalization_results = {}
        
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
        log_file = os.path.join(self.output_dir, 'cross_validation.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_cross_validation(self, models_dict: Dict[str, Any], 
                                     X: pd.DataFrame, y: pd.Series,
                                     time_column: Optional[str] = None,
                                     group_column: Optional[str] = None) -> Dict:
        """
        综合交叉验证分析
        
        Args:
            models_dict: 模型字典 {model_name: model}
            X: 特征数据
            y: 标签数据
            time_column: 时间列名（用于时间序列交叉验证）
            group_column: 分组列名（用于分组交叉验证）
            
        Returns:
            综合交叉验证结果
        """
        self.logger.info("开始综合交叉验证分析")
        
        cv_results = {}
        
        for model_name, model in models_dict.items():
            self.logger.info(f"对模型 {model_name} 进行交叉验证")
            
            model_cv_results = {}
            
            # 1. K折交叉验证
            kfold_results = self._k_fold_cross_validation(model, X, y, model_name)
            model_cv_results['k_fold'] = kfold_results
            
            # 2. 分层交叉验证
            stratified_results = self._stratified_cross_validation(model, X, y, model_name)
            model_cv_results['stratified'] = stratified_results
            
            # 3. 时间序列交叉验证（如果提供时间列）
            if time_column and time_column in X.columns:
                timeseries_results = self._time_series_cross_validation(model, X, y, time_column, model_name)
                model_cv_results['time_series'] = timeseries_results
            
            # 4. 分组交叉验证（如果提供分组列）
            if group_column and group_column in X.columns:
                group_results = self._group_cross_validation(model, X, y, group_column, model_name)
                model_cv_results['group'] = group_results
            
            # 5. 学习曲线分析
            learning_curve_results = self._learning_curve_analysis(model, X, y, model_name)
            model_cv_results['learning_curve'] = learning_curve_results
            
            # 6. 验证曲线分析
            validation_curve_results = self._validation_curve_analysis(model, X, y, model_name)
            model_cv_results['validation_curve'] = validation_curve_results
            
            # 7. 泛化能力评估
            generalization_results = self._evaluate_generalization_ability(model, X, y, model_name)
            model_cv_results['generalization'] = generalization_results
            
            cv_results[model_name] = model_cv_results
        
        # 生成交叉验证对比图表
        self._generate_cv_comparison_charts(cv_results)
        
        # 生成交叉验证报告
        cv_report = self._generate_cv_report(cv_results)
        
        self.logger.info("综合交叉验证分析完成")
        
        return {
            'detailed_results': cv_results,
            'cv_report': cv_report,
            'best_cv_model': self._find_best_cv_model(cv_results),
            'cv_recommendations': self._generate_cv_recommendations(cv_results)
        }
    
    def _k_fold_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """K折交叉验证"""
        results = {}
        
        try:
            # 不同的K值测试
            k_values = [3, 5, 10]
            
            for k in k_values:
                kfold = KFold(n_splits=k, shuffle=True, random_state=42)
                
                # 定义评估指标
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score, zero_division=0),
                    'f1': make_scorer(f1_score, zero_division=0),
                    'roc_auc': 'roc_auc'
                }
                
                # 执行交叉验证
                cv_scores = cross_validate(model, X, y, cv=kfold, scoring=scoring, 
                                         return_train_score=True, n_jobs=-1)
                
                # 计算统计信息
                results[f'k_{k}'] = {
                    'test_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('test_')
                    },
                    'train_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('train_')
                    },
                    'fit_time': {
                        'mean': np.mean(cv_scores['fit_time']),
                        'std': np.std(cv_scores['fit_time'])
                    },
                    'score_time': {
                        'mean': np.mean(cv_scores['score_time']),
                        'std': np.std(cv_scores['score_time'])
                    }
                }
        
        except Exception as e:
            self.logger.error(f"K折交叉验证失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _stratified_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """分层交叉验证"""
        results = {}
        
        try:
            # 不同的K值测试
            k_values = [3, 5, 10]
            
            for k in k_values:
                stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                
                # 定义评估指标
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score, zero_division=0),
                    'f1': make_scorer(f1_score, zero_division=0),
                    'roc_auc': 'roc_auc'
                }
                
                # 执行分层交叉验证
                cv_scores = cross_validate(model, X, y, cv=stratified_kfold, scoring=scoring,
                                         return_train_score=True, n_jobs=-1)
                
                # 计算统计信息
                results[f'stratified_k_{k}'] = {
                    'test_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('test_')
                    },
                    'train_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('train_')
                    },
                    'class_distribution': self._analyze_class_distribution_in_folds(y, stratified_kfold)
                }
        
        except Exception as e:
            self.logger.error(f"分层交叉验证失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _time_series_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                                    time_column: str, model_name: str) -> Dict:
        """时间序列交叉验证"""
        results = {}
        
        try:
            # 按时间排序
            time_sorted_indices = X[time_column].sort_values().index
            X_sorted = X.loc[time_sorted_indices]
            y_sorted = y.loc[time_sorted_indices]
            
            # 移除时间列用于模型训练
            X_features = X_sorted.drop(columns=[time_column])
            
            # 不同的分割数测试
            n_splits_values = [3, 5, 8]
            
            for n_splits in n_splits_values:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                
                fold_results = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X_features)):
                    X_train_fold = X_features.iloc[train_idx]
                    X_test_fold = X_features.iloc[test_idx]
                    y_train_fold = y_sorted.iloc[train_idx]
                    y_test_fold = y_sorted.iloc[test_idx]
                    
                    # 训练模型
                    model_clone = clone(model)
                    model_clone.fit(X_train_fold, y_train_fold)
                    
                    # 预测和评估
                    y_pred = model_clone.predict(X_test_fold)
                    
                    fold_scores = {
                        'accuracy': accuracy_score(y_test_fold, y_pred),
                        'precision': precision_score(y_test_fold, y_pred, zero_division=0),
                        'recall': recall_score(y_test_fold, y_pred, zero_division=0),
                        'f1': f1_score(y_test_fold, y_pred, zero_division=0)
                    }
                    
                    # 如果模型支持概率预测
                    if hasattr(model_clone, 'predict_proba'):
                        y_pred_proba = model_clone.predict_proba(X_test_fold)[:, 1]
                        fold_scores['roc_auc'] = roc_auc_score(y_test_fold, y_pred_proba)
                    
                    fold_results.append({
                        'fold': fold,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx),
                        'scores': fold_scores
                    })
                
                # 计算平均分数
                avg_scores = {}
                for metric in fold_results[0]['scores'].keys():
                    scores = [fold['scores'][metric] for fold in fold_results]
                    avg_scores[metric] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'scores': scores
                    }
                
                results[f'timeseries_{n_splits}_splits'] = {
                    'avg_scores': avg_scores,
                    'fold_details': fold_results,
                    'temporal_stability': self._calculate_temporal_stability(fold_results)
                }
        
        except Exception as e:
            self.logger.error(f"时间序列交叉验证失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _group_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                              group_column: str, model_name: str) -> Dict:
        """分组交叉验证"""
        results = {}
        
        try:
            groups = X[group_column]
            X_features = X.drop(columns=[group_column])
            
            # 不同的分割数测试
            n_splits_values = [3, 5]
            
            for n_splits in n_splits_values:
                group_kfold = GroupKFold(n_splits=n_splits)
                
                # 定义评估指标
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score, zero_division=0),
                    'f1': make_scorer(f1_score, zero_division=0),
                    'roc_auc': 'roc_auc'
                }
                
                # 执行分组交叉验证
                cv_scores = cross_validate(model, X_features, y, groups=groups, 
                                         cv=group_kfold, scoring=scoring,
                                         return_train_score=True, n_jobs=-1)
                
                # 计算统计信息
                results[f'group_{n_splits}_splits'] = {
                    'test_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('test_')
                    },
                    'train_scores': {
                        metric: {
                            'mean': np.mean(scores),
                            'std': np.std(scores),
                            'scores': scores.tolist()
                        }
                        for metric, scores in cv_scores.items() if metric.startswith('train_')
                    },
                    'group_analysis': self._analyze_group_distribution(groups, group_kfold)
                }
        
        except Exception as e:
            self.logger.error(f"分组交叉验证失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _learning_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """学习曲线分析"""
        results = {}
        
        try:
            # 定义训练集大小
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            # 计算学习曲线
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=5, 
                scoring='accuracy', n_jobs=-1, random_state=42
            )
            
            results = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores': {
                    'mean': np.mean(train_scores, axis=1).tolist(),
                    'std': np.std(train_scores, axis=1).tolist(),
                    'all_scores': train_scores.tolist()
                },
                'test_scores': {
                    'mean': np.mean(test_scores, axis=1).tolist(),
                    'std': np.std(test_scores, axis=1).tolist(),
                    'all_scores': test_scores.tolist()
                },
                'overfitting_analysis': self._analyze_overfitting(train_scores, test_scores),
                'convergence_analysis': self._analyze_convergence(test_scores)
            }
        
        except Exception as e:
            self.logger.error(f"学习曲线分析失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _validation_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """验证曲线分析"""
        results = {}
        
        try:
            # 根据模型类型选择超参数
            param_name, param_range = self._get_validation_param(model)
            
            if param_name and param_range:
                # 计算验证曲线
                train_scores, test_scores = validation_curve(
                    model, X, y, param_name=param_name, param_range=param_range,
                    cv=5, scoring='accuracy', n_jobs=-1
                )
                
                results = {
                    'param_name': param_name,
                    'param_range': param_range,
                    'train_scores': {
                        'mean': np.mean(train_scores, axis=1).tolist(),
                        'std': np.std(train_scores, axis=1).tolist()
                    },
                    'test_scores': {
                        'mean': np.mean(test_scores, axis=1).tolist(),
                        'std': np.std(test_scores, axis=1).tolist()
                    },
                    'optimal_param': self._find_optimal_param(param_range, test_scores),
                    'complexity_analysis': self._analyze_model_complexity_curve(param_range, train_scores, test_scores)
                }
            else:
                results['message'] = '无法为此模型类型确定验证参数'
        
        except Exception as e:
            self.logger.error(f"验证曲线分析失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_generalization_ability(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """评估模型泛化能力"""
        results = {}
        
        try:
            # 1. 偏差-方差分解
            bias_variance = self._bias_variance_decomposition(model, X, y)
            results['bias_variance'] = bias_variance
            
            # 2. 稳定性测试
            stability_test = self._stability_test(model, X, y)
            results['stability'] = stability_test
            
            # 3. 数据扰动测试
            perturbation_test = self._data_perturbation_test(model, X, y)
            results['perturbation'] = perturbation_test
            
            # 4. 交叉验证稳定性
            cv_stability = self._cross_validation_stability(model, X, y)
            results['cv_stability'] = cv_stability
            
            # 5. 泛化能力评分
            generalization_score = self._calculate_generalization_score(results)
            results['generalization_score'] = generalization_score
        
        except Exception as e:
            self.logger.error(f"泛化能力评估失败 {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_class_distribution_in_folds(self, y: pd.Series, cv_splitter) -> Dict:
        """分析折中的类别分布"""
        fold_distributions = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(np.zeros(len(y)), y)):
            train_dist = y.iloc[train_idx].value_counts(normalize=True).to_dict()
            test_dist = y.iloc[test_idx].value_counts(normalize=True).to_dict()
            
            fold_distributions.append({
                'fold': fold,
                'train_distribution': train_dist,
                'test_distribution': test_dist
            })
        
        return {
            'fold_distributions': fold_distributions,
            'distribution_consistency': self._calculate_distribution_consistency(fold_distributions)
        }
    
    def _calculate_temporal_stability(self, fold_results: List[Dict]) -> Dict:
        """计算时间序列的稳定性"""
        # 提取各折的准确率
        accuracies = [fold['scores']['accuracy'] for fold in fold_results]
        
        # 计算趋势
        fold_indices = list(range(len(accuracies)))
        trend_slope = np.polyfit(fold_indices, accuracies, 1)[0]
        
        # 计算稳定性指标
        stability_metrics = {
            'accuracy_trend_slope': trend_slope,
            'accuracy_variance': np.var(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'accuracy_range': max(accuracies) - min(accuracies),
            'is_stable': abs(trend_slope) < 0.01 and np.var(accuracies) < 0.01
        }
        
        return stability_metrics
    
    def _analyze_group_distribution(self, groups: pd.Series, cv_splitter) -> Dict:
        """分析分组交叉验证中的组分布"""
        group_analysis = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(np.zeros(len(groups)), groups=groups)):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            
            group_analysis.append({
                'fold': fold,
                'train_groups': len(train_groups),
                'test_groups': len(test_groups),
                'group_overlap': len(train_groups.intersection(test_groups)),
                'train_group_list': list(train_groups),
                'test_group_list': list(test_groups)
            })
        
        return {
            'fold_analysis': group_analysis,
            'total_groups': len(set(groups)),
            'group_separation_quality': self._calculate_group_separation_quality(group_analysis)
        }
    
    def _analyze_overfitting(self, train_scores: np.ndarray, test_scores: np.ndarray) -> Dict:
        """分析过拟合情况"""
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        # 计算训练和测试分数的差距
        gap = train_mean - test_mean
        
        # 过拟合指标
        overfitting_metrics = {
            'max_gap': np.max(gap),
            'final_gap': gap[-1],
            'avg_gap': np.mean(gap),
            'gap_trend': np.polyfit(range(len(gap)), gap, 1)[0],
            'is_overfitting': gap[-1] > 0.1,  # 如果最终差距>10%认为过拟合
            'overfitting_severity': 'high' if gap[-1] > 0.2 else 'medium' if gap[-1] > 0.1 else 'low'
        }
        
        return overfitting_metrics
    
    def _analyze_convergence(self, test_scores: np.ndarray) -> Dict:
        """分析收敛情况"""
        test_mean = np.mean(test_scores, axis=1)
        
        # 计算最后几个点的变化
        if len(test_mean) >= 3:
            last_3_change = abs(test_mean[-1] - test_mean[-3])
            convergence_metrics = {
                'final_score': test_mean[-1],
                'score_improvement': test_mean[-1] - test_mean[0],
                'last_3_points_change': last_3_change,
                'is_converged': last_3_change < 0.01,
                'convergence_trend': np.polyfit(range(len(test_mean)), test_mean, 1)[0]
            }
        else:
            convergence_metrics = {
                'final_score': test_mean[-1],
                'score_improvement': test_mean[-1] - test_mean[0] if len(test_mean) > 1 else 0,
                'is_converged': True,
                'convergence_trend': 0
            }
        
        return convergence_metrics
    
    def _get_validation_param(self, model) -> Tuple[Optional[str], Optional[List]]:
        """根据模型类型获取验证参数"""
        model_type = type(model).__name__.lower()
        
        if 'randomforest' in model_type:
            return 'n_estimators', [10, 50, 100, 200, 300]
        elif 'svc' in model_type or 'svm' in model_type:
            return 'C', [0.1, 1, 10, 100, 1000]
        elif 'logisticregression' in model_type:
            return 'C', [0.01, 0.1, 1, 10, 100]
        elif 'decisiontree' in model_type:
            return 'max_depth', [3, 5, 7, 10, 15, None]
        elif 'kneighbors' in model_type:
            return 'n_neighbors', [3, 5, 7, 9, 11, 15]
        elif 'gradientboosting' in model_type:
            return 'n_estimators', [50, 100, 200, 300]
        else:
            return None, None
    
    def _find_optimal_param(self, param_range: List, test_scores: np.ndarray) -> Any:
        """找到最优参数"""
        test_mean = np.mean(test_scores, axis=1)
        optimal_idx = np.argmax(test_mean)
        return param_range[optimal_idx]
    
    def _analyze_model_complexity_curve(self, param_range: List, train_scores: np.ndarray, 
                                      test_scores: np.ndarray) -> Dict:
        """分析模型复杂度曲线"""
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        # 找到最佳复杂度点
        best_idx = np.argmax(test_mean)
        
        complexity_analysis = {
            'optimal_complexity_param': param_range[best_idx],
            'optimal_test_score': test_mean[best_idx],
            'underfitting_region': list(range(0, best_idx)) if best_idx > 0 else [],
            'overfitting_region': list(range(best_idx + 1, len(param_range))) if best_idx < len(param_range) - 1 else [],
            'complexity_recommendation': self._get_complexity_recommendation(param_range, train_mean, test_mean, best_idx)
        }
        
        return complexity_analysis
    
    def _get_complexity_recommendation(self, param_range: List, train_mean: np.ndarray, 
                                     test_mean: np.ndarray, best_idx: int) -> str:
        """获取复杂度建议"""
        if best_idx == 0:
            return "模型可能欠拟合，建议增加模型复杂度"
        elif best_idx == len(param_range) - 1:
            return "可能需要更高的复杂度，建议扩展参数范围"
        else:
            gap = train_mean[best_idx] - test_mean[best_idx]
            if gap > 0.1:
                return "在最优复杂度点存在过拟合，建议适当降低复杂度"
            else:
                return "模型复杂度适中，性能良好"
    
    def _bias_variance_decomposition(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """偏差-方差分解"""
        try:
            n_bootstrap = 50
            n_samples = min(1000, len(X))  # 限制样本数量以提高效率
            
            predictions = []
            
            for i in range(n_bootstrap):
                # Bootstrap采样
                bootstrap_indices = np.random.choice(len(X), n_samples, replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                y_bootstrap = y.iloc[bootstrap_indices]
                
                # 训练模型
                model_clone = clone(model)
                model_clone.fit(X_bootstrap, y_bootstrap)
                
                # 在原始测试集上预测
                test_indices = np.random.choice(len(X), min(200, len(X)), replace=False)
                X_test = X.iloc[test_indices]
                
                if hasattr(model_clone, 'predict_proba'):
                    pred = model_clone.predict_proba(X_test)[:, 1]
                else:
                    pred = model_clone.predict(X_test).astype(float)
                
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # 计算偏差和方差
            mean_prediction = np.mean(predictions, axis=0)
            variance = np.mean(np.var(predictions, axis=0))
            
            # 真实标签（用于计算偏差）
            y_true = y.iloc[test_indices].values
            bias_squared = np.mean((mean_prediction - y_true) ** 2)
            
            return {
                'bias_squared': bias_squared,
                'variance': variance,
                'bias_variance_tradeoff': bias_squared / (bias_squared + variance) if (bias_squared + variance) > 0 else 0,
                'total_error': bias_squared + variance
            }
        
        except Exception as e:
            self.logger.warning(f"偏差-方差分解失败: {str(e)}")
            return {'error': str(e)}
    
    def _stability_test(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """稳定性测试"""
        try:
            n_runs = 20
            scores = []
            
            for i in range(n_runs):
                # 随机划分训练测试集
                indices = np.random.permutation(len(X))
                split_point = int(0.8 * len(X))
                
                train_indices = indices[:split_point]
                test_indices = indices[split_point:]
                
                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]
                
                # 训练和评估
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                y_pred = model_clone.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
            
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'score_range': np.max(scores) - np.min(scores),
                'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                'is_stable': np.std(scores) < 0.05  # 标准差小于5%认为稳定
            }
        
        except Exception as e:
            self.logger.warning(f"稳定性测试失败: {str(e)}")
            return {'error': str(e)}
    
    def _data_perturbation_test(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """数据扰动测试"""
        try:
            # 原始性能
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            original_score = accuracy_score(y_test, model_clone.predict(X_test))
            
            perturbation_results = {}
            
            # 特征扰动测试
            noise_levels = [0.01, 0.05, 0.1]
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            
            for noise_level in noise_levels:
                X_test_noisy = X_test.copy()
                
                # 为数值特征添加噪声
                for col in numeric_cols:
                    noise = np.random.normal(0, X_test[col].std() * noise_level, len(X_test))
                    X_test_noisy[col] = X_test_noisy[col] + noise
                
                try:
                    noisy_score = accuracy_score(y_test, model_clone.predict(X_test_noisy))
                    performance_drop = original_score - noisy_score
                    
                    perturbation_results[f'noise_{noise_level}'] = {
                        'score': noisy_score,
                        'performance_drop': performance_drop,
                        'relative_drop': performance_drop / original_score if original_score > 0 else 0
                    }
                except:
                    perturbation_results[f'noise_{noise_level}'] = {
                        'score': 0,
                        'performance_drop': original_score,
                        'relative_drop': 1
                    }
            
            # 计算平均扰动敏感性
            avg_relative_drop = np.mean([result['relative_drop'] for result in perturbation_results.values()])
            
            return {
                'original_score': original_score,
                'perturbation_results': perturbation_results,
                'avg_sensitivity': avg_relative_drop,
                'is_robust': avg_relative_drop < 0.1  # 平均性能下降<10%认为鲁棒
            }
        
        except Exception as e:
            self.logger.warning(f"数据扰动测试失败: {str(e)}")
            return {'error': str(e)}
    
    def _cross_validation_stability(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """交叉验证稳定性"""
        try:
            # 多次运行交叉验证
            n_runs = 10
            cv_scores_runs = []
            
            for run in range(n_runs):
                kfold = KFold(n_splits=5, shuffle=True, random_state=run)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
                cv_scores_runs.append(scores)
            
            cv_scores_runs = np.array(cv_scores_runs)
            
            # 计算稳定性指标
            mean_scores_per_run = np.mean(cv_scores_runs, axis=1)
            
            return {
                'mean_cv_score': np.mean(mean_scores_per_run),
                'std_cv_score': np.std(mean_scores_per_run),
                'min_cv_score': np.min(mean_scores_per_run),
                'max_cv_score': np.max(mean_scores_per_run),
                'cv_score_range': np.max(mean_scores_per_run) - np.min(mean_scores_per_run),
                'cv_coefficient_of_variation': np.std(mean_scores_per_run) / np.mean(mean_scores_per_run) if np.mean(mean_scores_per_run) > 0 else 0,
                'is_cv_stable': np.std(mean_scores_per_run) < 0.02  # CV标准差<2%认为稳定
            }
        
        except Exception as e:
            self.logger.warning(f"交叉验证稳定性测试失败: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_generalization_score(self, generalization_results: Dict) -> float:
        """计算泛化能力评分"""
        score = 0.0
        weight_sum = 0.0
        
        # 偏差-方差权衡 (权重: 0.3)
        if 'bias_variance' in generalization_results and 'error' not in generalization_results['bias_variance']:
            bv_tradeoff = generalization_results['bias_variance'].get('bias_variance_tradeoff', 0)
            # 理想的偏差-方差权衡应该平衡，接近0.5
            bv_score = 1 - abs(bv_tradeoff - 0.5) * 2
            score += bv_score * 0.3
            weight_sum += 0.3
        
        # 稳定性 (权重: 0.4)
        if 'stability' in generalization_results and 'error' not in generalization_results['stability']:
            is_stable = generalization_results['stability'].get('is_stable', False)
            cv_score = generalization_results['stability'].get('coefficient_of_variation', 1)
            stability_score = 1.0 if is_stable else max(0, 1 - cv_score * 10)
            score += stability_score * 0.4
            weight_sum += 0.4
        
        # 扰动鲁棒性 (权重: 0.3)
        if 'perturbation' in generalization_results and 'error' not in generalization_results['perturbation']:
            is_robust = generalization_results['perturbation'].get('is_robust', False)
            sensitivity = generalization_results['perturbation'].get('avg_sensitivity', 1)
            robustness_score = 1.0 if is_robust else max(0, 1 - sensitivity * 5)
            score += robustness_score * 0.3
            weight_sum += 0.3
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_distribution_consistency(self, fold_distributions: List[Dict]) -> float:
        """计算分布一致性"""
        if not fold_distributions:
            return 0.0
        
        # 计算各折之间类别分布的一致性
        train_distributions = [fold['train_distribution'] for fold in fold_distributions]
        test_distributions = [fold['test_distribution'] for fold in fold_distributions]
        
        # 计算训练集分布的标准差
        train_class_0_ratios = [dist.get(0, 0) for dist in train_distributions]
        test_class_0_ratios = [dist.get(0, 0) for dist in test_distributions]
        
        train_consistency = 1 - np.std(train_class_0_ratios)
        test_consistency = 1 - np.std(test_class_0_ratios)
        
        return (train_consistency + test_consistency) / 2
    
    def _calculate_group_separation_quality(self, group_analysis: List[Dict]) -> float:
        """计算组分离质量"""
        if not group_analysis:
            return 0.0
        
        # 检查组之间是否有重叠
        overlaps = [fold['group_overlap'] for fold in group_analysis]
        total_overlaps = sum(overlaps)
        
        # 如果没有重叠，分离质量为1.0
        if total_overlaps == 0:
            return 1.0
        
        # 否则根据重叠程度计算质量
        total_groups = sum([fold['train_groups'] + fold['test_groups'] for fold in group_analysis])
        separation_quality = 1 - (total_overlaps / total_groups) if total_groups > 0 else 0
        
        return max(0, separation_quality)
    
    def _generate_cv_comparison_charts(self, cv_results: Dict):
        """生成交叉验证对比图表"""
        
        # 1. K折交叉验证对比
        self._create_kfold_comparison_chart(cv_results)
        
        # 2. 学习曲线对比
        self._create_learning_curves_chart(cv_results)
        
        # 3. 泛化能力雷达图
        self._create_generalization_radar_chart(cv_results)
        
        # 4. 交叉验证稳定性对比
        self._create_cv_stability_chart(cv_results)
    
    def _create_kfold_comparison_chart(self, cv_results: Dict):
        """创建K折交叉验证对比图表"""
        
        models = list(cv_results.keys())
        k_values = [3, 5, 10]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['准确率对比', '精确率对比', '召回率对比', 'F1分数对比']
        )
        
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (metric, pos) in enumerate(zip(metrics, positions)):
            for model_name in models:
                if 'k_fold' in cv_results[model_name]:
                    k_fold_results = cv_results[model_name]['k_fold']
                    
                    means = []
                    stds = []
                    
                    for k in k_values:
                        k_key = f'k_{k}'
                        if k_key in k_fold_results and 'test_scores' in k_fold_results[k_key]:
                            test_scores = k_fold_results[k_key]['test_scores']
                            if metric in test_scores:
                                means.append(test_scores[metric]['mean'])
                                stds.append(test_scores[metric]['std'])
                            else:
                                means.append(0)
                                stds.append(0)
                        else:
                            means.append(0)
                            stds.append(0)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=k_values,
                            y=means,
                            error_y=dict(type='data', array=stds),
                            mode='lines+markers',
                            name=f'{model_name}',
                            showlegend=(i == 0)  # 只在第一个子图显示图例
                        ),
                        row=pos[0], col=pos[1]
                    )
        
        fig.update_layout(height=800, title_text="K折交叉验证性能对比")
        fig.write_html(os.path.join(self.output_dir, 'charts', 'kfold_comparison.html'))
    
    def _create_learning_curves_chart(self, cv_results: Dict):
        """创建学习曲线图表"""
        
        fig = go.Figure()
        
        for model_name, results in cv_results.items():
            if 'learning_curve' in results and 'error' not in results['learning_curve']:
                lc_results = results['learning_curve']
                
                train_sizes = lc_results['train_sizes']
                train_mean = lc_results['train_scores']['mean']
                train_std = lc_results['train_scores']['std']
                test_mean = lc_results['test_scores']['mean']
                test_std = lc_results['test_scores']['std']
                
                # 训练分数
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    error_y=dict(type='data', array=train_std),
                    mode='lines+markers',
                    name=f'{model_name} - 训练',
                    line=dict(dash='dash')
                ))
                
                # 测试分数
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=test_mean,
                    error_y=dict(type='data', array=test_std),
                    mode='lines+markers',
                    name=f'{model_name} - 验证'
                ))
        
        fig.update_layout(
            title='学习曲线对比',
            xaxis_title='训练样本数量',
            yaxis_title='准确率',
            height=600
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'learning_curves.html'))
    
    def _create_generalization_radar_chart(self, cv_results: Dict):
        """创建泛化能力雷达图"""
        
        models = list(cv_results.keys())
        metrics = ['稳定性', '鲁棒性', '偏差-方差权衡', '交叉验证稳定性', '泛化评分']
        
        fig = go.Figure()
        
        for model_name in models:
            if 'generalization' in cv_results[model_name]:
                gen_results = cv_results[model_name]['generalization']
                
                # 提取各维度评分
                stability_score = 1.0 if gen_results.get('stability', {}).get('is_stable', False) else 0.5
                robustness_score = 1.0 if gen_results.get('perturbation', {}).get('is_robust', False) else 0.5
                bv_score = 1 - abs(gen_results.get('bias_variance', {}).get('bias_variance_tradeoff', 0.5) - 0.5) * 2
                cv_stability_score = 1.0 if gen_results.get('cv_stability', {}).get('is_cv_stable', False) else 0.5
                generalization_score = gen_results.get('generalization_score', 0)
                
                values = [stability_score, robustness_score, bv_score, cv_stability_score, generalization_score]
                
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
            title="模型泛化能力雷达图"
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'generalization_radar.html'))
    
    def _create_cv_stability_chart(self, cv_results: Dict):
        """创建交叉验证稳定性图表"""
        
        models = list(cv_results.keys())
        
        # 提取稳定性指标
        stability_data = []
        cv_stability_data = []
        
        for model_name in models:
            if 'generalization' in cv_results[model_name]:
                gen_results = cv_results[model_name]['generalization']
                
                # 稳定性测试结果
                if 'stability' in gen_results:
                    stability_cv = gen_results['stability'].get('coefficient_of_variation', 0)
                    stability_data.append(stability_cv)
                else:
                    stability_data.append(0)
                
                # 交叉验证稳定性
                if 'cv_stability' in gen_results:
                    cv_stability_cv = gen_results['cv_stability'].get('cv_coefficient_of_variation', 0)
                    cv_stability_data.append(cv_stability_cv)
                else:
                    cv_stability_data.append(0)
        
        fig = go.Figure(data=[
            go.Bar(name='稳定性测试变异系数', x=models, y=stability_data),
            go.Bar(name='交叉验证变异系数', x=models, y=cv_stability_data)
        ])
        
        fig.update_layout(
            title='模型稳定性对比（变异系数越小越稳定）',
            xaxis_title='模型',
            yaxis_title='变异系数',
            barmode='group',
            height=500
        )
        
        fig.write_html(os.path.join(self.output_dir, 'charts', 'cv_stability.html'))
    
    def _find_best_cv_model(self, cv_results: Dict) -> Dict:
        """找出交叉验证最佳模型"""
        
        best_models = {}
        
        # 最佳K折交叉验证性能
        best_kfold_accuracy = 0
        best_kfold_model = None
        
        for model_name, results in cv_results.items():
            if 'k_fold' in results:
                k_fold_results = results['k_fold']
                # 使用5折的结果
                if 'k_5' in k_fold_results and 'test_scores' in k_fold_results['k_5']:
                    accuracy = k_fold_results['k_5']['test_scores'].get('test_accuracy', {}).get('mean', 0)
                    if accuracy > best_kfold_accuracy:
                        best_kfold_accuracy = accuracy
                        best_kfold_model = model_name
        
        best_models['best_kfold'] = {
            'model': best_kfold_model,
            'accuracy': best_kfold_accuracy
        }
        
        # 最佳泛化能力
        best_generalization_score = 0
        best_generalization_model = None
        
        for model_name, results in cv_results.items():
            if 'generalization' in results:
                gen_score = results['generalization'].get('generalization_score', 0)
                if gen_score > best_generalization_score:
                    best_generalization_score = gen_score
                    best_generalization_model = model_name
        
        best_models['best_generalization'] = {
            'model': best_generalization_model,
            'score': best_generalization_score
        }
        
        return best_models
    
    def _generate_cv_recommendations(self, cv_results: Dict) -> Dict:
        """生成交叉验证建议"""
        
        recommendations = {}
        
        for model_name, results in cv_results.items():
            model_recommendations = []
            
            # K折交叉验证建议
            if 'k_fold' in results:
                k_fold_results = results['k_fold']
                
                # 检查不同K值的稳定性
                k_accuracies = []
                for k in [3, 5, 10]:
                    k_key = f'k_{k}'
                    if k_key in k_fold_results and 'test_scores' in k_fold_results[k_key]:
                        acc = k_fold_results[k_key]['test_scores'].get('test_accuracy', {}).get('mean', 0)
                        k_accuracies.append(acc)
                
                if len(k_accuracies) >= 2:
                    k_std = np.std(k_accuracies)
                    if k_std > 0.05:
                        model_recommendations.append("不同K值下性能差异较大，建议检查数据分布或模型稳定性")
            
            # 学习曲线建议
            if 'learning_curve' in results and 'error' not in results['learning_curve']:
                lc_results = results['learning_curve']
                
                if 'overfitting_analysis' in lc_results:
                    overfitting = lc_results['overfitting_analysis']
                    if overfitting.get('is_overfitting', False):
                        severity = overfitting.get('overfitting_severity', 'medium')
                        if severity == 'high':
                            model_recommendations.append("存在严重过拟合，建议增加正则化或减少模型复杂度")
                        else:
                            model_recommendations.append("存在轻微过拟合，建议调整超参数")
                
                if 'convergence_analysis' in lc_results:
                    convergence = lc_results['convergence_analysis']
                    if not convergence.get('is_converged', True):
                        model_recommendations.append("学习曲线未收敛，建议增加训练数据或调整学习率")
            
            # 泛化能力建议
            if 'generalization' in results:
                gen_results = results['generalization']
                gen_score = gen_results.get('generalization_score', 0)
                
                if gen_score < 0.5:
                    model_recommendations.append("泛化能力较差，建议进行特征选择或模型调优")
                elif gen_score > 0.8:
                    model_recommendations.append("泛化能力优秀，适合部署到生产环境")
                
                # 稳定性建议
                if 'stability' in gen_results:
                    stability = gen_results['stability']
                    if not stability.get('is_stable', True):
                        model_recommendations.append("模型稳定性不足，建议增加训练数据或使用集成方法")
                
                # 鲁棒性建议
                if 'perturbation' in gen_results:
                    perturbation = gen_results['perturbation']
                    if not perturbation.get('is_robust', True):
                        model_recommendations.append("对数据扰动敏感，建议加强数据预处理和特征工程")
            
            recommendations[model_name] = model_recommendations
        
        return recommendations
    
    def _generate_cv_report(self, cv_results: Dict) -> str:
        """生成交叉验证报告"""
        
        report_lines = []
        report_lines.append("# 交叉验证分析报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 概述
        report_lines.append("## 概述")
        report_lines.append(f"本次交叉验证分析了 {len(cv_results)} 个模型，包括K折交叉验证、分层交叉验证、学习曲线分析和泛化能力评估。")
        report_lines.append("")
        
        # 最佳模型
        best_models = self._find_best_cv_model(cv_results)
        report_lines.append("## 最佳模型")
        
        if 'best_kfold' in best_models and best_models['best_kfold']['model']:
            report_lines.append(f"- **K折交叉验证最佳**: {best_models['best_kfold']['model']} (准确率: {best_models['best_kfold']['accuracy']:.4f})")
        
        if 'best_generalization' in best_models and best_models['best_generalization']['model']:
            report_lines.append(f"- **泛化能力最佳**: {best_models['best_generalization']['model']} (泛化评分: {best_models['best_generalization']['score']:.4f})")
        
        report_lines.append("")
        
        # 详细分析
        report_lines.append("## 详细分析")
        
        for model_name, results in cv_results.items():
            report_lines.append(f"### {model_name}")
            
            # K折交叉验证结果
            if 'k_fold' in results:
                report_lines.append("**K折交叉验证:**")
                k_fold_results = results['k_fold']
                
                for k in [3, 5, 10]:
                    k_key = f'k_{k}'
                    if k_key in k_fold_results and 'test_scores' in k_fold_results[k_key]:
                        test_scores = k_fold_results[k_key]['test_scores']
                        if 'test_accuracy' in test_scores:
                            acc_mean = test_scores['test_accuracy']['mean']
                            acc_std = test_scores['test_accuracy']['std']
                            report_lines.append(f"- {k}折: {acc_mean:.4f} ± {acc_std:.4f}")
            
            # 泛化能力评估
            if 'generalization' in results:
                gen_results = results['generalization']
                report_lines.append("**泛化能力评估:**")
                
                gen_score = gen_results.get('generalization_score', 0)
                report_lines.append(f"- 泛化评分: {gen_score:.4f}")
                
                if 'stability' in gen_results:
                    stability = gen_results['stability']
                    is_stable = stability.get('is_stable', False)
                    cv_coeff = stability.get('coefficient_of_variation', 0)
                    report_lines.append(f"- 稳定性: {'稳定' if is_stable else '不稳定'} (变异系数: {cv_coeff:.4f})")
                
                if 'perturbation' in gen_results:
                    perturbation = gen_results['perturbation']
                    is_robust = perturbation.get('is_robust', False)
                    sensitivity = perturbation.get('avg_sensitivity', 0)
                    report_lines.append(f"- 鲁棒性: {'鲁棒' if is_robust else '敏感'} (平均敏感性: {sensitivity:.4f})")
            
            report_lines.append("")
        
        # 建议
        recommendations = self._generate_cv_recommendations(cv_results)
        if recommendations:
            report_lines.append("## 建议")
            for model_name, model_recs in recommendations.items():
                if model_recs:
                    report_lines.append(f"### {model_name}")
                    for rec in model_recs:
                        report_lines.append(f"- {rec}")
                    report_lines.append("")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, 'reports', 'cross_validation_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"交叉验证报告已保存到: {report_file}")
        
        return report_content
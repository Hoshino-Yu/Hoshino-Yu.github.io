#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SME信贷风险评估系统 - 模型评估模块
实现全面的模型性能评估、可视化分析和业务指标计算
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
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
from sklearn.model_selection import learning_curve, validation_curve
import shap
from lime import lime_tabular

warnings.filterwarnings('ignore')

class ComprehensiveModelEvaluator:
    """
    全面的模型评估器
    支持多维度性能评估、可视化分析、业务指标计算和模型解释
    """
    
    def __init__(self, output_dir: str = '../输出结果/evaluation'):
        """
        初始化模型评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.evaluation_results = {}
        self.visualizations = {}
        self.business_metrics = {}
        
        # 设置日志
        self._setup_logging()
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'evaluation.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                               model_name: str = "Model", 
                               X_train: Optional[pd.DataFrame] = None,
                               y_train: Optional[pd.Series] = None) -> Dict:
        """
        全面模型评估
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            X_train: 训练特征（可选，用于学习曲线等分析）
            y_train: 训练标签（可选）
            
        Returns:
            评估结果字典
        """
        self.logger.info(f"开始全面评估模型: {model_name}")
        
        # 获取预测结果
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 基础性能指标
        basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        
        # 高级性能指标
        advanced_metrics = self._calculate_advanced_metrics(y_test, y_pred, y_pred_proba)
        
        # 业务指标
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        
        # 模型稳定性指标
        stability_metrics = self._calculate_stability_metrics(model, X_test, y_test)
        
        # 合并所有指标
        evaluation_result = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'business_metrics': business_metrics,
            'stability_metrics': stability_metrics,
            'evaluation_time': datetime.now().isoformat()
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        # 生成可视化
        self._generate_evaluation_visualizations(
            model, X_test, y_test, y_pred, y_pred_proba, model_name
        )
        
        # 如果有训练数据，生成学习曲线
        if X_train is not None and y_train is not None:
            self._generate_learning_curves(model, X_train, y_train, model_name)
        
        self.logger.info(f"模型 {model_name} 评估完成")
        
        return evaluation_result
    
    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算基础性能指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_class_0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'precision_class_1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'recall_class_0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall_class_1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_class_0': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            'f1_class_1': f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'average_precision': average_precision_score(y_true, y_pred_proba),
                'log_loss': log_loss(y_true, y_pred_proba),
                'brier_score': brier_score_loss(y_true, y_pred_proba)
            })
        
        return metrics
    
    def _calculate_advanced_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算高级性能指标"""
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'matthews_correlation_coefficient': self._calculate_mcc(tp, tn, fp, fn)
        }
        
        if y_pred_proba is not None:
            # 计算不同阈值下的指标
            thresholds = np.arange(0.1, 1.0, 0.1)
            threshold_metrics = {}
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                threshold_metrics[f'threshold_{threshold:.1f}'] = {
                    'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                    'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
                    'f1': f1_score(y_true, y_pred_thresh, zero_division=0)
                }
            
            metrics['threshold_analysis'] = threshold_metrics
            
            # 校准指标
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred_proba, n_bins=10
                )
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                metrics['calibration_error'] = calibration_error
            except:
                metrics['calibration_error'] = None
        
        return metrics
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """计算马修斯相关系数"""
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0
        return (tp * tn - fp * fn) / denominator
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算业务指标"""
        # 假设业务成本参数
        cost_params = {
            'cost_false_positive': 1000,  # 误判为高风险的成本
            'cost_false_negative': 5000,  # 漏判高风险的成本
            'cost_true_positive': 100,    # 正确识别高风险的成本
            'cost_true_negative': 50      # 正确识别低风险的成本
        }
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算总成本
        total_cost = (
            tn * cost_params['cost_true_negative'] +
            fp * cost_params['cost_false_positive'] +
            fn * cost_params['cost_false_negative'] +
            tp * cost_params['cost_true_positive']
        )
        
        # 计算收益指标
        total_samples = len(y_true)
        high_risk_samples = np.sum(y_true == 1)
        low_risk_samples = np.sum(y_true == 0)
        
        metrics = {
            'total_cost': total_cost,
            'average_cost_per_sample': total_cost / total_samples,
            'cost_savings_vs_random': self._calculate_cost_savings(y_true, y_pred, cost_params),
            'high_risk_detection_rate': tp / high_risk_samples if high_risk_samples > 0 else 0,
            'low_risk_approval_rate': tn / low_risk_samples if low_risk_samples > 0 else 0,
            'precision_at_top_10_percent': self._calculate_precision_at_k(y_true, y_pred_proba, 0.1) if y_pred_proba is not None else None,
            'precision_at_top_20_percent': self._calculate_precision_at_k(y_true, y_pred_proba, 0.2) if y_pred_proba is not None else None,
        }
        
        # 计算不同风险阈值下的业务指标
        if y_pred_proba is not None:
            risk_thresholds = [0.3, 0.5, 0.7, 0.9]
            threshold_business_metrics = {}
            
            for threshold in risk_thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                cm_thresh = confusion_matrix(y_true, y_pred_thresh)
                tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
                
                cost_thresh = (
                    tn_t * cost_params['cost_true_negative'] +
                    fp_t * cost_params['cost_false_positive'] +
                    fn_t * cost_params['cost_false_negative'] +
                    tp_t * cost_params['cost_true_positive']
                )
                
                threshold_business_metrics[f'threshold_{threshold}'] = {
                    'total_cost': cost_thresh,
                    'high_risk_detection_rate': tp_t / high_risk_samples if high_risk_samples > 0 else 0,
                    'false_alarm_rate': fp_t / low_risk_samples if low_risk_samples > 0 else 0
                }
            
            metrics['threshold_business_analysis'] = threshold_business_metrics
        
        return metrics
    
    def _calculate_cost_savings(self, y_true: pd.Series, y_pred: np.ndarray, cost_params: Dict) -> float:
        """计算相对于随机预测的成本节省"""
        # 随机预测的期望成本
        p_positive = np.mean(y_true)
        random_cost = (
            p_positive * (1 - p_positive) * cost_params['cost_false_positive'] +
            (1 - p_positive) * p_positive * cost_params['cost_false_negative'] +
            p_positive * p_positive * cost_params['cost_true_positive'] +
            (1 - p_positive) * (1 - p_positive) * cost_params['cost_true_negative']
        ) * len(y_true)
        
        # 实际模型成本
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        actual_cost = (
            tn * cost_params['cost_true_negative'] +
            fp * cost_params['cost_false_positive'] +
            fn * cost_params['cost_false_negative'] +
            tp * cost_params['cost_true_positive']
        )
        
        return random_cost - actual_cost
    
    def _calculate_precision_at_k(self, y_true: pd.Series, y_pred_proba: np.ndarray, k: float) -> float:
        """计算前k%样本的精确率"""
        n_samples = int(len(y_true) * k)
        top_k_indices = np.argsort(y_pred_proba)[-n_samples:]
        return np.mean(y_true.iloc[top_k_indices])
    
    def _calculate_stability_metrics(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """计算模型稳定性指标"""
        metrics = {}
        
        try:
            # 特征重要性稳定性（如果模型支持）
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                # 计算特征重要性的基尼系数（衡量集中度）
                sorted_importance = np.sort(feature_importance)[::-1]
                n = len(sorted_importance)
                gini_coeff = (2 * np.sum((np.arange(n) + 1) * sorted_importance)) / (n * np.sum(sorted_importance)) - (n + 1) / n
                metrics['feature_importance_gini'] = gini_coeff
                
                # 前10个特征的重要性占比
                top_10_importance = np.sum(sorted_importance[:min(10, n)])
                metrics['top_10_features_importance_ratio'] = top_10_importance / np.sum(sorted_importance)
            
            # 预测概率分布稳定性
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 预测概率的统计特征
                metrics['prediction_mean'] = np.mean(y_pred_proba)
                metrics['prediction_std'] = np.std(y_pred_proba)
                metrics['prediction_skewness'] = self._calculate_skewness(y_pred_proba)
                metrics['prediction_kurtosis'] = self._calculate_kurtosis(y_pred_proba)
                
                # 预测概率分布的KS统计量
                from scipy import stats
                ks_stat, ks_p_value = stats.kstest(y_pred_proba, 'uniform')
                metrics['ks_statistic'] = ks_stat
                metrics['ks_p_value'] = ks_p_value
        
        except Exception as e:
            self.logger.warning(f"计算稳定性指标时出错: {str(e)}")
        
        return metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _generate_evaluation_visualizations(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                          y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray],
                                          model_name: str):
        """生成评估可视化图表"""
        self.logger.info(f"生成 {model_name} 的可视化图表")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '混淆矩阵', 'ROC曲线', 'PR曲线',
                '特征重要性', '预测概率分布', '校准曲线',
                '阈值分析', '业务成本分析', '预测vs实际'
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['预测负类', '预测正类'],
                y=['实际负类', '实际正类'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=1, col=1
        )
        
        if y_pred_proba is not None:
            # 2. ROC曲线
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {auc_score:.3f})',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='随机分类器',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
            
            # 3. PR曲线
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'PR (AP = {avg_precision:.3f})',
                    line=dict(color='green', width=2)
                ),
                row=1, col=3
            )
            
            # 5. 预测概率分布
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[y_test == 0],
                    name='负类',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[y_test == 1],
                    name='正类',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=2, col=2
            )
            
            # 6. 校准曲线
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=mean_predicted_value,
                        y=fraction_of_positives,
                        mode='markers+lines',
                        name='校准曲线',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=3
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='完美校准',
                        line=dict(color='red', dash='dash')
                    ),
                    row=2, col=3
                )
            except:
                pass
        
        # 4. 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_names = X_test.columns[:len(feature_importance)]
            
            # 选择前15个最重要的特征
            top_indices = np.argsort(feature_importance)[-15:]
            
            fig.add_trace(
                go.Bar(
                    x=feature_importance[top_indices],
                    y=[feature_names[i] for i in top_indices],
                    orientation='h',
                    name='特征重要性'
                ),
                row=2, col=1
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text=f"{model_name} 模型评估报告",
            showlegend=True
        )
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f'{model_name}_evaluation_report.html')
        fig.write_html(output_path)
        
        self.visualizations[model_name] = output_path
        self.logger.info(f"可视化报告已保存: {output_path}")
    
    def _generate_learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series, model_name: str):
        """生成学习曲线"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='roc_auc'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # 训练分数
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='训练分数',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # 验证分数
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='验证分数',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title=f'{model_name} 学习曲线',
                xaxis_title='训练样本数',
                yaxis_title='AUC分数',
                height=500
            )
            
            # 保存学习曲线
            learning_curve_path = os.path.join(self.output_dir, f'{model_name}_learning_curve.html')
            fig.write_html(learning_curve_path)
            
            self.logger.info(f"学习曲线已保存: {learning_curve_path}")
            
        except Exception as e:
            self.logger.warning(f"生成学习曲线失败: {str(e)}")
    
    def model_interpretability_analysis(self, model, X_test: pd.DataFrame, 
                                      model_name: str, sample_size: int = 100) -> Dict:
        """
        模型可解释性分析
        
        Args:
            model: 训练好的模型
            X_test: 测试数据
            model_name: 模型名称
            sample_size: 分析样本数量
            
        Returns:
            可解释性分析结果
        """
        self.logger.info(f"开始 {model_name} 的可解释性分析")
        
        interpretability_results = {}
        
        # 选择样本
        sample_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        try:
            # SHAP分析
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                
                # 保存SHAP摘要图
                shap_summary_path = os.path.join(self.output_dir, f'{model_name}_shap_summary.png')
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                interpretability_results['shap_summary_plot'] = shap_summary_path
                
                # SHAP特征重要性
                feature_importance_shap = np.mean(np.abs(shap_values.values), axis=0)
                interpretability_results['shap_feature_importance'] = {
                    'features': list(X_sample.columns),
                    'importance': feature_importance_shap.tolist()
                }
                
                self.logger.info("SHAP分析完成")
        
        except Exception as e:
            self.logger.warning(f"SHAP分析失败: {str(e)}")
        
        try:
            # LIME分析
            explainer_lime = lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns,
                class_names=['低风险', '高风险'],
                mode='classification'
            )
            
            # 分析几个样本
            lime_explanations = []
            for i in range(min(5, len(X_sample))):
                exp = explainer_lime.explain_instance(
                    X_sample.iloc[i].values,
                    model.predict_proba,
                    num_features=10
                )
                
                lime_explanations.append({
                    'sample_index': i,
                    'explanation': exp.as_list(),
                    'prediction_proba': model.predict_proba(X_sample.iloc[i:i+1])[0].tolist()
                })
            
            interpretability_results['lime_explanations'] = lime_explanations
            self.logger.info("LIME分析完成")
            
        except Exception as e:
            self.logger.warning(f"LIME分析失败: {str(e)}")
        
        return interpretability_results
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> Dict:
        """
        比较多个模型的性能
        
        Args:
            evaluation_results: 多个模型的评估结果
            
        Returns:
            模型比较结果
        """
        self.logger.info("开始模型比较分析")
        
        # 提取关键指标
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        comparison_data = {}
        
        for model_name, results in evaluation_results.items():
            comparison_data[model_name] = {}
            for metric in comparison_metrics:
                if metric in results.get('basic_metrics', {}):
                    comparison_data[model_name][metric] = results['basic_metrics'][metric]
        
        # 创建比较DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        # 生成比较可视化
        fig = go.Figure()
        
        for metric in comparison_metrics:
            if metric in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(comparison_df.index),
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='模型性能比较',
            xaxis_title='模型',
            yaxis_title='分数',
            barmode='group',
            height=600
        )
        
        # 保存比较图表
        comparison_path = os.path.join(self.output_dir, 'model_comparison.html')
        fig.write_html(comparison_path)
        
        # 排名分析
        rankings = {}
        for metric in comparison_metrics:
            if metric in comparison_df.columns:
                rankings[metric] = comparison_df[metric].rank(ascending=False).to_dict()
        
        # 综合排名（基于平均排名）
        avg_rankings = {}
        for model in comparison_df.index:
            avg_rank = np.mean([rankings[metric][model] for metric in comparison_metrics if metric in rankings])
            avg_rankings[model] = avg_rank
        
        best_model = min(avg_rankings.keys(), key=lambda x: avg_rankings[x])
        
        comparison_result = {
            'comparison_table': comparison_df.to_dict(),
            'rankings': rankings,
            'average_rankings': avg_rankings,
            'best_model': best_model,
            'comparison_chart': comparison_path
        }
        
        self.logger.info(f"模型比较完成，最佳模型: {best_model}")
        
        return comparison_result
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            
        Returns:
            报告文件路径
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"模型 {model_name} 的评估结果不存在")
        
        results = self.evaluation_results[model_name]
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} 模型评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric-table {{ border-collapse: collapse; width: 100%; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{model_name} 模型评估报告</h1>
                <p>生成时间: {results['evaluation_time']}</p>
            </div>
            
            <div class="section">
                <h2>基础性能指标</h2>
                <table class="metric-table">
                    <tr><th>指标</th><th>值</th></tr>
        """
        
        # 添加基础指标
        for metric, value in results['basic_metrics'].items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>业务指标</h2>
                <table class="metric-table">
                    <tr><th>指标</th><th>值</th></tr>
        """
        
        # 添加业务指标
        for metric, value in results['business_metrics'].items():
            if isinstance(value, (int, float)):
                html_content += f"<tr><td>{metric}</td><td>{value:.2f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>模型建议</h2>
                <ul>
        """
        
        # 根据性能给出建议
        auc_score = results['basic_metrics'].get('roc_auc', 0)
        if auc_score > 0.8:
            html_content += "<li>模型性能优秀，可以投入生产使用</li>"
        elif auc_score > 0.7:
            html_content += "<li>模型性能良好，建议进一步优化后使用</li>"
        else:
            html_content += "<li>模型性能需要改进，建议重新训练或调整特征</li>"
        
        precision = results['basic_metrics'].get('precision', 0)
        recall = results['basic_metrics'].get('recall', 0)
        
        if precision > recall:
            html_content += "<li>模型倾向于保守预测，适合风险厌恶场景</li>"
        elif recall > precision:
            html_content += "<li>模型倾向于激进预测，适合风险敏感场景</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        report_path = os.path.join(self.output_dir, f'{model_name}_evaluation_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"评估报告已保存: {report_path}")
        
        return report_path
    
    def save_evaluation_results(self, filename: str = 'evaluation_results.json'):
        """保存评估结果"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"评估结果已保存: {output_path}")


def main():
    """主函数 - 演示模型评估流程"""
    try:
        # 这里应该加载训练好的模型和测试数据
        # 由于这是演示，我们创建一个简单的示例
        
        print("模型评估模块已创建完成！")
        print("主要功能包括:")
        print("1. 全面的性能指标计算")
        print("2. 业务指标分析")
        print("3. 可视化报告生成")
        print("4. 模型可解释性分析")
        print("5. 多模型比较")
        print("6. 自动化报告生成")
        
    except Exception as e:
        print(f"模型评估过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
"""
智能报告生成系统
用于自动化生成各类数据分析和模型评估报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Optional, Tuple
from jinja2 import Template
import base64
from io import BytesIO
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class IntelligentReportGenerator:
    """智能报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self.setup_logging()
        self.create_output_directory()
        
        # 报告模板配置
        self.report_templates = {
            'data_analysis': self._get_data_analysis_template(),
            'model_evaluation': self._get_model_evaluation_template(),
            'risk_assessment': self._get_risk_assessment_template(),
            'business_summary': self._get_business_summary_template()
        }
        
        # 图表样式配置
        self.chart_config = {
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'figure_size': (12, 8),
            'dpi': 300,
            'style': 'whitegrid'
        }
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/report_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_output_directory(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.output_dir}/data", exist_ok=True)
    
    def generate_data_analysis_report(self, data: pd.DataFrame, 
                                    target_column: str = None,
                                    report_name: str = "数据分析报告") -> str:
        """
        生成数据分析报告
        
        Args:
            data: 数据集
            target_column: 目标变量列名
            report_name: 报告名称
            
        Returns:
            报告文件路径
        """
        self.logger.info(f"开始生成数据分析报告: {report_name}")
        
        try:
            # 基础统计分析
            basic_stats = self._analyze_basic_statistics(data)
            
            # 数据质量分析
            quality_analysis = self._analyze_data_quality(data)
            
            # 特征分析
            feature_analysis = self._analyze_features(data, target_column)
            
            # 相关性分析
            correlation_analysis = self._analyze_correlations(data)
            
            # 生成图表
            charts = self._generate_data_analysis_charts(data, target_column)
            
            # 生成报告
            report_data = {
                'report_name': report_name,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_shape': data.shape,
                'basic_stats': basic_stats,
                'quality_analysis': quality_analysis,
                'feature_analysis': feature_analysis,
                'correlation_analysis': correlation_analysis,
                'charts': charts
            }
            
            # 渲染HTML报告
            html_content = self.report_templates['data_analysis'].render(**report_data)
            
            # 保存报告
            report_path = f"{self.output_dir}/{report_name}_数据分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"数据分析报告生成完成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成数据分析报告时出错: {str(e)}")
            raise
    
    def generate_model_evaluation_report(self, model_results: Dict[str, Any],
                                       model_name: str = "模型",
                                       report_name: str = "模型评估报告") -> str:
        """
        生成模型评估报告
        
        Args:
            model_results: 模型评估结果
            model_name: 模型名称
            report_name: 报告名称
            
        Returns:
            报告文件路径
        """
        self.logger.info(f"开始生成模型评估报告: {report_name}")
        
        try:
            # 性能指标分析
            performance_metrics = self._analyze_model_performance(model_results)
            
            # 特征重要性分析
            feature_importance = self._analyze_feature_importance(model_results)
            
            # 预测结果分析
            prediction_analysis = self._analyze_predictions(model_results)
            
            # 生成图表
            charts = self._generate_model_evaluation_charts(model_results)
            
            # 模型诊断
            model_diagnostics = self._diagnose_model(model_results)
            
            # 生成报告
            report_data = {
                'report_name': report_name,
                'model_name': model_name,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance_metrics': performance_metrics,
                'feature_importance': feature_importance,
                'prediction_analysis': prediction_analysis,
                'model_diagnostics': model_diagnostics,
                'charts': charts
            }
            
            # 渲染HTML报告
            html_content = self.report_templates['model_evaluation'].render(**report_data)
            
            # 保存报告
            report_path = f"{self.output_dir}/{report_name}_模型评估报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"模型评估报告生成完成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成模型评估报告时出错: {str(e)}")
            raise
    
    def generate_risk_assessment_report(self, risk_data: Dict[str, Any],
                                      report_name: str = "风险评估报告") -> str:
        """
        生成风险评估报告
        
        Args:
            risk_data: 风险评估数据
            report_name: 报告名称
            
        Returns:
            报告文件路径
        """
        self.logger.info(f"开始生成风险评估报告: {report_name}")
        
        try:
            # 风险分布分析
            risk_distribution = self._analyze_risk_distribution(risk_data)
            
            # 风险因子分析
            risk_factors = self._analyze_risk_factors(risk_data)
            
            # 风险预警分析
            risk_warnings = self._analyze_risk_warnings(risk_data)
            
            # 生成图表
            charts = self._generate_risk_assessment_charts(risk_data)
            
            # 风险建议
            risk_recommendations = self._generate_risk_recommendations(risk_data)
            
            # 生成报告
            report_data = {
                'report_name': report_name,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'risk_distribution': risk_distribution,
                'risk_factors': risk_factors,
                'risk_warnings': risk_warnings,
                'risk_recommendations': risk_recommendations,
                'charts': charts
            }
            
            # 渲染HTML报告
            html_content = self.report_templates['risk_assessment'].render(**report_data)
            
            # 保存报告
            report_path = f"{self.output_dir}/{report_name}_风险评估报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"风险评估报告生成完成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成风险评估报告时出错: {str(e)}")
            raise
    
    def generate_business_summary_report(self, business_data: Dict[str, Any],
                                       report_name: str = "业务总结报告") -> str:
        """
        生成业务总结报告
        
        Args:
            business_data: 业务数据
            report_name: 报告名称
            
        Returns:
            报告文件路径
        """
        self.logger.info(f"开始生成业务总结报告: {report_name}")
        
        try:
            # 业务指标分析
            business_metrics = self._analyze_business_metrics(business_data)
            
            # 趋势分析
            trend_analysis = self._analyze_business_trends(business_data)
            
            # 关键洞察
            key_insights = self._generate_key_insights(business_data)
            
            # 生成图表
            charts = self._generate_business_summary_charts(business_data)
            
            # 行动建议
            action_recommendations = self._generate_action_recommendations(business_data)
            
            # 生成报告
            report_data = {
                'report_name': report_name,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'business_metrics': business_metrics,
                'trend_analysis': trend_analysis,
                'key_insights': key_insights,
                'action_recommendations': action_recommendations,
                'charts': charts
            }
            
            # 渲染HTML报告
            html_content = self.report_templates['business_summary'].render(**report_data)
            
            # 保存报告
            report_path = f"{self.output_dir}/{report_name}_业务总结报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"业务总结报告生成完成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成业务总结报告时出错: {str(e)}")
            raise
    
    def _analyze_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析基础统计信息"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        stats = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_summary': data[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            'categorical_summary': {col: data[col].value_counts().head(10).to_dict() 
                                  for col in categorical_cols[:5]}  # 只显示前5个分类变量
        }
        
        return stats
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析数据质量"""
        quality = {
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_percentage': data.duplicated().sum() / len(data) * 100,
            'data_types': data.dtypes.astype(str).to_dict(),
            'unique_values': data.nunique().to_dict()
        }
        
        # 识别潜在问题
        issues = []
        for col, missing_pct in quality['missing_percentage'].items():
            if missing_pct > 50:
                issues.append(f"列 '{col}' 缺失值过多 ({missing_pct:.1f}%)")
            elif missing_pct > 20:
                issues.append(f"列 '{col}' 缺失值较多 ({missing_pct:.1f}%)")
        
        if quality['duplicate_percentage'] > 5:
            issues.append(f"重复行较多 ({quality['duplicate_percentage']:.1f}%)")
        
        quality['issues'] = issues
        
        return quality
    
    def _analyze_features(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """分析特征"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        feature_analysis = {
            'numeric_features': {},
            'categorical_features': {},
            'target_analysis': {}
        }
        
        # 数值特征分析
        for col in numeric_cols:
            if col != target_column:
                feature_analysis['numeric_features'][col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'skewness': data[col].skew(),
                    'kurtosis': data[col].kurtosis(),
                    'outliers': len(data[col][np.abs(data[col] - data[col].mean()) > 3 * data[col].std()])
                }
        
        # 分类特征分析
        for col in categorical_cols:
            if col != target_column:
                feature_analysis['categorical_features'][col] = {
                    'unique_count': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'frequency_distribution': data[col].value_counts().head(10).to_dict()
                }
        
        # 目标变量分析
        if target_column and target_column in data.columns:
            if data[target_column].dtype in ['object', 'category']:
                feature_analysis['target_analysis'] = {
                    'type': 'categorical',
                    'unique_values': data[target_column].nunique(),
                    'distribution': data[target_column].value_counts().to_dict()
                }
            else:
                feature_analysis['target_analysis'] = {
                    'type': 'numeric',
                    'mean': data[target_column].mean(),
                    'std': data[target_column].std(),
                    'distribution': 'continuous'
                }
        
        return feature_analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析相关性"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'message': '数值特征不足，无法进行相关性分析'}
        
        correlation_matrix = numeric_data.corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        }
    
    def _generate_data_analysis_charts(self, data: pd.DataFrame, target_column: str = None) -> List[str]:
        """生成数据分析图表"""
        charts = []
        
        try:
            # 1. 缺失值热图
            if data.isnull().sum().sum() > 0:
                plt.figure(figsize=self.chart_config['figure_size'])
                sns.heatmap(data.isnull(), cbar=True, yticklabels=False, cmap='viridis')
                plt.title('缺失值分布热图')
                chart_path = f"{self.output_dir}/charts/missing_values_heatmap.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 2. 数值特征分布
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                n_cols = min(4, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, color=self.chart_config['color_palette'][i % len(self.chart_config['color_palette'])])
                        axes[i].set_title(f'{col} 分布')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('频次')
                
                # 隐藏多余的子图
                for i in range(len(numeric_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                chart_path = f"{self.output_dir}/charts/numeric_distributions.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 3. 相关性热图
            if len(numeric_cols) > 1:
                plt.figure(figsize=self.chart_config['figure_size'])
                correlation_matrix = data[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title('特征相关性热图')
                chart_path = f"{self.output_dir}/charts/correlation_heatmap.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 4. 目标变量分布（如果存在）
            if target_column and target_column in data.columns:
                plt.figure(figsize=(8, 6))
                if data[target_column].dtype in ['object', 'category']:
                    data[target_column].value_counts().plot(kind='bar', color=self.chart_config['color_palette'][0])
                    plt.title(f'{target_column} 分布')
                    plt.xlabel(target_column)
                    plt.ylabel('计数')
                    plt.xticks(rotation=45)
                else:
                    plt.hist(data[target_column].dropna(), bins=30, alpha=0.7, color=self.chart_config['color_palette'][0])
                    plt.title(f'{target_column} 分布')
                    plt.xlabel(target_column)
                    plt.ylabel('频次')
                
                chart_path = f"{self.output_dir}/charts/target_distribution.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
        
        except Exception as e:
            self.logger.error(f"生成数据分析图表时出错: {str(e)}")
        
        return charts
    
    def _analyze_model_performance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析模型性能"""
        # 这里需要根据实际的model_results结构来实现
        # 假设model_results包含常见的评估指标
        performance = {
            'accuracy': model_results.get('accuracy', 0),
            'precision': model_results.get('precision', 0),
            'recall': model_results.get('recall', 0),
            'f1_score': model_results.get('f1_score', 0),
            'auc_score': model_results.get('auc_score', 0),
            'confusion_matrix': model_results.get('confusion_matrix', []),
            'classification_report': model_results.get('classification_report', {})
        }
        
        # 性能评级
        if performance['accuracy'] >= 0.9:
            performance['grade'] = '优秀'
        elif performance['accuracy'] >= 0.8:
            performance['grade'] = '良好'
        elif performance['accuracy'] >= 0.7:
            performance['grade'] = '一般'
        else:
            performance['grade'] = '需要改进'
        
        return performance
    
    def _analyze_feature_importance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析特征重要性"""
        feature_importance = model_results.get('feature_importance', {})
        
        if feature_importance:
            # 排序特征重要性
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'top_features': sorted_features[:10],  # 前10个重要特征
                'total_features': len(feature_importance),
                'importance_distribution': feature_importance
            }
        
        return {'message': '无特征重要性信息'}
    
    def _analyze_predictions(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析预测结果"""
        predictions = model_results.get('predictions', [])
        actual = model_results.get('actual', [])
        
        if predictions and actual:
            predictions = np.array(predictions)
            actual = np.array(actual)
            
            analysis = {
                'total_predictions': len(predictions),
                'correct_predictions': np.sum(predictions == actual),
                'accuracy': np.mean(predictions == actual),
                'prediction_distribution': np.bincount(predictions).tolist(),
                'actual_distribution': np.bincount(actual).tolist()
            }
            
            return analysis
        
        return {'message': '无预测结果信息'}
    
    def _diagnose_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """模型诊断"""
        diagnostics = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        accuracy = model_results.get('accuracy', 0)
        precision = model_results.get('precision', 0)
        recall = model_results.get('recall', 0)
        
        # 优势分析
        if accuracy > 0.85:
            diagnostics['strengths'].append('模型整体准确率较高')
        if precision > 0.8:
            diagnostics['strengths'].append('模型精确率表现良好')
        if recall > 0.8:
            diagnostics['strengths'].append('模型召回率表现良好')
        
        # 弱点分析
        if accuracy < 0.7:
            diagnostics['weaknesses'].append('模型准确率偏低')
        if precision < 0.7:
            diagnostics['weaknesses'].append('模型精确率需要提升')
        if recall < 0.7:
            diagnostics['weaknesses'].append('模型召回率需要提升')
        
        # 建议
        if accuracy < 0.8:
            diagnostics['recommendations'].append('考虑增加训练数据或调整模型参数')
        if abs(precision - recall) > 0.1:
            diagnostics['recommendations'].append('考虑调整分类阈值以平衡精确率和召回率')
        
        return diagnostics
    
    def _generate_model_evaluation_charts(self, model_results: Dict[str, Any]) -> List[str]:
        """生成模型评估图表"""
        charts = []
        
        try:
            # 1. 混淆矩阵
            confusion_matrix = model_results.get('confusion_matrix')
            if confusion_matrix is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title('混淆矩阵')
                plt.xlabel('预测值')
                plt.ylabel('实际值')
                chart_path = f"{self.output_dir}/charts/confusion_matrix.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 2. 特征重要性
            feature_importance = model_results.get('feature_importance')
            if feature_importance:
                plt.figure(figsize=(10, 8))
                features = list(feature_importance.keys())[:15]  # 前15个特征
                importances = [feature_importance[f] for f in features]
                
                plt.barh(range(len(features)), importances, color=self.chart_config['color_palette'][0])
                plt.yticks(range(len(features)), features)
                plt.xlabel('重要性')
                plt.title('特征重要性排序')
                plt.gca().invert_yaxis()
                chart_path = f"{self.output_dir}/charts/feature_importance.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
        
        except Exception as e:
            self.logger.error(f"生成模型评估图表时出错: {str(e)}")
        
        return charts
    
    def _analyze_risk_distribution(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析风险分布"""
        # 实现风险分布分析逻辑
        return {
            'high_risk_count': risk_data.get('high_risk_count', 0),
            'medium_risk_count': risk_data.get('medium_risk_count', 0),
            'low_risk_count': risk_data.get('low_risk_count', 0),
            'total_count': risk_data.get('total_count', 0),
            'high_risk_ratio': risk_data.get('high_risk_ratio', 0)
        }
    
    def _analyze_risk_factors(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析风险因子"""
        return {
            'primary_factors': risk_data.get('primary_factors', []),
            'secondary_factors': risk_data.get('secondary_factors', []),
            'factor_weights': risk_data.get('factor_weights', {})
        }
    
    def _analyze_risk_warnings(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析风险预警"""
        return {
            'critical_warnings': risk_data.get('critical_warnings', []),
            'moderate_warnings': risk_data.get('moderate_warnings', []),
            'warning_trends': risk_data.get('warning_trends', [])
        }
    
    def _generate_risk_assessment_charts(self, risk_data: Dict[str, Any]) -> List[str]:
        """生成风险评估图表"""
        charts = []
        
        try:
            # 风险分布饼图
            risk_counts = [
                risk_data.get('high_risk_count', 0),
                risk_data.get('medium_risk_count', 0),
                risk_data.get('low_risk_count', 0)
            ]
            
            if sum(risk_counts) > 0:
                plt.figure(figsize=(8, 8))
                labels = ['高风险', '中风险', '低风险']
                colors = ['#ff4444', '#ffaa00', '#44ff44']
                plt.pie(risk_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('风险分布')
                chart_path = f"{self.output_dir}/charts/risk_distribution.png"
                plt.savefig(chart_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
        
        except Exception as e:
            self.logger.error(f"生成风险评估图表时出错: {str(e)}")
        
        return charts
    
    def _generate_risk_recommendations(self, risk_data: Dict[str, Any]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        high_risk_ratio = risk_data.get('high_risk_ratio', 0)
        
        if high_risk_ratio > 0.3:
            recommendations.append("高风险客户比例过高，建议加强风控措施")
        elif high_risk_ratio > 0.1:
            recommendations.append("高风险客户比例适中，建议持续监控")
        else:
            recommendations.append("风险控制良好，建议保持现有策略")
        
        return recommendations
    
    def _analyze_business_metrics(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业务指标"""
        return {
            'revenue': business_data.get('revenue', 0),
            'profit': business_data.get('profit', 0),
            'customer_count': business_data.get('customer_count', 0),
            'conversion_rate': business_data.get('conversion_rate', 0),
            'retention_rate': business_data.get('retention_rate', 0)
        }
    
    def _analyze_business_trends(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业务趋势"""
        return {
            'revenue_trend': business_data.get('revenue_trend', 'stable'),
            'customer_growth': business_data.get('customer_growth', 0),
            'market_share': business_data.get('market_share', 0)
        }
    
    def _generate_key_insights(self, business_data: Dict[str, Any]) -> List[str]:
        """生成关键洞察"""
        insights = [
            "业务整体表现稳定",
            "客户增长趋势良好",
            "盈利能力有待提升"
        ]
        return insights
    
    def _generate_business_summary_charts(self, business_data: Dict[str, Any]) -> List[str]:
        """生成业务总结图表"""
        charts = []
        # 实现业务图表生成逻辑
        return charts
    
    def _generate_action_recommendations(self, business_data: Dict[str, Any]) -> List[str]:
        """生成行动建议"""
        recommendations = [
            "优化客户获取渠道",
            "提升产品服务质量",
            "加强数据分析能力"
        ]
        return recommendations
    
    def _get_data_analysis_template(self) -> Template:
        """获取数据分析报告模板"""
        template_str = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report_name }}</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .info-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #3498db; color: white; border-radius: 5px; text-align: center; min-width: 120px; }
                .chart { text-align: center; margin: 20px 0; }
                .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #3498db; color: white; }
                .warning { background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .success { background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ report_name }}</h1>
                <div class="info-box">
                    <strong>生成时间:</strong> {{ generation_time }}<br>
                    <strong>数据规模:</strong> {{ data_shape[0] }} 行 × {{ data_shape[1] }} 列
                </div>
                
                <h2>基础统计信息</h2>
                <div class="metric">总行数<br><strong>{{ basic_stats.total_rows }}</strong></div>
                <div class="metric">总列数<br><strong>{{ basic_stats.total_columns }}</strong></div>
                <div class="metric">数值列<br><strong>{{ basic_stats.numeric_columns }}</strong></div>
                <div class="metric">分类列<br><strong>{{ basic_stats.categorical_columns }}</strong></div>
                <div class="metric">内存使用<br><strong>{{ "%.2f"|format(basic_stats.memory_usage) }} MB</strong></div>
                
                <h2>数据质量分析</h2>
                {% if quality_analysis.issues %}
                    <h3>发现的问题:</h3>
                    {% for issue in quality_analysis.issues %}
                        <div class="warning">{{ issue }}</div>
                    {% endfor %}
                {% else %}
                    <div class="success">数据质量良好，未发现明显问题</div>
                {% endif %}
                
                <h3>缺失值统计:</h3>
                <table>
                    <tr><th>列名</th><th>缺失数量</th><th>缺失比例</th></tr>
                    {% for col, missing_count in quality_analysis.missing_values.items() %}
                        {% if missing_count > 0 %}
                            <tr>
                                <td>{{ col }}</td>
                                <td>{{ missing_count }}</td>
                                <td>{{ "%.2f"|format(quality_analysis.missing_percentage[col]) }}%</td>
                            </tr>
                        {% endif %}
                    {% endfor %}
                </table>
                
                <h2>图表分析</h2>
                {% for chart in charts %}
                    <div class="chart">
                        <img src="{{ chart }}" alt="数据分析图表">
                    </div>
                {% endfor %}
                
                <h2>相关性分析</h2>
                {% if correlation_analysis.high_correlation_pairs %}
                    <h3>高相关性特征对 (|r| > 0.7):</h3>
                    <table>
                        <tr><th>特征1</th><th>特征2</th><th>相关系数</th></tr>
                        {% for pair in correlation_analysis.high_correlation_pairs %}
                            <tr>
                                <td>{{ pair.feature1 }}</td>
                                <td>{{ pair.feature2 }}</td>
                                <td>{{ "%.3f"|format(pair.correlation) }}</td>
                            </tr>
                        {% endfor %}
                    </table>
                {% else %}
                    <div class="info-box">未发现高相关性特征对</div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _get_model_evaluation_template(self) -> Template:
        """获取模型评估报告模板"""
        template_str = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report_name }}</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }
                h2 { color: #34495e; border-left: 4px solid #e74c3c; padding-left: 15px; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .info-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #e74c3c; color: white; border-radius: 5px; text-align: center; min-width: 120px; }
                .chart { text-align: center; margin: 20px 0; }
                .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                .grade { font-size: 24px; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .grade.excellent { background-color: #27ae60; color: white; }
                .grade.good { background-color: #f39c12; color: white; }
                .grade.average { background-color: #e67e22; color: white; }
                .grade.poor { background-color: #e74c3c; color: white; }
                ul { list-style-type: none; padding: 0; }
                li { background-color: #ecf0f1; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ report_name }}</h1>
                <div class="info-box">
                    <strong>模型名称:</strong> {{ model_name }}<br>
                    <strong>生成时间:</strong> {{ generation_time }}
                </div>
                
                <h2>性能指标</h2>
                <div class="metric">准确率<br><strong>{{ "%.3f"|format(performance_metrics.accuracy) }}</strong></div>
                <div class="metric">精确率<br><strong>{{ "%.3f"|format(performance_metrics.precision) }}</strong></div>
                <div class="metric">召回率<br><strong>{{ "%.3f"|format(performance_metrics.recall) }}</strong></div>
                <div class="metric">F1分数<br><strong>{{ "%.3f"|format(performance_metrics.f1_score) }}</strong></div>
                <div class="metric">AUC分数<br><strong>{{ "%.3f"|format(performance_metrics.auc_score) }}</strong></div>
                
                <div class="grade {{ performance_metrics.grade|lower }}">
                    模型评级: {{ performance_metrics.grade }}
                </div>
                
                <h2>模型诊断</h2>
                {% if model_diagnostics.strengths %}
                    <h3>模型优势:</h3>
                    <ul>
                        {% for strength in model_diagnostics.strengths %}
                            <li>{{ strength }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
                
                {% if model_diagnostics.weaknesses %}
                    <h3>需要改进:</h3>
                    <ul>
                        {% for weakness in model_diagnostics.weaknesses %}
                            <li>{{ weakness }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
                
                {% if model_diagnostics.recommendations %}
                    <h3>改进建议:</h3>
                    <ul>
                        {% for recommendation in model_diagnostics.recommendations %}
                            <li>{{ recommendation }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
                
                <h2>图表分析</h2>
                {% for chart in charts %}
                    <div class="chart">
                        <img src="{{ chart }}" alt="模型评估图表">
                    </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _get_risk_assessment_template(self) -> Template:
        """获取风险评估报告模板"""
        template_str = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report_name }}</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #e67e22; padding-bottom: 10px; }
                h2 { color: #34495e; border-left: 4px solid #e67e22; padding-left: 15px; margin-top: 30px; }
                .risk-high { background-color: #e74c3c; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .risk-medium { background-color: #f39c12; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .risk-low { background-color: #27ae60; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .chart { text-align: center; margin: 20px 0; }
                .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                ul { list-style-type: none; padding: 0; }
                li { background-color: #ecf0f1; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #e67e22; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ report_name }}</h1>
                <div class="info-box">
                    <strong>生成时间:</strong> {{ generation_time }}
                </div>
                
                <h2>风险分布</h2>
                <div class="risk-high">高风险: {{ risk_distribution.high_risk_count }} 个 ({{ "%.1f"|format(risk_distribution.high_risk_ratio * 100) }}%)</div>
                <div class="risk-medium">中风险: {{ risk_distribution.medium_risk_count }} 个</div>
                <div class="risk-low">低风险: {{ risk_distribution.low_risk_count }} 个</div>
                
                <h2>风险建议</h2>
                <ul>
                    {% for recommendation in risk_recommendations %}
                        <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
                
                <h2>图表分析</h2>
                {% for chart in charts %}
                    <div class="chart">
                        <img src="{{ chart }}" alt="风险评估图表">
                    </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _get_business_summary_template(self) -> Template:
        """获取业务总结报告模板"""
        template_str = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report_name }}</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }
                h2 { color: #34495e; border-left: 4px solid #9b59b6; padding-left: 15px; margin-top: 30px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #9b59b6; color: white; border-radius: 5px; text-align: center; min-width: 120px; }
                .insight { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #9b59b6; margin: 10px 0; }
                ul { list-style-type: none; padding: 0; }
                li { background-color: #ecf0f1; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #9b59b6; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ report_name }}</h1>
                <div class="info-box">
                    <strong>生成时间:</strong> {{ generation_time }}
                </div>
                
                <h2>业务指标</h2>
                <div class="metric">营收<br><strong>{{ business_metrics.revenue }}</strong></div>
                <div class="metric">利润<br><strong>{{ business_metrics.profit }}</strong></div>
                <div class="metric">客户数<br><strong>{{ business_metrics.customer_count }}</strong></div>
                
                <h2>关键洞察</h2>
                {% for insight in key_insights %}
                    <div class="insight">{{ insight }}</div>
                {% endfor %}
                
                <h2>行动建议</h2>
                <ul>
                    {% for recommendation in action_recommendations %}
                        <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
        return Template(template_str)

if __name__ == "__main__":
    # 创建报告生成器
    generator = IntelligentReportGenerator()
"""
模型解释性工具
集成SHAP、LIME等工具实现模型可解释性分析
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json

# 模型解释性库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP未安装，部分功能将不可用")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME未安装，部分功能将不可用")

# 机器学习库
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

class ModelInterpretabilityAnalyzer:
    """模型解释性分析器"""
    
    def __init__(self, output_dir: str = "interpretability_analysis"):
        """
        初始化模型解释性分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.charts_dir = os.path.join(output_dir, "charts")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.data_dir = os.path.join(output_dir, "data")
        
        for dir_path in [self.charts_dir, self.reports_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化解释器
        self.shap_explainers = {}
        self.lime_explainers = {}
        
        # 配置matplotlib中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def comprehensive_interpretability_analysis(self, 
                                              models: Dict[str, Any],
                                              X_train: np.ndarray,
                                              X_test: np.ndarray,
                                              y_train: np.ndarray,
                                              y_test: np.ndarray,
                                              feature_names: List[str] = None) -> Dict:
        """
        综合可解释性分析
        
        Args:
            models: 模型字典
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            y_test: 测试标签
            feature_names: 特征名称列表
            
        Returns:
            解释性分析结果
        """
        
        print("🔍 开始综合可解释性分析...")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        results = {
            'global_explanations': {},
            'local_explanations': {},
            'feature_importance': {},
            'interaction_analysis': {},
            'model_behavior': {},
            'charts': {},
            'summary': {}
        }
        
        for model_name, model in models.items():
            print(f"  📊 分析模型: {model_name}")
            
            try:
                # 全局解释性分析
                global_exp = self._analyze_global_interpretability(
                    model, model_name, X_train, X_test, y_train, feature_names
                )
                results['global_explanations'][model_name] = global_exp
                
                # 局部解释性分析
                local_exp = self._analyze_local_interpretability(
                    model, model_name, X_train, X_test, y_train, feature_names
                )
                results['local_explanations'][model_name] = local_exp
                
                # 特征重要性分析
                feature_imp = self._analyze_feature_importance(
                    model, model_name, X_train, X_test, y_train, y_test, feature_names
                )
                results['feature_importance'][model_name] = feature_imp
                
                # 特征交互分析
                interaction_analysis = self._analyze_feature_interactions(
                    model, model_name, X_train, feature_names
                )
                results['interaction_analysis'][model_name] = interaction_analysis
                
                # 模型行为分析
                behavior_analysis = self._analyze_model_behavior(
                    model, model_name, X_train, X_test, y_train, feature_names
                )
                results['model_behavior'][model_name] = behavior_analysis
                
            except Exception as e:
                print(f"    ❌ {model_name} 分析失败: {str(e)}")
                results['global_explanations'][model_name] = {'error': str(e)}
        
        # 生成对比分析
        results['comparison'] = self._compare_model_interpretability(results)
        
        # 生成图表
        results['charts'] = self._generate_interpretability_charts(results, feature_names)
        
        # 生成摘要
        results['summary'] = self._generate_interpretability_summary(results)
        
        print("✅ 综合可解释性分析完成")
        return results
    
    def _analyze_global_interpretability(self, 
                                       model: Any,
                                       model_name: str,
                                       X_train: np.ndarray,
                                       X_test: np.ndarray,
                                       y_train: np.ndarray,
                                       feature_names: List[str]) -> Dict:
        """分析全局可解释性"""
        
        global_analysis = {
            'shap_analysis': None,
            'permutation_importance': None,
            'model_specific_importance': None
        }
        
        try:
            # SHAP全局分析
            if SHAP_AVAILABLE:
                global_analysis['shap_analysis'] = self._shap_global_analysis(
                    model, model_name, X_train, X_test, feature_names
                )
            
            # 排列重要性
            global_analysis['permutation_importance'] = self._permutation_importance_analysis(
                model, X_test, y_train, feature_names
            )
            
            # 模型特定重要性
            global_analysis['model_specific_importance'] = self._model_specific_importance(
                model, feature_names
            )
            
        except Exception as e:
            global_analysis['error'] = str(e)
        
        return global_analysis
    
    def _shap_global_analysis(self, 
                            model: Any,
                            model_name: str,
                            X_train: np.ndarray,
                            X_test: np.ndarray,
                            feature_names: List[str]) -> Dict:
        """SHAP全局分析"""
        
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            # 选择合适的解释器
            if hasattr(model, 'predict_proba'):
                # 对于概率预测模型
                if 'tree' in model_name.lower() or 'forest' in model_name.lower():
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train[:100])  # 使用样本作为背景
            else:
                explainer = shap.Explainer(model, X_train[:100])
            
            # 计算SHAP值
            shap_values = explainer(X_test[:200])  # 限制样本数量以提高速度
            
            # 全局特征重要性
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:  # 多类分类
                    global_importance = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)
                else:
                    global_importance = np.mean(np.abs(shap_values.values), axis=0)
            else:
                global_importance = np.mean(np.abs(shap_values), axis=0)
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': global_importance
            }).sort_values('importance', ascending=False)
            
            # 保存SHAP图表
            self._save_shap_plots(shap_values, feature_names, model_name)
            
            return {
                'explainer_type': type(explainer).__name__,
                'feature_importance': importance_df.to_dict('records'),
                'shap_values_shape': shap_values.values.shape if hasattr(shap_values, 'values') else np.array(shap_values).shape,
                'top_features': importance_df.head(10).to_dict('records')
            }
            
        except Exception as e:
            return {'error': f'SHAP analysis failed: {str(e)}'}
    
    def _save_shap_plots(self, shap_values, feature_names: List[str], model_name: str):
        """保存SHAP图表"""
        
        try:
            # 摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP特征重要性摘要')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, f'{model_name}_shap_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f'{model_name} - SHAP特征重要性条形图')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, f'{model_name}_shap_bar.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️  SHAP图表保存失败: {str(e)}")
    
    def _permutation_importance_analysis(self, 
                                       model: Any,
                                       X_test: np.ndarray,
                                       y_test: np.ndarray,
                                       feature_names: List[str]) -> Dict:
        """排列重要性分析"""
        
        try:
            # 计算排列重要性
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            # 创建重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                'feature_importance': importance_df.to_dict('records'),
                'top_features': importance_df.head(10).to_dict('records'),
                'statistics': {
                    'mean_importance': float(np.mean(perm_importance.importances_mean)),
                    'std_importance': float(np.std(perm_importance.importances_mean)),
                    'max_importance': float(np.max(perm_importance.importances_mean)),
                    'min_importance': float(np.min(perm_importance.importances_mean))
                }
            }
            
        except Exception as e:
            return {'error': f'Permutation importance analysis failed: {str(e)}'}
    
    def _model_specific_importance(self, model: Any, feature_names: List[str]) -> Dict:
        """模型特定重要性分析"""
        
        try:
            importance_data = {}
            
            # 线性模型系数
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # 取第一类的系数
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coef,
                    'abs_coefficient': np.abs(coef)
                }).sort_values('abs_coefficient', ascending=False)
                
                importance_data['linear_coefficients'] = importance_df.to_dict('records')
            
            # 树模型特征重要性
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_data['tree_importance'] = importance_df.to_dict('records')
            
            # 集成模型特征重要性
            if hasattr(model, 'estimators_'):
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    # 计算所有基学习器的平均重要性
                    importances = np.array([est.feature_importances_ for est in model.estimators_])
                    mean_importance = np.mean(importances, axis=0)
                    std_importance = np.std(importances, axis=0)
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': mean_importance,
                        'importance_std': std_importance
                    }).sort_values('importance_mean', ascending=False)
                    
                    importance_data['ensemble_importance'] = importance_df.to_dict('records')
            
            return importance_data
            
        except Exception as e:
            return {'error': f'Model-specific importance analysis failed: {str(e)}'}
    
    def _analyze_local_interpretability(self, 
                                      model: Any,
                                      model_name: str,
                                      X_train: np.ndarray,
                                      X_test: np.ndarray,
                                      y_train: np.ndarray,
                                      feature_names: List[str]) -> Dict:
        """分析局部可解释性"""
        
        local_analysis = {
            'shap_local': None,
            'lime_local': None,
            'sample_explanations': []
        }
        
        try:
            # 选择代表性样本进行局部解释
            sample_indices = self._select_representative_samples(X_test, y_train, n_samples=5)
            
            # SHAP局部分析
            if SHAP_AVAILABLE:
                local_analysis['shap_local'] = self._shap_local_analysis(
                    model, model_name, X_train, X_test, sample_indices, feature_names
                )
            
            # LIME局部分析
            if LIME_AVAILABLE:
                local_analysis['lime_local'] = self._lime_local_analysis(
                    model, model_name, X_train, X_test, sample_indices, feature_names
                )
            
            # 样本解释
            for idx in sample_indices:
                sample_explanation = self._explain_single_sample(
                    model, X_test[idx:idx+1], feature_names, idx
                )
                local_analysis['sample_explanations'].append(sample_explanation)
            
        except Exception as e:
            local_analysis['error'] = str(e)
        
        return local_analysis
    
    def _select_representative_samples(self, X_test: np.ndarray, y_train: np.ndarray, n_samples: int = 5) -> List[int]:
        """选择代表性样本"""
        
        try:
            # 简单策略：选择不同区域的样本
            indices = []
            
            # 随机选择
            np.random.seed(42)
            random_indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
            indices.extend(random_indices)
            
            return list(set(indices))[:n_samples]
            
        except Exception as e:
            # 如果失败，返回前n个样本
            return list(range(min(n_samples, len(X_test))))
    
    def _shap_local_analysis(self, 
                           model: Any,
                           model_name: str,
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           sample_indices: List[int],
                           feature_names: List[str]) -> Dict:
        """SHAP局部分析"""
        
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            # 创建解释器
            if 'tree' in model_name.lower() or 'forest' in model_name.lower():
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_train[:100])
            
            local_explanations = []
            
            for idx in sample_indices:
                sample = X_test[idx:idx+1]
                shap_values = explainer(sample)
                
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:  # 多类分类
                        values = shap_values.values[0, :, 1]  # 取正类的SHAP值
                    else:
                        values = shap_values.values[0]
                else:
                    values = shap_values[0]
                
                # 创建解释DataFrame
                explanation_df = pd.DataFrame({
                    'feature': feature_names,
                    'feature_value': sample[0],
                    'shap_value': values
                }).sort_values('shap_value', key=abs, ascending=False)
                
                local_explanations.append({
                    'sample_index': idx,
                    'explanation': explanation_df.to_dict('records'),
                    'prediction': float(model.predict_proba(sample)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(sample)[0])
                })
                
                # 保存单个样本的SHAP图
                self._save_single_shap_plot(shap_values, feature_names, model_name, idx)
            
            return {
                'explanations': local_explanations,
                'summary': f'Generated local explanations for {len(sample_indices)} samples'
            }
            
        except Exception as e:
            return {'error': f'SHAP local analysis failed: {str(e)}'}
    
    def _save_single_shap_plot(self, shap_values, feature_names: List[str], model_name: str, sample_idx: int):
        """保存单个样本的SHAP图"""
        
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[0], show=False)
            plt.title(f'{model_name} - 样本{sample_idx} SHAP解释')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, f'{model_name}_sample_{sample_idx}_shap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️  样本{sample_idx} SHAP图保存失败: {str(e)}")
    
    def _lime_local_analysis(self, 
                           model: Any,
                           model_name: str,
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           sample_indices: List[int],
                           feature_names: List[str]) -> Dict:
        """LIME局部分析"""
        
        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}
        
        try:
            # 创建LIME解释器
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=['低风险', '高风险'],
                mode='classification'
            )
            
            local_explanations = []
            
            for idx in sample_indices:
                sample = X_test[idx]
                
                # 生成解释
                explanation = explainer.explain_instance(
                    sample, 
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(feature_names)
                )
                
                # 提取解释数据
                explanation_data = []
                for feature, weight in explanation.as_list():
                    explanation_data.append({
                        'feature': feature,
                        'weight': weight,
                        'abs_weight': abs(weight)
                    })
                
                # 按重要性排序
                explanation_data.sort(key=lambda x: x['abs_weight'], reverse=True)
                
                local_explanations.append({
                    'sample_index': idx,
                    'explanation': explanation_data,
                    'prediction_proba': explanation.predict_proba.tolist() if hasattr(explanation, 'predict_proba') else None
                })
                
                # 保存LIME图
                self._save_lime_plot(explanation, model_name, idx)
            
            return {
                'explanations': local_explanations,
                'summary': f'Generated LIME explanations for {len(sample_indices)} samples'
            }
            
        except Exception as e:
            return {'error': f'LIME local analysis failed: {str(e)}'}
    
    def _save_lime_plot(self, explanation, model_name: str, sample_idx: int):
        """保存LIME图"""
        
        try:
            # 保存为HTML
            html_path = os.path.join(self.charts_dir, f'{model_name}_sample_{sample_idx}_lime.html')
            explanation.save_to_file(html_path)
            
        except Exception as e:
            print(f"    ⚠️  样本{sample_idx} LIME图保存失败: {str(e)}")
    
    def _explain_single_sample(self, 
                             model: Any,
                             sample: np.ndarray,
                             feature_names: List[str],
                             sample_idx: int) -> Dict:
        """解释单个样本"""
        
        try:
            # 基本预测信息
            prediction = model.predict(sample)[0]
            prediction_proba = model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None
            
            # 特征值分析
            feature_analysis = []
            for i, (feature_name, value) in enumerate(zip(feature_names, sample[0])):
                feature_analysis.append({
                    'feature': feature_name,
                    'value': float(value),
                    'normalized_value': float((value - np.mean(sample)) / (np.std(sample) + 1e-8))
                })
            
            return {
                'sample_index': sample_idx,
                'prediction': int(prediction),
                'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
                'feature_values': feature_analysis,
                'risk_level': '高风险' if prediction == 1 else '低风险'
            }
            
        except Exception as e:
            return {'error': f'Sample explanation failed: {str(e)}'}
    
    def _analyze_feature_importance(self, 
                                  model: Any,
                                  model_name: str,
                                  X_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_train: np.ndarray,
                                  y_test: np.ndarray,
                                  feature_names: List[str]) -> Dict:
        """特征重要性综合分析"""
        
        importance_analysis = {
            'methods_comparison': {},
            'stability_analysis': {},
            'correlation_analysis': {}
        }
        
        try:
            # 多种方法的特征重要性
            methods = {}
            
            # 1. 排列重要性
            perm_imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            methods['permutation'] = perm_imp.importances_mean
            
            # 2. 模型内置重要性
            if hasattr(model, 'feature_importances_'):
                methods['builtin'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                methods['coefficients'] = np.abs(coef)
            
            # 3. SHAP重要性（如果可用）
            if SHAP_AVAILABLE:
                try:
                    if 'tree' in model_name.lower() or 'forest' in model_name.lower():
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model, X_train[:50])
                    
                    shap_values = explainer(X_test[:100])
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.values.shape) == 3:
                            shap_importance = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)
                        else:
                            shap_importance = np.mean(np.abs(shap_values.values), axis=0)
                    else:
                        shap_importance = np.mean(np.abs(shap_values), axis=0)
                    
                    methods['shap'] = shap_importance
                except:
                    pass
            
            # 创建对比DataFrame
            comparison_data = {'feature': feature_names}
            for method_name, importance in methods.items():
                comparison_data[method_name] = importance
            
            comparison_df = pd.DataFrame(comparison_data)
            importance_analysis['methods_comparison'] = comparison_df.to_dict('records')
            
            # 稳定性分析
            if len(methods) > 1:
                # 计算不同方法之间的相关性
                method_names = list(methods.keys())
                correlations = {}
                
                for i, method1 in enumerate(method_names):
                    for j, method2 in enumerate(method_names[i+1:], i+1):
                        corr = np.corrcoef(methods[method1], methods[method2])[0, 1]
                        correlations[f'{method1}_vs_{method2}'] = float(corr)
                
                importance_analysis['stability_analysis'] = {
                    'method_correlations': correlations,
                    'average_correlation': float(np.mean(list(correlations.values()))),
                    'stability_score': float(np.mean(list(correlations.values())))
                }
            
            # 特征相关性分析
            feature_corr = np.corrcoef(X_train.T)
            high_corr_pairs = []
            
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr = feature_corr[i, j]
                    if abs(corr) > 0.7:  # 高相关性阈值
                        high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr)
                        })
            
            importance_analysis['correlation_analysis'] = {
                'high_correlation_pairs': high_corr_pairs,
                'feature_correlation_matrix': feature_corr.tolist()
            }
            
        except Exception as e:
            importance_analysis['error'] = str(e)
        
        return importance_analysis
    
    def _analyze_feature_interactions(self, 
                                    model: Any,
                                    model_name: str,
                                    X_train: np.ndarray,
                                    feature_names: List[str]) -> Dict:
        """特征交互分析"""
        
        interaction_analysis = {
            'shap_interactions': None,
            'statistical_interactions': None,
            'pairwise_analysis': None
        }
        
        try:
            # SHAP交互分析
            if SHAP_AVAILABLE and len(feature_names) <= 20:  # 限制特征数量以提高速度
                try:
                    if 'tree' in model_name.lower() or 'forest' in model_name.lower():
                        explainer = shap.TreeExplainer(model)
                        shap_interaction_values = explainer.shap_interaction_values(X_train[:100])
                        
                        # 计算交互强度
                        interaction_strength = np.abs(shap_interaction_values).mean(axis=0)
                        
                        # 找到最强的交互
                        top_interactions = []
                        for i in range(len(feature_names)):
                            for j in range(i+1, len(feature_names)):
                                strength = interaction_strength[i, j]
                                top_interactions.append({
                                    'feature1': feature_names[i],
                                    'feature2': feature_names[j],
                                    'interaction_strength': float(strength)
                                })
                        
                        top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                        
                        interaction_analysis['shap_interactions'] = {
                            'top_interactions': top_interactions[:10],
                            'interaction_matrix_shape': interaction_strength.shape
                        }
                except Exception as e:
                    interaction_analysis['shap_interactions'] = {'error': str(e)}
            
            # 统计交互分析
            interaction_analysis['statistical_interactions'] = self._statistical_interaction_analysis(
                X_train, feature_names
            )
            
        except Exception as e:
            interaction_analysis['error'] = str(e)
        
        return interaction_analysis
    
    def _statistical_interaction_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """统计交互分析"""
        
        try:
            # 计算特征间的统计关系
            correlations = np.corrcoef(X.T)
            
            # 找到强相关的特征对
            strong_correlations = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr = correlations[i, j]
                    if abs(corr) > 0.3:  # 相关性阈值
                        strong_correlations.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr),
                            'correlation_strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                        })
            
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'strong_correlations': strong_correlations[:20],
                'correlation_statistics': {
                    'mean_abs_correlation': float(np.mean(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))),
                    'max_correlation': float(np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))),
                    'num_strong_correlations': len([c for c in strong_correlations if abs(c['correlation']) > 0.7])
                }
            }
            
        except Exception as e:
            return {'error': f'Statistical interaction analysis failed: {str(e)}'}
    
    def _analyze_model_behavior(self, 
                              model: Any,
                              model_name: str,
                              X_train: np.ndarray,
                              X_test: np.ndarray,
                              y_train: np.ndarray,
                              feature_names: List[str]) -> Dict:
        """模型行为分析"""
        
        behavior_analysis = {
            'decision_boundary': None,
            'prediction_distribution': None,
            'confidence_analysis': None,
            'robustness_analysis': None
        }
        
        try:
            # 预测分布分析
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                
                behavior_analysis['prediction_distribution'] = {
                    'class_distribution': {
                        'class_0': float(np.sum(y_proba[:, 0] > 0.5)),
                        'class_1': float(np.sum(y_proba[:, 1] > 0.5))
                    },
                    'confidence_stats': {
                        'mean_confidence': float(np.mean(np.max(y_proba, axis=1))),
                        'std_confidence': float(np.std(np.max(y_proba, axis=1))),
                        'low_confidence_samples': float(np.sum(np.max(y_proba, axis=1) < 0.6))
                    }
                }
            
            # 鲁棒性分析
            behavior_analysis['robustness_analysis'] = self._robustness_analysis(
                model, X_test, feature_names
            )
            
        except Exception as e:
            behavior_analysis['error'] = str(e)
        
        return behavior_analysis
    
    def _robustness_analysis(self, model: Any, X_test: np.ndarray, feature_names: List[str]) -> Dict:
        """鲁棒性分析"""
        
        try:
            # 添加噪声测试
            noise_levels = [0.01, 0.05, 0.1]
            robustness_results = []
            
            original_predictions = model.predict(X_test)
            
            for noise_level in noise_levels:
                # 添加高斯噪声
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise
                
                noisy_predictions = model.predict(X_noisy)
                
                # 计算预测一致性
                consistency = np.mean(original_predictions == noisy_predictions)
                
                robustness_results.append({
                    'noise_level': noise_level,
                    'prediction_consistency': float(consistency),
                    'changed_predictions': int(np.sum(original_predictions != noisy_predictions))
                })
            
            return {
                'noise_robustness': robustness_results,
                'overall_robustness_score': float(np.mean([r['prediction_consistency'] for r in robustness_results]))
            }
            
        except Exception as e:
            return {'error': f'Robustness analysis failed: {str(e)}'}
    
    def _compare_model_interpretability(self, results: Dict) -> Dict:
        """对比模型可解释性"""
        
        comparison = {
            'interpretability_scores': {},
            'method_availability': {},
            'complexity_analysis': {}
        }
        
        try:
            for model_name in results['global_explanations'].keys():
                if 'error' not in results['global_explanations'][model_name]:
                    # 计算可解释性评分
                    score = 0
                    available_methods = 0
                    
                    # SHAP可用性
                    if (results['global_explanations'][model_name].get('shap_analysis') and 
                        'error' not in results['global_explanations'][model_name]['shap_analysis']):
                        score += 30
                        available_methods += 1
                    
                    # 排列重要性可用性
                    if (results['global_explanations'][model_name].get('permutation_importance') and
                        'error' not in results['global_explanations'][model_name]['permutation_importance']):
                        score += 20
                        available_methods += 1
                    
                    # 模型特定重要性
                    if (results['global_explanations'][model_name].get('model_specific_importance') and
                        'error' not in results['global_explanations'][model_name]['model_specific_importance']):
                        score += 25
                        available_methods += 1
                    
                    # 局部解释可用性
                    if (results['local_explanations'][model_name] and
                        'error' not in results['local_explanations'][model_name]):
                        score += 25
                        available_methods += 1
                    
                    comparison['interpretability_scores'][model_name] = {
                        'total_score': score,
                        'available_methods': available_methods,
                        'interpretability_level': self._get_interpretability_level(score)
                    }
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _get_interpretability_level(self, score: int) -> str:
        """获取可解释性等级"""
        
        if score >= 80:
            return '高'
        elif score >= 60:
            return '中'
        elif score >= 40:
            return '低'
        else:
            return '很低'
    
    def _generate_interpretability_charts(self, results: Dict, feature_names: List[str]) -> Dict:
        """生成可解释性图表"""
        
        charts = {}
        
        try:
            # 特征重要性对比图
            charts['feature_importance_comparison'] = self._create_feature_importance_comparison_chart(
                results, feature_names
            )
            
            # 可解释性评分雷达图
            charts['interpretability_radar'] = self._create_interpretability_radar_chart(results)
            
            # 特征相关性热力图
            charts['feature_correlation_heatmap'] = self._create_feature_correlation_heatmap(results)
            
        except Exception as e:
            charts['error'] = str(e)
        
        return charts
    
    def _create_feature_importance_comparison_chart(self, results: Dict, feature_names: List[str]) -> str:
        """创建特征重要性对比图表"""
        
        try:
            fig = make_subplots(
                rows=len(results['feature_importance']), cols=1,
                subplot_titles=[f'{model_name} 特征重要性' for model_name in results['feature_importance'].keys()],
                vertical_spacing=0.1
            )
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (model_name, importance_data) in enumerate(results['feature_importance'].items()):
                if 'error' not in importance_data and 'methods_comparison' in importance_data:
                    # 使用第一个可用的重要性方法
                    comparison_data = importance_data['methods_comparison']
                    if comparison_data:
                        df = pd.DataFrame(comparison_data)
                        
                        # 选择第一个数值列作为重要性
                        importance_col = None
                        for col in df.columns:
                            if col != 'feature' and df[col].dtype in ['float64', 'int64']:
                                importance_col = col
                                break
                        
                        if importance_col:
                            # 取前10个最重要的特征
                            top_features = df.nlargest(10, importance_col)
                            
                            fig.add_trace(
                                go.Bar(
                                    x=top_features['feature'],
                                    y=top_features[importance_col],
                                    name=f'{model_name}',
                                    marker_color=colors[i % len(colors)]
                                ),
                                row=i+1, col=1
                            )
            
            fig.update_layout(
                title='模型特征重要性对比',
                height=300 * len(results['feature_importance']),
                showlegend=False
            )
            
            chart_path = os.path.join(self.charts_dir, 'feature_importance_comparison.html')
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"特征重要性对比图创建失败: {str(e)}")
            return None
    
    def _create_interpretability_radar_chart(self, results: Dict) -> str:
        """创建可解释性雷达图"""
        
        try:
            if 'comparison' not in results or 'interpretability_scores' not in results['comparison']:
                return None
            
            scores = results['comparison']['interpretability_scores']
            
            fig = go.Figure()
            
            categories = ['SHAP可用性', '排列重要性', '模型特定重要性', '局部解释', '整体评分']
            
            for model_name, score_data in scores.items():
                # 构建雷达图数据
                values = [
                    score_data.get('total_score', 0) / 100 * 30,  # SHAP
                    score_data.get('total_score', 0) / 100 * 20,  # 排列重要性
                    score_data.get('total_score', 0) / 100 * 25,  # 模型特定
                    score_data.get('total_score', 0) / 100 * 25,  # 局部解释
                    score_data.get('total_score', 0) / 100        # 整体评分
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
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
                title="模型可解释性评估雷达图"
            )
            
            chart_path = os.path.join(self.charts_dir, 'interpretability_radar.html')
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"可解释性雷达图创建失败: {str(e)}")
            return None
    
    def _create_feature_correlation_heatmap(self, results: Dict) -> str:
        """创建特征相关性热力图"""
        
        try:
            # 从第一个模型的相关性分析中获取数据
            for model_name, importance_data in results['feature_importance'].items():
                if ('correlation_analysis' in importance_data and 
                    'feature_correlation_matrix' in importance_data['correlation_analysis']):
                    
                    corr_matrix = np.array(importance_data['correlation_analysis']['feature_correlation_matrix'])
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    
                    fig.update_layout(
                        title='特征相关性热力图',
                        xaxis_title='特征',
                        yaxis_title='特征'
                    )
                    
                    chart_path = os.path.join(self.charts_dir, 'feature_correlation_heatmap.html')
                    fig.write_html(chart_path)
                    
                    return chart_path
            
            return None
            
        except Exception as e:
            print(f"特征相关性热力图创建失败: {str(e)}")
            return None
    
    def _generate_interpretability_summary(self, results: Dict) -> Dict:
        """生成可解释性摘要"""
        
        summary = {
            'total_models_analyzed': len(results['global_explanations']),
            'successful_analyses': 0,
            'available_methods': {
                'shap': SHAP_AVAILABLE,
                'lime': LIME_AVAILABLE,
                'permutation_importance': True,
                'model_specific': True
            },
            'key_findings': [],
            'recommendations': []
        }
        
        try:
            # 统计成功分析的模型数量
            for model_name, analysis in results['global_explanations'].items():
                if 'error' not in analysis:
                    summary['successful_analyses'] += 1
            
            # 提取关键发现
            if 'comparison' in results and 'interpretability_scores' in results['comparison']:
                scores = results['comparison']['interpretability_scores']
                
                # 找到最可解释的模型
                best_model = max(scores.keys(), key=lambda k: scores[k]['total_score'])
                summary['key_findings'].append(f"最可解释的模型: {best_model}")
                
                # 平均可解释性评分
                avg_score = np.mean([s['total_score'] for s in scores.values()])
                summary['key_findings'].append(f"平均可解释性评分: {avg_score:.1f}")
            
            # 生成建议
            if SHAP_AVAILABLE:
                summary['recommendations'].append("建议使用SHAP进行深入的特征重要性分析")
            else:
                summary['recommendations'].append("建议安装SHAP库以获得更好的解释性分析")
            
            if LIME_AVAILABLE:
                summary['recommendations'].append("可以使用LIME进行局部解释分析")
            else:
                summary['recommendations'].append("建议安装LIME库以进行局部解释分析")
            
            summary['recommendations'].append("建议结合多种解释方法以获得更全面的理解")
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def generate_interpretability_report(self, results: Dict, output_path: str = None) -> str:
        """生成可解释性分析报告"""
        
        if output_path is None:
            output_path = os.path.join(self.reports_dir, f"interpretability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        try:
            # 生成HTML报告
            html_content = self._create_interpretability_html_report(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ 可解释性分析报告已生成: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 报告生成失败: {str(e)}")
            return None
    
    def _create_interpretability_html_report(self, results: Dict) -> str:
        """创建可解释性HTML报告"""
        
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>模型可解释性分析报告</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1, h2, h3 { color: #2c3e50; }
                .summary-box { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .model-section { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; margin-top: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .chart-container { margin: 30px 0; text-align: center; }
                .recommendation { background: #d5f4e6; border-left: 4px solid #27ae60; padding: 15px; margin: 10px 0; }
                .warning { background: #fdeaa7; border-left: 4px solid #f39c12; padding: 15px; margin: 10px 0; }
                .error { background: #fadbd8; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 模型可解释性分析报告</h1>
                
                <div class="summary-box">
                    <h2>📊 分析摘要</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{total_models}</div>
                            <div class="metric-label">分析模型数量</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{successful_analyses}</div>
                            <div class="metric-label">成功分析数量</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{shap_available}</div>
                            <div class="metric-label">SHAP可用</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{lime_available}</div>
                            <div class="metric-label">LIME可用</div>
                        </div>
                    </div>
                </div>
                
                {model_sections}
                
                <div class="summary-box">
                    <h2>🎯 关键发现</h2>
                    {key_findings}
                </div>
                
                <div class="summary-box">
                    <h2>💡 建议</h2>
                    {recommendations}
                </div>
                
                <div class="summary-box">
                    <h2>📈 图表分析</h2>
                    <p>详细的可视化图表已保存到charts目录中，包括：</p>
                    <ul>
                        <li>特征重要性对比图</li>
                        <li>可解释性评估雷达图</li>
                        <li>特征相关性热力图</li>
                        <li>SHAP分析图表</li>
                        <li>LIME解释图表</li>
                    </ul>
                </div>
                
                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
                    <p>报告生成时间: {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 填充模板数据
        summary = results.get('summary', {})
        
        # 生成模型分析部分
        model_sections = ""
        for model_name, global_analysis in results.get('global_explanations', {}).items():
            if 'error' not in global_analysis:
                model_sections += f"""
                <div class="model-section">
                    <h3>🤖 {model_name} 分析结果</h3>
                    
                    <h4>全局解释性</h4>
                    {self._format_global_analysis(global_analysis)}
                    
                    <h4>局部解释性</h4>
                    {self._format_local_analysis(results.get('local_explanations', {}).get(model_name, {}))}
                    
                    <h4>特征重要性</h4>
                    {self._format_feature_importance(results.get('feature_importance', {}).get(model_name, {}))}
                </div>
                """
        
        # 生成关键发现
        key_findings_html = ""
        for finding in summary.get('key_findings', []):
            key_findings_html += f"<p>• {finding}</p>"
        
        # 生成建议
        recommendations_html = ""
        for rec in summary.get('recommendations', []):
            recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        # 填充模板
        html_content = html_template.format(
            total_models=summary.get('total_models_analyzed', 0),
            successful_analyses=summary.get('successful_analyses', 0),
            shap_available="✅" if summary.get('available_methods', {}).get('shap', False) else "❌",
            lime_available="✅" if summary.get('available_methods', {}).get('lime', False) else "❌",
            model_sections=model_sections,
            key_findings=key_findings_html,
            recommendations=recommendations_html,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def _format_global_analysis(self, analysis: Dict) -> str:
        """格式化全局分析结果"""
        
        html = ""
        
        if 'shap_analysis' in analysis and analysis['shap_analysis'] and 'error' not in analysis['shap_analysis']:
            shap_data = analysis['shap_analysis']
            html += f"""
            <p><strong>SHAP分析:</strong> 使用{shap_data.get('explainer_type', 'Unknown')}解释器</p>
            <p>前5个重要特征:</p>
            <ul>
            """
            for feature in shap_data.get('top_features', [])[:5]:
                html += f"<li>{feature.get('feature', 'Unknown')}: {feature.get('importance', 0):.4f}</li>"
            html += "</ul>"
        
        if 'permutation_importance' in analysis and analysis['permutation_importance'] and 'error' not in analysis['permutation_importance']:
            perm_data = analysis['permutation_importance']
            html += f"""
            <p><strong>排列重要性:</strong></p>
            <p>平均重要性: {perm_data.get('statistics', {}).get('mean_importance', 0):.4f}</p>
            """
        
        return html if html else "<p>无可用的全局解释性分析结果</p>"
    
    def _format_local_analysis(self, analysis: Dict) -> str:
        """格式化局部分析结果"""
        
        if 'error' in analysis or not analysis:
            return "<p>无可用的局部解释性分析结果</p>"
        
        html = ""
        
        if 'shap_local' in analysis and analysis['shap_local'] and 'error' not in analysis['shap_local']:
            shap_local = analysis['shap_local']
            html += f"<p><strong>SHAP局部分析:</strong> {shap_local.get('summary', '')}</p>"
        
        if 'lime_local' in analysis and analysis['lime_local'] and 'error' not in analysis['lime_local']:
            lime_local = analysis['lime_local']
            html += f"<p><strong>LIME局部分析:</strong> {lime_local.get('summary', '')}</p>"
        
        return html if html else "<p>无可用的局部解释性分析结果</p>"
    
    def _format_feature_importance(self, analysis: Dict) -> str:
        """格式化特征重要性分析结果"""
        
        if 'error' in analysis or not analysis:
            return "<p>无可用的特征重要性分析结果</p>"
        
        html = ""
        
        if 'methods_comparison' in analysis:
            html += "<p><strong>多方法特征重要性对比:</strong> 已完成</p>"
        
        if 'stability_analysis' in analysis and 'stability_score' in analysis['stability_analysis']:
            stability_score = analysis['stability_analysis']['stability_score']
            html += f"<p><strong>稳定性评分:</strong> {stability_score:.3f}</p>"
        
        return html if html else "<p>无可用的特征重要性分析结果</p>"


# 使用示例
if __name__ == "__main__":
    # 创建可解释性分析器
    analyzer = ModelInterpretabilityAnalyzer("interpretability_output")
    
    print("🔍 模型可解释性分析器已初始化")
    print(f"📁 输出目录: {analyzer.output_dir}")
    print(f"🛠️  SHAP可用: {SHAP_AVAILABLE}")
    print(f"🛠️  LIME可用: {LIME_AVAILABLE}")
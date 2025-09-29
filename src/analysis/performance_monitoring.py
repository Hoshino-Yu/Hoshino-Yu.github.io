"""
性能监控和稳定性测试系统
实现模型训练和预测的性能监控，以及不同数据分布下的稳定性评估
"""

import os
import sys
import time
import psutil
import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
from contextlib import contextmanager

# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.base import clone

# 统计测试
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, output_dir: str = "performance_monitoring"):
        """
        初始化性能监控器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.logs_dir = os.path.join(output_dir, "logs")
        self.charts_dir = os.path.join(output_dir, "charts")
        self.reports_dir = os.path.join(output_dir, "reports")
        
        for dir_path in [self.logs_dir, self.charts_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 监控数据存储
        self.performance_logs = []
        self.stability_logs = []
        self.resource_logs = []
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 配置matplotlib中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    @contextmanager
    def monitor_performance(self, operation_name: str = "operation"):
        """
        性能监控上下文管理器
        
        Args:
            operation_name: 操作名称
        """
        
        # 开始监控
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent()
        
        process = psutil.Process()
        start_process_memory = process.memory_info().rss
        
        try:
            yield
        finally:
            # 结束监控
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent()
            end_process_memory = process.memory_info().rss
            
            # 记录性能数据
            performance_data = {
                'operation': operation_name,
                'timestamp': datetime.now().isoformat(),
                'duration': end_time - start_time,
                'memory_usage': {
                    'start_mb': start_memory / 1024 / 1024,
                    'end_mb': end_memory / 1024 / 1024,
                    'delta_mb': (end_memory - start_memory) / 1024 / 1024
                },
                'process_memory': {
                    'start_mb': start_process_memory / 1024 / 1024,
                    'end_mb': end_process_memory / 1024 / 1024,
                    'delta_mb': (end_process_memory - start_process_memory) / 1024 / 1024
                },
                'cpu_usage': {
                    'start_percent': start_cpu,
                    'end_percent': end_cpu
                }
            }
            
            self.performance_logs.append(performance_data)
            
            # 保存日志
            self._save_performance_log(performance_data)
    
    def _save_performance_log(self, performance_data: Dict):
        """保存性能日志"""
        
        log_file = os.path.join(self.logs_dir, f"performance_{datetime.now().strftime('%Y%m%d')}.json")
        
        # 读取现有日志
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # 添加新日志
        logs.append(performance_data)
        
        # 保存日志
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    
    def start_resource_monitoring(self, interval: float = 1.0):
        """
        启动资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        
        if self.monitoring_active:
            print("⚠️  资源监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("🔍 资源监控已启动")
    
    def stop_resource_monitoring(self):
        """停止资源监控"""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("⏹️  资源监控已停止")
    
    def _resource_monitor_loop(self, interval: float):
        """资源监控循环"""
        
        while self.monitoring_active:
            try:
                # 获取系统资源信息
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取进程资源信息
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                resource_data = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_mb': memory.used / 1024 / 1024,
                        'memory_available_mb': memory.available / 1024 / 1024,
                        'disk_percent': disk.percent
                    },
                    'process': {
                        'cpu_percent': process_cpu,
                        'memory_mb': process_memory.rss / 1024 / 1024,
                        'memory_percent': process_memory.rss / memory.total * 100
                    }
                }
                
                self.resource_logs.append(resource_data)
                
                # 限制日志数量
                if len(self.resource_logs) > 1000:
                    self.resource_logs = self.resource_logs[-500:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"资源监控错误: {str(e)}")
                time.sleep(interval)
    
    def benchmark_model_performance(self, 
                                  models: Dict[str, Any],
                                  X_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_train: np.ndarray,
                                  y_test: np.ndarray,
                                  n_runs: int = 5) -> Dict:
        """
        模型性能基准测试
        
        Args:
            models: 模型字典
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练标签
            y_test: 测试标签
            n_runs: 运行次数
            
        Returns:
            基准测试结果
        """
        
        print("🏃‍♂️ 开始模型性能基准测试...")
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            print(f"  📊 测试模型: {model_name}")
            
            model_results = {
                'training_performance': [],
                'prediction_performance': [],
                'accuracy_scores': [],
                'resource_usage': []
            }
            
            for run in range(n_runs):
                print(f"    🔄 运行 {run + 1}/{n_runs}")
                
                # 克隆模型以确保每次运行的独立性
                model_copy = clone(model)
                
                # 训练性能测试
                with self.monitor_performance(f"{model_name}_training_run_{run}"):
                    start_time = time.time()
                    model_copy.fit(X_train, y_train)
                    training_time = time.time() - start_time
                
                # 预测性能测试
                with self.monitor_performance(f"{model_name}_prediction_run_{run}"):
                    start_time = time.time()
                    y_pred = model_copy.predict(X_test)
                    prediction_time = time.time() - start_time
                
                # 计算准确率
                accuracy = accuracy_score(y_test, y_pred)
                
                # 记录结果
                model_results['training_performance'].append(training_time)
                model_results['prediction_performance'].append(prediction_time)
                model_results['accuracy_scores'].append(accuracy)
                
                # 获取最新的性能日志
                if self.performance_logs:
                    latest_logs = self.performance_logs[-2:]  # 训练和预测的日志
                    model_results['resource_usage'].extend(latest_logs)
            
            # 计算统计信息
            model_results['statistics'] = {
                'training_time': {
                    'mean': np.mean(model_results['training_performance']),
                    'std': np.std(model_results['training_performance']),
                    'min': np.min(model_results['training_performance']),
                    'max': np.max(model_results['training_performance'])
                },
                'prediction_time': {
                    'mean': np.mean(model_results['prediction_performance']),
                    'std': np.std(model_results['prediction_performance']),
                    'min': np.min(model_results['prediction_performance']),
                    'max': np.max(model_results['prediction_performance'])
                },
                'accuracy': {
                    'mean': np.mean(model_results['accuracy_scores']),
                    'std': np.std(model_results['accuracy_scores']),
                    'min': np.min(model_results['accuracy_scores']),
                    'max': np.max(model_results['accuracy_scores'])
                }
            }
            
            benchmark_results[model_name] = model_results
        
        print("✅ 模型性能基准测试完成")
        return benchmark_results
    
    def analyze_performance_trends(self, benchmark_results: Dict) -> Dict:
        """分析性能趋势"""
        
        trends = {
            'efficiency_ranking': {},
            'stability_ranking': {},
            'resource_efficiency': {},
            'performance_comparison': {}
        }
        
        try:
            # 效率排名（训练时间 + 预测时间）
            efficiency_scores = {}
            for model_name, results in benchmark_results.items():
                stats = results['statistics']
                total_time = stats['training_time']['mean'] + stats['prediction_time']['mean']
                accuracy = stats['accuracy']['mean']
                
                # 效率评分 = 准确率 / 总时间
                efficiency_scores[model_name] = accuracy / total_time if total_time > 0 else 0
            
            # 按效率排序
            sorted_efficiency = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
            trends['efficiency_ranking'] = {
                'ranking': [{'model': model, 'score': score} for model, score in sorted_efficiency],
                'best_model': sorted_efficiency[0][0] if sorted_efficiency else None
            }
            
            # 稳定性排名（标准差越小越稳定）
            stability_scores = {}
            for model_name, results in benchmark_results.items():
                stats = results['statistics']
                # 综合稳定性评分（准确率稳定性 + 时间稳定性）
                acc_stability = 1 / (stats['accuracy']['std'] + 1e-6)
                time_stability = 1 / (stats['training_time']['std'] + stats['prediction_time']['std'] + 1e-6)
                
                stability_scores[model_name] = (acc_stability + time_stability) / 2
            
            sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            trends['stability_ranking'] = {
                'ranking': [{'model': model, 'score': score} for model, score in sorted_stability],
                'most_stable_model': sorted_stability[0][0] if sorted_stability else None
            }
            
            # 资源效率分析
            for model_name, results in benchmark_results.items():
                if results['resource_usage']:
                    memory_usage = [log['process_memory']['delta_mb'] for log in results['resource_usage'] 
                                  if 'process_memory' in log]
                    
                    if memory_usage:
                        trends['resource_efficiency'][model_name] = {
                            'avg_memory_usage_mb': np.mean(memory_usage),
                            'max_memory_usage_mb': np.max(memory_usage),
                            'memory_efficiency_score': stats['accuracy']['mean'] / (np.mean(memory_usage) + 1e-6)
                        }
            
        except Exception as e:
            trends['error'] = str(e)
        
        return trends


class StabilityTester:
    """稳定性测试器"""
    
    def __init__(self, output_dir: str = "stability_testing"):
        """
        初始化稳定性测试器
        
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
    
    def comprehensive_stability_test(self, 
                                   models: Dict[str, Any],
                                   X_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_train: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: List[str] = None) -> Dict:
        """
        综合稳定性测试
        
        Args:
            models: 模型字典
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练标签
            y_test: 测试标签
            feature_names: 特征名称
            
        Returns:
            稳定性测试结果
        """
        
        print("🔬 开始综合稳定性测试...")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        stability_results = {}
        
        for model_name, model in models.items():
            print(f"  🧪 测试模型: {model_name}")
            
            model_stability = {
                'data_distribution_stability': {},
                'noise_robustness': {},
                'feature_perturbation': {},
                'temporal_stability': {},
                'cross_validation_stability': {},
                'overall_stability_score': 0
            }
            
            try:
                # 1. 数据分布稳定性测试
                model_stability['data_distribution_stability'] = self._test_data_distribution_stability(
                    model, X_train, X_test, y_train, y_test
                )
                
                # 2. 噪声鲁棒性测试
                model_stability['noise_robustness'] = self._test_noise_robustness(
                    model, X_train, X_test, y_train, y_test
                )
                
                # 3. 特征扰动测试
                model_stability['feature_perturbation'] = self._test_feature_perturbation(
                    model, X_train, X_test, y_train, y_test, feature_names
                )
                
                # 4. 时间稳定性测试
                model_stability['temporal_stability'] = self._test_temporal_stability(
                    model, X_train, X_test, y_train, y_test
                )
                
                # 5. 交叉验证稳定性
                model_stability['cross_validation_stability'] = self._test_cross_validation_stability(
                    model, X_train, y_train
                )
                
                # 计算综合稳定性评分
                model_stability['overall_stability_score'] = self._calculate_overall_stability_score(
                    model_stability
                )
                
            except Exception as e:
                model_stability['error'] = str(e)
                print(f"    ❌ {model_name} 稳定性测试失败: {str(e)}")
            
            stability_results[model_name] = model_stability
        
        # 生成稳定性对比分析
        stability_results['comparison'] = self._compare_model_stability(stability_results)
        
        print("✅ 综合稳定性测试完成")
        return stability_results
    
    def _test_data_distribution_stability(self, 
                                        model: Any,
                                        X_train: np.ndarray,
                                        X_test: np.ndarray,
                                        y_train: np.ndarray,
                                        y_test: np.ndarray) -> Dict:
        """测试数据分布稳定性"""
        
        try:
            # 训练模型
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            
            # 基准性能
            baseline_pred = model_copy.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            
            distribution_tests = []
            
            # 测试不同的数据分布变化
            distribution_changes = [
                ('标准化', lambda X: StandardScaler().fit_transform(X)),
                ('添加偏移', lambda X: X + np.random.normal(0, 0.1, X.shape)),
                ('缩放变换', lambda X: X * np.random.uniform(0.8, 1.2, X.shape[1])),
                ('特征重排', lambda X: X[:, np.random.permutation(X.shape[1])]),
            ]
            
            for change_name, transform_func in distribution_changes:
                try:
                    # 应用变换
                    X_test_transformed = transform_func(X_test.copy())
                    
                    # 预测
                    pred_transformed = model_copy.predict(X_test_transformed)
                    accuracy_transformed = accuracy_score(y_test, pred_transformed)
                    
                    # 计算稳定性指标
                    accuracy_drop = baseline_accuracy - accuracy_transformed
                    stability_score = max(0, 1 - abs(accuracy_drop) / baseline_accuracy)
                    
                    distribution_tests.append({
                        'transformation': change_name,
                        'baseline_accuracy': baseline_accuracy,
                        'transformed_accuracy': accuracy_transformed,
                        'accuracy_drop': accuracy_drop,
                        'stability_score': stability_score
                    })
                    
                except Exception as e:
                    distribution_tests.append({
                        'transformation': change_name,
                        'error': str(e)
                    })
            
            # 计算平均稳定性
            valid_tests = [t for t in distribution_tests if 'error' not in t]
            avg_stability = np.mean([t['stability_score'] for t in valid_tests]) if valid_tests else 0
            
            return {
                'tests': distribution_tests,
                'average_stability_score': avg_stability,
                'most_stable_transformation': max(valid_tests, key=lambda x: x['stability_score'])['transformation'] if valid_tests else None,
                'least_stable_transformation': min(valid_tests, key=lambda x: x['stability_score'])['transformation'] if valid_tests else None
            }
            
        except Exception as e:
            return {'error': f'Data distribution stability test failed: {str(e)}'}
    
    def _test_noise_robustness(self, 
                             model: Any,
                             X_train: np.ndarray,
                             X_test: np.ndarray,
                             y_train: np.ndarray,
                             y_test: np.ndarray) -> Dict:
        """测试噪声鲁棒性"""
        
        try:
            # 训练模型
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            
            # 基准性能
            baseline_pred = model_copy.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            
            noise_tests = []
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
            
            for noise_level in noise_levels:
                try:
                    # 添加高斯噪声
                    noise = np.random.normal(0, noise_level, X_test.shape)
                    X_test_noisy = X_test + noise
                    
                    # 预测
                    pred_noisy = model_copy.predict(X_test_noisy)
                    accuracy_noisy = accuracy_score(y_test, pred_noisy)
                    
                    # 计算鲁棒性指标
                    accuracy_drop = baseline_accuracy - accuracy_noisy
                    robustness_score = max(0, 1 - abs(accuracy_drop) / baseline_accuracy)
                    
                    # 预测一致性
                    consistency = np.mean(baseline_pred == pred_noisy)
                    
                    noise_tests.append({
                        'noise_level': noise_level,
                        'baseline_accuracy': baseline_accuracy,
                        'noisy_accuracy': accuracy_noisy,
                        'accuracy_drop': accuracy_drop,
                        'robustness_score': robustness_score,
                        'prediction_consistency': consistency
                    })
                    
                except Exception as e:
                    noise_tests.append({
                        'noise_level': noise_level,
                        'error': str(e)
                    })
            
            # 计算平均鲁棒性
            valid_tests = [t for t in noise_tests if 'error' not in t]
            avg_robustness = np.mean([t['robustness_score'] for t in valid_tests]) if valid_tests else 0
            avg_consistency = np.mean([t['prediction_consistency'] for t in valid_tests]) if valid_tests else 0
            
            return {
                'tests': noise_tests,
                'average_robustness_score': avg_robustness,
                'average_consistency': avg_consistency,
                'robustness_trend': self._analyze_robustness_trend(valid_tests)
            }
            
        except Exception as e:
            return {'error': f'Noise robustness test failed: {str(e)}'}
    
    def _analyze_robustness_trend(self, tests: List[Dict]) -> Dict:
        """分析鲁棒性趋势"""
        
        if len(tests) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        noise_levels = [t['noise_level'] for t in tests]
        robustness_scores = [t['robustness_score'] for t in tests]
        
        # 计算趋势斜率
        slope, intercept, r_value, p_value, std_err = stats.linregress(noise_levels, robustness_scores)
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'trend_direction': 'decreasing' if slope < -0.1 else 'stable' if abs(slope) <= 0.1 else 'increasing',
            'degradation_rate': abs(slope) if slope < 0 else 0
        }
    
    def _test_feature_perturbation(self, 
                                 model: Any,
                                 X_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_train: np.ndarray,
                                 y_test: np.ndarray,
                                 feature_names: List[str]) -> Dict:
        """测试特征扰动稳定性"""
        
        try:
            # 训练模型
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            
            # 基准性能
            baseline_pred = model_copy.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            
            feature_tests = []
            
            # 对每个特征进行扰动测试
            for i, feature_name in enumerate(feature_names):
                try:
                    X_test_perturbed = X_test.copy()
                    
                    # 特征扰动策略
                    perturbation_methods = [
                        ('零值替换', lambda x: np.zeros_like(x)),
                        ('均值替换', lambda x: np.full_like(x, np.mean(x))),
                        ('随机噪声', lambda x: x + np.random.normal(0, np.std(x) * 0.1, x.shape)),
                        ('随机打乱', lambda x: np.random.permutation(x))
                    ]
                    
                    feature_perturbation_results = []
                    
                    for method_name, perturbation_func in perturbation_methods:
                        X_test_perturbed[:, i] = perturbation_func(X_test[:, i])
                        
                        pred_perturbed = model_copy.predict(X_test_perturbed)
                        accuracy_perturbed = accuracy_score(y_test, pred_perturbed)
                        
                        accuracy_drop = baseline_accuracy - accuracy_perturbed
                        sensitivity = abs(accuracy_drop) / baseline_accuracy
                        
                        feature_perturbation_results.append({
                            'method': method_name,
                            'accuracy_drop': accuracy_drop,
                            'sensitivity': sensitivity
                        })
                    
                    # 计算特征重要性（基于扰动敏感性）
                    avg_sensitivity = np.mean([r['sensitivity'] for r in feature_perturbation_results])
                    
                    feature_tests.append({
                        'feature_name': feature_name,
                        'feature_index': i,
                        'perturbation_results': feature_perturbation_results,
                        'average_sensitivity': avg_sensitivity,
                        'importance_rank': 0  # 将在后面计算
                    })
                    
                except Exception as e:
                    feature_tests.append({
                        'feature_name': feature_name,
                        'feature_index': i,
                        'error': str(e)
                    })
            
            # 计算重要性排名
            valid_tests = [t for t in feature_tests if 'error' not in t]
            if valid_tests:
                sorted_tests = sorted(valid_tests, key=lambda x: x['average_sensitivity'], reverse=True)
                for rank, test in enumerate(sorted_tests):
                    test['importance_rank'] = rank + 1
            
            return {
                'feature_tests': feature_tests,
                'most_sensitive_features': sorted(valid_tests, key=lambda x: x['average_sensitivity'], reverse=True)[:5] if valid_tests else [],
                'least_sensitive_features': sorted(valid_tests, key=lambda x: x['average_sensitivity'])[:5] if valid_tests else [],
                'average_feature_sensitivity': np.mean([t['average_sensitivity'] for t in valid_tests]) if valid_tests else 0
            }
            
        except Exception as e:
            return {'error': f'Feature perturbation test failed: {str(e)}'}
    
    def _test_temporal_stability(self, 
                               model: Any,
                               X_train: np.ndarray,
                               X_test: np.ndarray,
                               y_train: np.ndarray,
                               y_test: np.ndarray) -> Dict:
        """测试时间稳定性"""
        
        try:
            # 模拟时间序列稳定性测试
            temporal_results = []
            n_time_splits = 5
            
            # 将测试数据分成时间段
            split_size = len(X_test) // n_time_splits
            
            # 训练模型
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            
            for i in range(n_time_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_time_splits - 1 else len(X_test)
                
                X_time_split = X_test[start_idx:end_idx]
                y_time_split = y_test[start_idx:end_idx]
                
                if len(X_time_split) > 0:
                    pred_time_split = model_copy.predict(X_time_split)
                    accuracy_time_split = accuracy_score(y_time_split, pred_time_split)
                    
                    temporal_results.append({
                        'time_period': i + 1,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'accuracy': accuracy_time_split,
                        'sample_count': len(X_time_split)
                    })
            
            # 计算时间稳定性指标
            accuracies = [r['accuracy'] for r in temporal_results]
            
            temporal_stability = {
                'time_periods': temporal_results,
                'accuracy_variance': np.var(accuracies),
                'accuracy_std': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'accuracy_range': np.max(accuracies) - np.min(accuracies),
                'stability_score': 1 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0
            }
            
            return temporal_stability
            
        except Exception as e:
            return {'error': f'Temporal stability test failed: {str(e)}'}
    
    def _test_cross_validation_stability(self, 
                                       model: Any,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray) -> Dict:
        """测试交叉验证稳定性"""
        
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            # 使用分层K折交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 多次运行交叉验证以测试稳定性
            cv_runs = []
            n_runs = 3
            
            for run in range(n_runs):
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                cv_runs.append({
                    'run': run + 1,
                    'scores': cv_scores.tolist(),
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores)
                })
            
            # 计算跨运行的稳定性
            mean_scores = [run['mean_score'] for run in cv_runs]
            std_scores = [run['std_score'] for run in cv_runs]
            
            cv_stability = {
                'cv_runs': cv_runs,
                'inter_run_stability': {
                    'mean_score_variance': np.var(mean_scores),
                    'mean_score_std': np.std(mean_scores),
                    'avg_intra_run_std': np.mean(std_scores),
                    'stability_score': 1 - (np.std(mean_scores) / np.mean(mean_scores)) if np.mean(mean_scores) > 0 else 0
                },
                'overall_cv_performance': {
                    'grand_mean': np.mean(mean_scores),
                    'grand_std': np.mean(std_scores),
                    'best_run': max(cv_runs, key=lambda x: x['mean_score'])['run'],
                    'worst_run': min(cv_runs, key=lambda x: x['mean_score'])['run']
                }
            }
            
            return cv_stability
            
        except Exception as e:
            return {'error': f'Cross-validation stability test failed: {str(e)}'}
    
    def _calculate_overall_stability_score(self, model_stability: Dict) -> float:
        """计算综合稳定性评分"""
        
        try:
            scores = []
            weights = []
            
            # 数据分布稳定性 (权重: 0.25)
            if ('data_distribution_stability' in model_stability and 
                'average_stability_score' in model_stability['data_distribution_stability']):
                scores.append(model_stability['data_distribution_stability']['average_stability_score'])
                weights.append(0.25)
            
            # 噪声鲁棒性 (权重: 0.25)
            if ('noise_robustness' in model_stability and 
                'average_robustness_score' in model_stability['noise_robustness']):
                scores.append(model_stability['noise_robustness']['average_robustness_score'])
                weights.append(0.25)
            
            # 特征扰动稳定性 (权重: 0.2)
            if ('feature_perturbation' in model_stability and 
                'average_feature_sensitivity' in model_stability['feature_perturbation']):
                # 敏感性越低，稳定性越高
                sensitivity = model_stability['feature_perturbation']['average_feature_sensitivity']
                stability = max(0, 1 - sensitivity)
                scores.append(stability)
                weights.append(0.2)
            
            # 时间稳定性 (权重: 0.15)
            if ('temporal_stability' in model_stability and 
                'stability_score' in model_stability['temporal_stability']):
                scores.append(model_stability['temporal_stability']['stability_score'])
                weights.append(0.15)
            
            # 交叉验证稳定性 (权重: 0.15)
            if ('cross_validation_stability' in model_stability and 
                'inter_run_stability' in model_stability['cross_validation_stability']):
                cv_stability = model_stability['cross_validation_stability']['inter_run_stability']['stability_score']
                scores.append(cv_stability)
                weights.append(0.15)
            
            # 计算加权平均
            if scores and weights:
                # 归一化权重
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                overall_score = sum(s * w for s, w in zip(scores, normalized_weights))
                return max(0, min(1, overall_score))  # 确保在[0,1]范围内
            else:
                return 0.0
                
        except Exception as e:
            print(f"计算综合稳定性评分失败: {str(e)}")
            return 0.0
    
    def _compare_model_stability(self, stability_results: Dict) -> Dict:
        """对比模型稳定性"""
        
        comparison = {
            'stability_ranking': [],
            'stability_categories': {},
            'best_performers': {},
            'recommendations': []
        }
        
        try:
            # 提取有效的稳定性评分
            model_scores = {}
            for model_name, results in stability_results.items():
                if model_name != 'comparison' and 'overall_stability_score' in results:
                    model_scores[model_name] = results['overall_stability_score']
            
            # 稳定性排名
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['stability_ranking'] = [
                {'model': model, 'stability_score': score, 'rank': i + 1}
                for i, (model, score) in enumerate(sorted_models)
            ]
            
            # 稳定性分类
            for model, score in model_scores.items():
                if score >= 0.8:
                    category = '高稳定性'
                elif score >= 0.6:
                    category = '中等稳定性'
                elif score >= 0.4:
                    category = '低稳定性'
                else:
                    category = '不稳定'
                
                if category not in comparison['stability_categories']:
                    comparison['stability_categories'][category] = []
                comparison['stability_categories'][category].append(model)
            
            # 各维度最佳表现者
            dimensions = [
                ('data_distribution_stability', 'average_stability_score', '数据分布稳定性'),
                ('noise_robustness', 'average_robustness_score', '噪声鲁棒性'),
                ('temporal_stability', 'stability_score', '时间稳定性')
            ]
            
            for dim_key, score_key, dim_name in dimensions:
                best_model = None
                best_score = -1
                
                for model_name, results in stability_results.items():
                    if (model_name != 'comparison' and 
                        dim_key in results and 
                        score_key in results[dim_key]):
                        score = results[dim_key][score_key]
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                
                if best_model:
                    comparison['best_performers'][dim_name] = {
                        'model': best_model,
                        'score': best_score
                    }
            
            # 生成建议
            if sorted_models:
                best_model = sorted_models[0][0]
                worst_model = sorted_models[-1][0]
                
                comparison['recommendations'].extend([
                    f"推荐使用 {best_model}，具有最高的稳定性评分 ({sorted_models[0][1]:.3f})",
                    f"需要改进 {worst_model} 的稳定性 ({sorted_models[-1][1]:.3f})"
                ])
                
                # 基于稳定性分类的建议
                if '不稳定' in comparison['stability_categories']:
                    unstable_models = comparison['stability_categories']['不稳定']
                    comparison['recommendations'].append(
                        f"以下模型稳定性较差，建议重新调参或更换算法: {', '.join(unstable_models)}"
                    )
                
                if '高稳定性' in comparison['stability_categories']:
                    stable_models = comparison['stability_categories']['高稳定性']
                    comparison['recommendations'].append(
                        f"以下模型表现稳定，适合生产环境部署: {', '.join(stable_models)}"
                    )
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison


class PerformanceStabilitySystem:
    """性能监控和稳定性测试集成系统"""
    
    def __init__(self, output_dir: str = "performance_stability_system"):
        """
        初始化性能稳定性系统
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化子系统
        self.performance_monitor = PerformanceMonitor(
            os.path.join(output_dir, "performance_monitoring")
        )
        self.stability_tester = StabilityTester(
            os.path.join(output_dir, "stability_testing")
        )
        
        # 创建报告目录
        self.reports_dir = os.path.join(output_dir, "reports")
        self.charts_dir = os.path.join(output_dir, "charts")
        
        for dir_path in [self.reports_dir, self.charts_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def comprehensive_analysis(self, 
                             models: Dict[str, Any],
                             X_train: np.ndarray,
                             X_test: np.ndarray,
                             y_train: np.ndarray,
                             y_test: np.ndarray,
                             feature_names: List[str] = None) -> Dict:
        """
        综合性能和稳定性分析
        
        Args:
            models: 模型字典
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练标签
            y_test: 测试标签
            feature_names: 特征名称
            
        Returns:
            综合分析结果
        """
        
        print("🚀 开始综合性能和稳定性分析...")
        
        # 启动资源监控
        self.performance_monitor.start_resource_monitoring()
        
        try:
            results = {
                'performance_analysis': {},
                'stability_analysis': {},
                'integrated_analysis': {},
                'recommendations': [],
                'charts': {}
            }
            
            # 性能基准测试
            print("📊 执行性能基准测试...")
            benchmark_results = self.performance_monitor.benchmark_model_performance(
                models, X_train, X_test, y_train, y_test
            )
            results['performance_analysis']['benchmark'] = benchmark_results
            
            # 性能趋势分析
            print("📈 分析性能趋势...")
            performance_trends = self.performance_monitor.analyze_performance_trends(benchmark_results)
            results['performance_analysis']['trends'] = performance_trends
            
            # 稳定性测试
            print("🔬 执行稳定性测试...")
            stability_results = self.stability_tester.comprehensive_stability_test(
                models, X_train, X_test, y_train, y_test, feature_names
            )
            results['stability_analysis'] = stability_results
            
            # 集成分析
            print("🔄 执行集成分析...")
            results['integrated_analysis'] = self._integrated_analysis(
                benchmark_results, performance_trends, stability_results
            )
            
            # 生成建议
            results['recommendations'] = self._generate_comprehensive_recommendations(results)
            
            # 生成图表
            results['charts'] = self._generate_comprehensive_charts(results)
            
            print("✅ 综合分析完成")
            return results
            
        finally:
            # 停止资源监控
            self.performance_monitor.stop_resource_monitoring()
    
    def _integrated_analysis(self, 
                           benchmark_results: Dict,
                           performance_trends: Dict,
                           stability_results: Dict) -> Dict:
        """集成分析"""
        
        integrated = {
            'model_rankings': {},
            'performance_stability_matrix': {},
            'optimal_model_selection': {},
            'trade_off_analysis': {}
        }
        
        try:
            # 提取模型评分
            model_scores = {}
            
            for model_name in benchmark_results.keys():
                scores = {
                    'performance_score': 0,
                    'stability_score': 0,
                    'efficiency_score': 0,
                    'overall_score': 0
                }
                
                # 性能评分（基于准确率）
                if 'statistics' in benchmark_results[model_name]:
                    accuracy = benchmark_results[model_name]['statistics']['accuracy']['mean']
                    scores['performance_score'] = accuracy
                
                # 稳定性评分
                if (model_name in stability_results and 
                    'overall_stability_score' in stability_results[model_name]):
                    scores['stability_score'] = stability_results[model_name]['overall_stability_score']
                
                # 效率评分
                if ('efficiency_ranking' in performance_trends and 
                    'ranking' in performance_trends['efficiency_ranking']):
                    for rank_info in performance_trends['efficiency_ranking']['ranking']:
                        if rank_info['model'] == model_name:
                            # 归一化效率评分
                            max_score = max([r['score'] for r in performance_trends['efficiency_ranking']['ranking']])
                            scores['efficiency_score'] = rank_info['score'] / max_score if max_score > 0 else 0
                            break
                
                # 综合评分（加权平均）
                weights = {'performance': 0.4, 'stability': 0.4, 'efficiency': 0.2}
                scores['overall_score'] = (
                    scores['performance_score'] * weights['performance'] +
                    scores['stability_score'] * weights['stability'] +
                    scores['efficiency_score'] * weights['efficiency']
                )
                
                model_scores[model_name] = scores
            
            # 模型排名
            integrated['model_rankings'] = {
                'by_performance': sorted(model_scores.items(), key=lambda x: x[1]['performance_score'], reverse=True),
                'by_stability': sorted(model_scores.items(), key=lambda x: x[1]['stability_score'], reverse=True),
                'by_efficiency': sorted(model_scores.items(), key=lambda x: x[1]['efficiency_score'], reverse=True),
                'by_overall': sorted(model_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
            }
            
            # 性能-稳定性矩阵
            integrated['performance_stability_matrix'] = self._create_performance_stability_matrix(model_scores)
            
            # 最优模型选择
            integrated['optimal_model_selection'] = self._select_optimal_models(model_scores)
            
            # 权衡分析
            integrated['trade_off_analysis'] = self._analyze_trade_offs(model_scores)
            
        except Exception as e:
            integrated['error'] = str(e)
        
        return integrated
    
    def _create_performance_stability_matrix(self, model_scores: Dict) -> Dict:
        """创建性能-稳定性矩阵"""
        
        matrix = {
            'high_performance_high_stability': [],
            'high_performance_low_stability': [],
            'low_performance_high_stability': [],
            'low_performance_low_stability': []
        }
        
        # 计算阈值（中位数）
        performance_scores = [scores['performance_score'] for scores in model_scores.values()]
        stability_scores = [scores['stability_score'] for scores in model_scores.values()]
        
        perf_threshold = np.median(performance_scores) if performance_scores else 0.5
        stab_threshold = np.median(stability_scores) if stability_scores else 0.5
        
        for model_name, scores in model_scores.items():
            perf = scores['performance_score']
            stab = scores['stability_score']
            
            if perf >= perf_threshold and stab >= stab_threshold:
                matrix['high_performance_high_stability'].append(model_name)
            elif perf >= perf_threshold and stab < stab_threshold:
                matrix['high_performance_low_stability'].append(model_name)
            elif perf < perf_threshold and stab >= stab_threshold:
                matrix['low_performance_high_stability'].append(model_name)
            else:
                matrix['low_performance_low_stability'].append(model_name)
        
        return matrix
    
    def _select_optimal_models(self, model_scores: Dict) -> Dict:
        """选择最优模型"""
        
        selection = {}
        
        # 综合最优
        best_overall = max(model_scores.items(), key=lambda x: x[1]['overall_score'])
        selection['best_overall'] = {
            'model': best_overall[0],
            'scores': best_overall[1]
        }
        
        # 性能最优
        best_performance = max(model_scores.items(), key=lambda x: x[1]['performance_score'])
        selection['best_performance'] = {
            'model': best_performance[0],
            'scores': best_performance[1]
        }
        
        # 稳定性最优
        best_stability = max(model_scores.items(), key=lambda x: x[1]['stability_score'])
        selection['best_stability'] = {
            'model': best_stability[0],
            'scores': best_stability[1]
        }
        
        # 效率最优
        best_efficiency = max(model_scores.items(), key=lambda x: x[1]['efficiency_score'])
        selection['best_efficiency'] = {
            'model': best_efficiency[0],
            'scores': best_efficiency[1]
        }
        
        return selection
    
    def _analyze_trade_offs(self, model_scores: Dict) -> Dict:
        """分析权衡关系"""
        
        trade_offs = {
            'performance_vs_stability': {},
            'performance_vs_efficiency': {},
            'stability_vs_efficiency': {},
            'correlations': {}
        }
        
        try:
            # 提取评分数组
            models = list(model_scores.keys())
            performance = [model_scores[m]['performance_score'] for m in models]
            stability = [model_scores[m]['stability_score'] for m in models]
            efficiency = [model_scores[m]['efficiency_score'] for m in models]
            
            # 计算相关性
            if len(models) > 1:
                trade_offs['correlations'] = {
                    'performance_stability': float(np.corrcoef(performance, stability)[0, 1]) if len(set(performance)) > 1 and len(set(stability)) > 1 else 0,
                    'performance_efficiency': float(np.corrcoef(performance, efficiency)[0, 1]) if len(set(performance)) > 1 and len(set(efficiency)) > 1 else 0,
                    'stability_efficiency': float(np.corrcoef(stability, efficiency)[0, 1]) if len(set(stability)) > 1 and len(set(efficiency)) > 1 else 0
                }
            
            # 权衡分析
            trade_offs['performance_vs_stability']['analysis'] = self._analyze_dimension_trade_off(
                models, performance, stability, 'performance', 'stability'
            )
            
            trade_offs['performance_vs_efficiency']['analysis'] = self._analyze_dimension_trade_off(
                models, performance, efficiency, 'performance', 'efficiency'
            )
            
            trade_offs['stability_vs_efficiency']['analysis'] = self._analyze_dimension_trade_off(
                models, stability, efficiency, 'stability', 'efficiency'
            )
            
        except Exception as e:
            trade_offs['error'] = str(e)
        
        return trade_offs
    
    def _analyze_dimension_trade_off(self, models: List[str], dim1_scores: List[float], 
                                   dim2_scores: List[float], dim1_name: str, dim2_name: str) -> Dict:
        """分析两个维度之间的权衡"""
        
        analysis = {
            'best_balance': None,
            'extreme_cases': {},
            'trade_off_strength': 0
        }
        
        try:
            # 找到最佳平衡点（两个维度的乘积最大）
            balance_scores = [d1 * d2 for d1, d2 in zip(dim1_scores, dim2_scores)]
            best_balance_idx = np.argmax(balance_scores)
            
            analysis['best_balance'] = {
                'model': models[best_balance_idx],
                f'{dim1_name}_score': dim1_scores[best_balance_idx],
                f'{dim2_name}_score': dim2_scores[best_balance_idx],
                'balance_score': balance_scores[best_balance_idx]
            }
            
            # 极端情况
            analysis['extreme_cases'] = {
                f'high_{dim1_name}_low_{dim2_name}': {
                    'models': [models[i] for i in range(len(models)) 
                             if dim1_scores[i] > np.median(dim1_scores) and dim2_scores[i] < np.median(dim2_scores)]
                },
                f'low_{dim1_name}_high_{dim2_name}': {
                    'models': [models[i] for i in range(len(models)) 
                             if dim1_scores[i] < np.median(dim1_scores) and dim2_scores[i] > np.median(dim2_scores)]
                }
            }
            
            # 权衡强度（负相关性的强度）
            if len(set(dim1_scores)) > 1 and len(set(dim2_scores)) > 1:
                correlation = np.corrcoef(dim1_scores, dim2_scores)[0, 1]
                analysis['trade_off_strength'] = max(0, -correlation)  # 负相关表示权衡
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_comprehensive_recommendations(self, results: Dict) -> List[str]:
        """生成综合建议"""
        
        recommendations = []
        
        try:
            # 基于集成分析的建议
            if 'integrated_analysis' in results and 'optimal_model_selection' in results['integrated_analysis']:
                selection = results['integrated_analysis']['optimal_model_selection']
                
                if 'best_overall' in selection:
                    best_model = selection['best_overall']['model']
                    recommendations.append(f"🏆 综合推荐模型: {best_model}（综合评分最高）")
                
                if 'best_performance' in selection and 'best_stability' in selection:
                    perf_model = selection['best_performance']['model']
                    stab_model = selection['best_stability']['model']
                    
                    if perf_model != stab_model:
                        recommendations.append(f"⚖️  性能与稳定性权衡: {perf_model}（高性能）vs {stab_model}（高稳定性）")
            
            # 基于性能-稳定性矩阵的建议
            if ('integrated_analysis' in results and 
                'performance_stability_matrix' in results['integrated_analysis']):
                matrix = results['integrated_analysis']['performance_stability_matrix']
                
                if matrix['high_performance_high_stability']:
                    models = ', '.join(matrix['high_performance_high_stability'])
                    recommendations.append(f"✅ 理想选择: {models}（高性能高稳定性）")
                
                if matrix['high_performance_low_stability']:
                    models = ', '.join(matrix['high_performance_low_stability'])
                    recommendations.append(f"⚠️  需要稳定性改进: {models}（高性能但稳定性不足）")
                
                if matrix['low_performance_high_stability']:
                    models = ', '.join(matrix['low_performance_high_stability'])
                    recommendations.append(f"🔧 需要性能优化: {models}（稳定但性能不足）")
            
            # 基于权衡分析的建议
            if ('integrated_analysis' in results and 
                'trade_off_analysis' in results['integrated_analysis'] and
                'correlations' in results['integrated_analysis']['trade_off_analysis']):
                
                correlations = results['integrated_analysis']['trade_off_analysis']['correlations']
                
                if correlations.get('performance_stability', 0) < -0.5:
                    recommendations.append("📊 发现性能与稳定性存在明显权衡，需要根据业务需求选择")
                
                if correlations.get('performance_efficiency', 0) < -0.5:
                    recommendations.append("⚡ 发现性能与效率存在权衡，高性能模型可能需要更多资源")
            
            # 基于稳定性测试的建议
            if ('stability_analysis' in results and 
                'comparison' in results['stability_analysis'] and
                'recommendations' in results['stability_analysis']['comparison']):
                
                stability_recommendations = results['stability_analysis']['comparison']['recommendations']
                recommendations.extend([f"🔬 {rec}" for rec in stability_recommendations])
            
            # 通用建议
            recommendations.extend([
                "📈 建议定期监控模型性能，及时发现性能退化",
                "🔄 建议建立模型更新机制，应对数据分布变化",
                "📊 建议在生产环境中持续收集性能和稳定性指标",
                "🛡️  建议为关键业务场景选择高稳定性模型"
            ])
            
        except Exception as e:
            recommendations.append(f"❌ 建议生成失败: {str(e)}")
        
        return recommendations
    
    def _generate_comprehensive_charts(self, results: Dict) -> Dict:
        """生成综合图表"""
        
        charts = {}
        
        try:
            # 性能对比图表
            charts['performance_comparison'] = self._create_performance_comparison_chart(results)
            
            # 稳定性雷达图
            charts['stability_radar'] = self._create_stability_radar_chart(results)
            
            # 性能-稳定性散点图
            charts['performance_stability_scatter'] = self._create_performance_stability_scatter(results)
            
            # 资源使用趋势图
            charts['resource_usage_trend'] = self._create_resource_usage_trend_chart()
            
        except Exception as e:
            charts['error'] = str(e)
        
        return charts
    
    def _create_performance_comparison_chart(self, results: Dict) -> str:
        """创建性能对比图表"""
        
        try:
            if ('performance_analysis' not in results or 
                'benchmark' not in results['performance_analysis']):
                return None
            
            benchmark_data = results['performance_analysis']['benchmark']
            
            models = list(benchmark_data.keys())
            training_times = [benchmark_data[model]['statistics']['training_time']['mean'] 
                            for model in models]
            prediction_times = [benchmark_data[model]['statistics']['prediction_time']['mean'] 
                              for model in models]
            accuracies = [benchmark_data[model]['statistics']['accuracy']['mean'] 
                         for model in models]
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('训练时间对比', '预测时间对比', '准确率对比', '效率综合评分'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 训练时间
            fig.add_trace(
                go.Bar(x=models, y=training_times, name='训练时间(秒)', 
                      marker_color='lightblue'),
                row=1, col=1
            )
            
            # 预测时间
            fig.add_trace(
                go.Bar(x=models, y=prediction_times, name='预测时间(秒)', 
                      marker_color='lightgreen'),
                row=1, col=2
            )
            
            # 准确率
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name='准确率', 
                      marker_color='lightcoral'),
                row=2, col=1
            )
            
            # 效率评分（准确率/总时间）
            efficiency_scores = [acc / (train_t + pred_t) if (train_t + pred_t) > 0 else 0
                               for acc, train_t, pred_t in zip(accuracies, training_times, prediction_times)]
            
            fig.add_trace(
                go.Bar(x=models, y=efficiency_scores, name='效率评分', 
                      marker_color='gold'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="模型性能综合对比",
                showlegend=False,
                height=600
            )
            
            chart_path = os.path.join(self.charts_dir, "performance_comparison.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"创建性能对比图表失败: {str(e)}")
            return None
    
    def _create_stability_radar_chart(self, results: Dict) -> str:
        """创建稳定性雷达图"""
        
        try:
            if ('stability_analysis' not in results):
                return None
            
            stability_data = results['stability_analysis']
            models = [model for model in stability_data.keys() if model != 'comparison']
            
            if not models:
                return None
            
            # 稳定性维度
            dimensions = [
                ('data_distribution_stability', 'average_stability_score', '数据分布稳定性'),
                ('noise_robustness', 'average_robustness_score', '噪声鲁棒性'),
                ('temporal_stability', 'stability_score', '时间稳定性'),
                ('cross_validation_stability', 'inter_run_stability', '交叉验证稳定性')
            ]
            
            fig = go.Figure()
            
            for model in models:
                if model in stability_data and 'overall_stability_score' in stability_data[model]:
                    scores = []
                    labels = []
                    
                    for dim_key, score_key, label in dimensions:
                        if (dim_key in stability_data[model] and 
                            isinstance(stability_data[model][dim_key], dict)):
                            
                            if score_key == 'inter_run_stability':
                                # 特殊处理交叉验证稳定性
                                if ('inter_run_stability' in stability_data[model][dim_key] and
                                    'stability_score' in stability_data[model][dim_key]['inter_run_stability']):
                                    score = stability_data[model][dim_key]['inter_run_stability']['stability_score']
                                else:
                                    score = 0
                            else:
                                score = stability_data[model][dim_key].get(score_key, 0)
                            
                            scores.append(score)
                            labels.append(label)
                    
                    if scores:
                        # 添加第一个点到末尾以闭合雷达图
                        scores.append(scores[0])
                        labels.append(labels[0])
                        
                        fig.add_trace(go.Scatterpolar(
                            r=scores,
                            theta=labels,
                            fill='toself',
                            name=model
                        ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="模型稳定性雷达图"
            )
            
            chart_path = os.path.join(self.charts_dir, "stability_radar.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"创建稳定性雷达图失败: {str(e)}")
            return None
    
    def _create_performance_stability_scatter(self, results: Dict) -> str:
        """创建性能-稳定性散点图"""
        
        try:
            if ('integrated_analysis' not in results or 
                'model_rankings' not in results['integrated_analysis']):
                return None
            
            rankings = results['integrated_analysis']['model_rankings']
            
            if 'by_overall' not in rankings:
                return None
            
            models = []
            performance_scores = []
            stability_scores = []
            overall_scores = []
            
            for model, scores in rankings['by_overall']:
                models.append(model)
                performance_scores.append(scores['performance_score'])
                stability_scores.append(scores['stability_score'])
                overall_scores.append(scores['overall_score'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_scores,
                y=stability_scores,
                mode='markers+text',
                text=models,
                textposition="top center",
                marker=dict(
                    size=[score * 50 + 10 for score in overall_scores],  # 气泡大小表示综合评分
                    color=overall_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="综合评分")
                ),
                name='模型'
            ))
            
            fig.update_layout(
                title="模型性能-稳定性散点图",
                xaxis_title="性能评分",
                yaxis_title="稳定性评分",
                showlegend=False
            )
            
            # 添加象限分割线
            fig.add_hline(y=np.median(stability_scores), line_dash="dash", line_color="gray")
            fig.add_vline(x=np.median(performance_scores), line_dash="dash", line_color="gray")
            
            chart_path = os.path.join(self.charts_dir, "performance_stability_scatter.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"创建性能-稳定性散点图失败: {str(e)}")
            return None
    
    def _create_resource_usage_trend_chart(self) -> str:
        """创建资源使用趋势图"""
        
        try:
            if not self.performance_monitor.resource_logs:
                return None
            
            timestamps = []
            cpu_usage = []
            memory_usage = []
            process_memory = []
            
            for log in self.performance_monitor.resource_logs:
                timestamps.append(datetime.fromisoformat(log['timestamp']))
                cpu_usage.append(log['system']['cpu_percent'])
                memory_usage.append(log['system']['memory_percent'])
                process_memory.append(log['process']['memory_mb'])
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('CPU使用率 (%)', '系统内存使用率 (%)', '进程内存使用 (MB)'),
                shared_xaxes=True
            )
            
            # CPU使用率
            fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_usage, name='CPU使用率', 
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # 系统内存使用率
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_usage, name='系统内存使用率', 
                          line=dict(color='blue')),
                row=2, col=1
            )
            
            # 进程内存使用
            fig.add_trace(
                go.Scatter(x=timestamps, y=process_memory, name='进程内存使用', 
                          line=dict(color='green')),
                row=3, col=1
            )
            
            fig.update_layout(
                title_text="资源使用趋势",
                showlegend=False,
                height=800
            )
            
            chart_path = os.path.join(self.charts_dir, "resource_usage_trend.html")
            fig.write_html(chart_path)
            
            return chart_path
            
        except Exception as e:
            print(f"创建资源使用趋势图失败: {str(e)}")
            return None
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """生成综合报告"""
        
        try:
            report_content = self._create_report_content(results)
            
            report_path = os.path.join(self.reports_dir, 
                                     f"performance_stability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"📄 综合报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"生成综合报告失败: {str(e)}")
            return None
    
    def _create_report_content(self, results: Dict) -> str:
        """创建报告内容"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>模型性能与稳定性综合分析报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .summary-box {{
                    background-color: #e8f5e8;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 5px solid #4CAF50;
                }}
                .warning-box {{
                    background-color: #fff3cd;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 5px solid #ffc107;
                }}
                .error-box {{
                    background-color: #f8d7da;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 5px solid #dc3545;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                .chart-container {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .recommendation {{
                    background-color: #d1ecf1;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                    border-left: 4px solid #17a2b8;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 模型性能与稳定性综合分析报告</h1>
                
                <div class="summary-box">
                    <h3>📊 报告概要</h3>
                    <p><strong>生成时间:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                    <p><strong>分析范围:</strong> 模型性能基准测试、稳定性评估、资源使用监控</p>
                    <p><strong>评估维度:</strong> 准确率、训练时间、预测时间、内存使用、稳定性评分</p>
                </div>
                
                {self._generate_performance_section(results)}
                
                {self._generate_stability_section(results)}
                
                {self._generate_integrated_section(results)}
                
                {self._generate_recommendations_section(results)}
                
                <div class="summary-box">
                    <h3>📈 总结</h3>
                    <p>本报告通过综合性能基准测试和多维度稳定性评估，为模型选择和优化提供了数据支持。
                    建议根据具体业务需求，在性能、稳定性和效率之间找到最佳平衡点。</p>
                </div>
                
                <footer style="margin-top: 50px; text-align: center; color: #666;">
                    <p>报告由性能监控和稳定性测试系统自动生成</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_performance_section(self, results: Dict) -> str:
        """生成性能分析部分"""
        
        section = "<h2>📊 性能分析</h2>"
        
        try:
            if ('performance_analysis' in results and 
                'benchmark' in results['performance_analysis']):
                
                benchmark_data = results['performance_analysis']['benchmark']
                
                section += "<h3>🏃‍♂️ 性能基准测试结果</h3>"
                section += "<table>"
                section += "<tr><th>模型</th><th>平均训练时间(秒)</th><th>平均预测时间(秒)</th><th>平均准确率</th><th>准确率标准差</th></tr>"
                
                for model_name, data in benchmark_data.items():
                    if 'statistics' in data:
                        stats = data['statistics']
                        section += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{stats['training_time']['mean']:.4f}</td>
                            <td>{stats['prediction_time']['mean']:.4f}</td>
                            <td>{stats['accuracy']['mean']:.4f}</td>
                            <td>{stats['accuracy']['std']:.4f}</td>
                        </tr>
                        """
                
                section += "</table>"
                
                # 性能趋势分析
                if 'trends' in results['performance_analysis']:
                    trends = results['performance_analysis']['trends']
                    section += "<h3>📈 性能趋势分析</h3>"
                    
                    if 'efficiency_ranking' in trends and 'best_model' in trends['efficiency_ranking']:
                        best_model = trends['efficiency_ranking']['best_model']
                        section += f"<div class='summary-box'>🏆 效率最优模型: <strong>{best_model}</strong></div>"
                    
                    if 'stability_ranking' in trends and 'most_stable_model' in trends['stability_ranking']:
                        stable_model = trends['stability_ranking']['most_stable_model']
                        section += f"<div class='summary-box'>🛡️ 最稳定模型: <strong>{stable_model}</strong></div>"
        
        except Exception as e:
            section += f"<div class='error-box'>❌ 性能分析部分生成失败: {str(e)}</div>"
        
        return section
    
    def _generate_stability_section(self, results: Dict) -> str:
        """生成稳定性分析部分"""
        
        section = "<h2>🔬 稳定性分析</h2>"
        
        try:
            if 'stability_analysis' in results:
                stability_data = results['stability_analysis']
                
                section += "<h3>🧪 稳定性测试结果</h3>"
                section += "<table>"
                section += "<tr><th>模型</th><th>综合稳定性评分</th><th>数据分布稳定性</th><th>噪声鲁棒性</th><th>时间稳定性</th></tr>"
                
                for model_name, data in stability_data.items():
                    if model_name != 'comparison' and 'overall_stability_score' in data:
                        overall_score = data['overall_stability_score']
                        
                        # 提取各维度评分
                        dist_score = data.get('data_distribution_stability', {}).get('average_stability_score', 'N/A')
                        noise_score = data.get('noise_robustness', {}).get('average_robustness_score', 'N/A')
                        temp_score = data.get('temporal_stability', {}).get('stability_score', 'N/A')
                        
                        section += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{overall_score:.4f}</td>
                            <td>{dist_score if isinstance(dist_score, str) else f'{dist_score:.4f}'}</td>
                            <td>{noise_score if isinstance(noise_score, str) else f'{noise_score:.4f}'}</td>
                            <td>{temp_score if isinstance(temp_score, str) else f'{temp_score:.4f}'}</td>
                        </tr>
                        """
                
                section += "</table>"
                
                # 稳定性对比分析
                if 'comparison' in stability_data:
                    comparison = stability_data['comparison']
                    section += "<h3>⚖️ 稳定性对比分析</h3>"
                    
                    if 'stability_categories' in comparison:
                        categories = comparison['stability_categories']
                        for category, models in categories.items():
                            if models:
                                section += f"<div class='metric'><strong>{category}:</strong> {', '.join(models)}</div>"
        
        except Exception as e:
            section += f"<div class='error-box'>❌ 稳定性分析部分生成失败: {str(e)}</div>"
        
        return section
    
    def _generate_integrated_section(self, results: Dict) -> str:
        """生成集成分析部分"""
        
        section = "<h2>🔄 集成分析</h2>"
        
        try:
            if 'integrated_analysis' in results:
                integrated = results['integrated_analysis']
                
                # 最优模型选择
                if 'optimal_model_selection' in integrated:
                    selection = integrated['optimal_model_selection']
                    section += "<h3>🏆 最优模型选择</h3>"
                    
                    for category, info in selection.items():
                        if isinstance(info, dict) and 'model' in info:
                            category_name = {
                                'best_overall': '综合最优',
                                'best_performance': '性能最优',
                                'best_stability': '稳定性最优',
                                'best_efficiency': '效率最优'
                            }.get(category, category)
                            
                            section += f"<div class='summary-box'><strong>{category_name}:</strong> {info['model']}</div>"
                
                # 性能-稳定性矩阵
                if 'performance_stability_matrix' in integrated:
                    matrix = integrated['performance_stability_matrix']
                    section += "<h3>📊 性能-稳定性矩阵</h3>"
                    
                    for category, models in matrix.items():
                        if models:
                            category_name = {
                                'high_performance_high_stability': '高性能高稳定性',
                                'high_performance_low_stability': '高性能低稳定性',
                                'low_performance_high_stability': '低性能高稳定性',
                                'low_performance_low_stability': '低性能低稳定性'
                            }.get(category, category)
                            
                            section += f"<div class='metric'><strong>{category_name}:</strong> {', '.join(models)}</div>"
        
        except Exception as e:
            section += f"<div class='error-box'>❌ 集成分析部分生成失败: {str(e)}</div>"
        
        return section
    
    def _generate_recommendations_section(self, results: Dict) -> str:
        """生成建议部分"""
        
        section = "<h2>💡 建议与总结</h2>"
        
        try:
            if 'recommendations' in results:
                recommendations = results['recommendations']
                section += "<h3>🎯 具体建议</h3>"
                
                for rec in recommendations:
                    section += f"<div class='recommendation'>{rec}</div>"
        
        except Exception as e:
            section += f"<div class='error-box'>❌ 建议部分生成失败: {str(e)}</div>"
        
        return section


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # 创建性能稳定性系统
    system = PerformanceStabilitySystem("performance_stability_output")
    
    # 执行综合分析
    results = system.comprehensive_analysis(
        models, X_train, X_test, y_train, y_test
    )
    
    # 生成报告
    report_path = system.generate_comprehensive_report(results)
    
    print(f"✅ 分析完成，报告路径: {report_path}")
    print(f"📊 图表目录: {system.charts_dir}")
    print(f"📄 报告目录: {system.reports_dir}")
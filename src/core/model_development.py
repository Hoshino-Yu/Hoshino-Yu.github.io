#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SME信贷风险评估系统 - 模型开发模块
集成多种机器学习算法，支持自动化模型训练、调参和选择
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
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class AdvancedModelDeveloper:
    """
    高级模型开发器
    支持多种机器学习算法的自动化训练、调参和模型选择
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化模型开发器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.target_name = None
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型配置
        self._initialize_models()
        
        # 超参数搜索空间
        self._initialize_hyperparameter_spaces()
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../输出结果/model_development.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_models(self):
        """初始化模型库"""
        self.models = {
            # 传统机器学习模型
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'svm': SVC(
                random_state=self.random_state, probability=True
            ),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'ada_boost': AdaBoostClassifier(
                random_state=self.random_state
            ),
            'extra_trees': ExtraTreesClassifier(
                random_state=self.random_state, n_jobs=-1
            ),
            'mlp': MLPClassifier(
                random_state=self.random_state, max_iter=500
            ),
            
            # 梯度提升模型
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state, eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state, verbose=-1
            ),
            'catboost': CatBoostClassifier(
                random_state=self.random_state, verbose=False
            )
        }
        
        self.logger.info(f"初始化了 {len(self.models)} 个模型")
    
    def _initialize_hyperparameter_spaces(self):
        """初始化超参数搜索空间"""
        self.hyperparameter_spaces = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'ada_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0, 2.0]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'catboost': {
                'iterations': [50, 100, 200],
                'depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        }
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据
        
        Args:
            data: 输入数据
            target_column: 目标列名
            test_size: 测试集比例
            stratify: 是否分层抽样
            
        Returns:
            训练集和测试集
        """
        self.logger.info("开始准备训练数据...")
        
        # 分离特征和目标变量
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        self.feature_names = list(X.columns)
        self.target_name = target_column
        
        # 处理分类特征
        X = self._encode_categorical_features(X)
        
        # 数据分割
        stratify_param = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        self.logger.info(f"数据分割完成 - 训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.logger.info(f"对列 {col} 进行标签编码")
        
        return X_encoded
    
    def train_all_models(self, cv_folds: int = 5) -> Dict[str, Dict]:
        """
        训练所有模型
        
        Args:
            cv_folds: 交叉验证折数
            
        Returns:
            所有模型的性能结果
        """
        self.logger.info("开始训练所有模型...")
        
        if self.X_train is None:
            raise ValueError("请先调用 prepare_data 方法准备数据")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"训练模型: {model_name}")
                
                # 训练模型
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model
                
                # 交叉验证
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, 
                    cv=cv, scoring='roc_auc', n_jobs=-1
                )
                
                # 测试集预测
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算性能指标
                performance = self._calculate_performance_metrics(
                    self.y_test, y_pred, y_pred_proba
                )
                performance['cv_mean'] = cv_scores.mean()
                performance['cv_std'] = cv_scores.std()
                
                self.model_performance[model_name] = performance
                
                self.logger.info(f"模型 {model_name} 训练完成 - AUC: {performance['auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 失败: {str(e)}")
                continue
        
        # 选择最佳模型
        self._select_best_model()
        
        return self.model_performance
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray, 
                                     y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算性能指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def _select_best_model(self):
        """选择最佳模型"""
        if not self.model_performance:
            return
        
        # 基于AUC选择最佳模型
        best_model_name = max(
            self.model_performance.keys(),
            key=lambda x: self.model_performance[x]['auc']
        )
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        self.logger.info(f"最佳模型: {best_model_name} (AUC: {self.model_performance[best_model_name]['auc']:.4f})")
    
    def hyperparameter_optimization(self, model_names: Optional[List[str]] = None,
                                  optimization_method: str = 'optuna',
                                  n_trials: int = 100) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            model_names: 要优化的模型名称列表
            optimization_method: 优化方法 ('grid', 'random', 'optuna')
            n_trials: Optuna优化试验次数
            
        Returns:
            优化结果
        """
        self.logger.info(f"开始超参数优化 - 方法: {optimization_method}")
        
        if model_names is None:
            # 选择性能最好的前5个模型进行优化
            sorted_models = sorted(
                self.model_performance.items(),
                key=lambda x: x[1]['auc'],
                reverse=True
            )
            model_names = [name for name, _ in sorted_models[:5]]
        
        optimization_results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
                
            try:
                self.logger.info(f"优化模型: {model_name}")
                
                if optimization_method == 'optuna':
                    result = self._optuna_optimization(model_name, n_trials)
                elif optimization_method == 'grid':
                    result = self._grid_search_optimization(model_name)
                elif optimization_method == 'random':
                    result = self._random_search_optimization(model_name)
                else:
                    raise ValueError(f"不支持的优化方法: {optimization_method}")
                
                optimization_results[model_name] = result
                
            except Exception as e:
                self.logger.error(f"优化模型 {model_name} 失败: {str(e)}")
                continue
        
        return optimization_results
    
    def _optuna_optimization(self, model_name: str, n_trials: int) -> Dict:
        """使用Optuna进行超参数优化"""
        def objective(trial):
            # 根据模型类型定义超参数
            params = {}
            param_space = self.hyperparameter_spaces.get(model_name, {})
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
            
            # 创建模型
            model = self._create_model_with_params(model_name, params)
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # 使用最佳参数训练模型
        best_model = self._create_model_with_params(model_name, study.best_params)
        best_model.fit(self.X_train, self.y_train)
        
        # 更新训练好的模型
        self.trained_models[f"{model_name}_optimized"] = best_model
        
        # 计算优化后的性能
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        optimized_performance = self._calculate_performance_metrics(
            self.y_test, y_pred, y_pred_proba
        )
        
        self.model_performance[f"{model_name}_optimized"] = optimized_performance
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'performance': optimized_performance
        }
    
    def _create_model_with_params(self, model_name: str, params: Dict) -> Any:
        """根据参数创建模型"""
        base_model = self.models[model_name]
        model_class = type(base_model)
        
        # 合并默认参数和新参数
        default_params = base_model.get_params()
        default_params.update(params)
        
        return model_class(**default_params)
    
    def create_ensemble_model(self, top_n: int = 5) -> VotingClassifier:
        """
        创建集成模型
        
        Args:
            top_n: 选择性能最好的前N个模型
            
        Returns:
            集成模型
        """
        self.logger.info(f"创建集成模型 - 选择前{top_n}个模型")
        
        # 选择性能最好的模型
        sorted_models = sorted(
            self.model_performance.items(),
            key=lambda x: x[1]['auc'],
            reverse=True
        )
        
        top_models = []
        for model_name, _ in sorted_models[:top_n]:
            if model_name in self.trained_models:
                top_models.append((model_name, self.trained_models[model_name]))
        
        # 创建投票分类器
        ensemble_model = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        # 训练集成模型
        ensemble_model.fit(self.X_train, self.y_train)
        
        # 评估集成模型
        y_pred = ensemble_model.predict(self.X_test)
        y_pred_proba = ensemble_model.predict_proba(self.X_test)[:, 1]
        
        ensemble_performance = self._calculate_performance_metrics(
            self.y_test, y_pred, y_pred_proba
        )
        
        # 保存集成模型
        self.trained_models['ensemble'] = ensemble_model
        self.model_performance['ensemble'] = ensemble_performance
        
        self.logger.info(f"集成模型创建完成 - AUC: {ensemble_performance['auc']:.4f}")
        
        return ensemble_model
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称，默认使用最佳模型
            
        Returns:
            特征重要性DataFrame
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError(f"模型 {model_name} 不存在")
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"模型 {model_name} 不支持特征重要性")
            return pd.DataFrame()
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_models(self, output_dir: str = '../输出结果/models'):
        """
        保存训练好的模型
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            self.logger.info(f"模型 {model_name} 已保存到: {model_path}")
        
        # 保存性能结果
        performance_path = os.path.join(output_dir, 'model_performance.json')
        with open(performance_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_performance, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"模型性能结果已保存到: {performance_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        加载保存的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        return joblib.load(model_path)
    
    def generate_model_comparison_report(self) -> Dict:
        """
        生成模型比较报告
        
        Returns:
            模型比较报告
        """
        if not self.model_performance:
            return {}
        
        # 创建比较表格
        comparison_df = pd.DataFrame(self.model_performance).T
        comparison_df = comparison_df.round(4)
        
        # 排序
        comparison_df = comparison_df.sort_values('auc', ascending=False)
        
        report = {
            'model_comparison': comparison_df.to_dict(),
            'best_model': {
                'name': self.best_model_name,
                'performance': self.model_performance.get(self.best_model_name, {})
            },
            'summary': {
                'total_models': len(self.model_performance),
                'best_auc': comparison_df['auc'].max(),
                'mean_auc': comparison_df['auc'].mean(),
                'std_auc': comparison_df['auc'].std()
            }
        }
        
        return report
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练好的模型进行预测
        
        Args:
            X: 输入特征
            model_name: 模型名称，默认使用最佳模型
            
        Returns:
            预测结果和预测概率
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError(f"模型 {model_name} 不存在")
        
        # 编码分类特征
        X_encoded = self._encode_categorical_features(X)
        
        # 预测
        predictions = model.predict(X_encoded)
        probabilities = model.predict_proba(X_encoded)[:, 1] if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities


def main():
    """主函数 - 演示模型开发流程"""
    try:
        # 加载处理后的数据
        data_path = '../输出结果/processed_data_final.xlsx'
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            return
        
        data = pd.read_excel(data_path)
        print(f"加载数据: {data.shape}")
        
        # 假设目标列为 '风险等级' 或 '违约标签'
        target_columns = ['风险等级', '违约标签', '是否违约', 'default']
        target_column = None
        
        for col in target_columns:
            if col in data.columns:
                target_column = col
                break
        
        if target_column is None:
            print("未找到目标列，请检查数据")
            return
        
        # 创建模型开发器
        developer = AdvancedModelDeveloper()
        
        # 准备数据
        X_train, X_test, y_train, y_test = developer.prepare_data(
            data, target_column, test_size=0.2
        )
        
        # 训练所有模型
        performance_results = developer.train_all_models()
        print(f"训练了 {len(performance_results)} 个模型")
        
        # 创建集成模型
        ensemble_model = developer.create_ensemble_model(top_n=3)
        
        # 保存模型
        developer.save_models()
        
        # 生成比较报告
        report = developer.generate_model_comparison_report()
        print("\n模型比较报告:")
        print(f"最佳模型: {report['best_model']['name']}")
        print(f"最佳AUC: {report['summary']['best_auc']:.4f}")
        
        print("\n模型开发完成！")
        
    except Exception as e:
        print(f"模型开发过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
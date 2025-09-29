"""
综合报告生成系统
重新实现完整的报告生成功能，包含项目概述、解决方案说明、算法分析、效益分析和图表数据
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

class ComprehensiveReportGenerator:
    """
    综合报告生成器
    生成包含项目概述、技术方案、算法分析、效益评估的完整报告
    """
    
    def __init__(self, output_dir: str = '../输出结果/comprehensive_reports'):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.report_data = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'assets'), exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        pio.templates.default = "plotly_white"
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.output_dir, 'report_generation.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, 
                                    project_info: Dict,
                                    model_results: Dict,
                                    data_info: Dict,
                                    business_metrics: Optional[Dict] = None) -> Dict:
        """
        生成综合报告
        
        Args:
            project_info: 项目信息
            model_results: 模型结果
            data_info: 数据信息
            business_metrics: 业务指标
            
        Returns:
            报告生成结果
        """
        self.logger.info("开始生成综合报告")
        
        # 收集报告数据
        self.report_data = {
            'project_info': project_info,
            'model_results': model_results,
            'data_info': data_info,
            'business_metrics': business_metrics or {},
            'generation_time': datetime.now()
        }
        
        # 生成各部分内容
        report_sections = {}
        
        # 1. 项目概述
        report_sections['project_overview'] = self._generate_project_overview()
        
        # 2. 解决方案说明
        report_sections['solution_description'] = self._generate_solution_description()
        
        # 3. 算法实现与分析
        report_sections['algorithm_analysis'] = self._generate_algorithm_analysis()
        
        # 4. 模型性能评估
        report_sections['performance_evaluation'] = self._generate_performance_evaluation()
        
        # 5. 经济效益分析
        report_sections['economic_benefits'] = self._generate_economic_benefits_analysis()
        
        # 6. 社会效益分析
        report_sections['social_benefits'] = self._generate_social_benefits_analysis()
        
        # 7. 风险分析与建议
        report_sections['risk_analysis'] = self._generate_risk_analysis()
        
        # 8. 图表和数据分析
        report_sections['charts_analysis'] = self._generate_charts_analysis()
        
        # 生成HTML报告
        html_report = self._generate_html_report(report_sections)
        
        # 生成PDF报告（如果需要）
        pdf_report = self._generate_pdf_report(report_sections)
        
        # 生成执行摘要
        executive_summary = self._generate_executive_summary(report_sections)
        
        self.logger.info("综合报告生成完成")
        
        return {
            'html_report_path': html_report,
            'pdf_report_path': pdf_report,
            'executive_summary': executive_summary,
            'report_sections': report_sections,
            'charts_generated': len([f for f in os.listdir(os.path.join(self.output_dir, 'charts')) if f.endswith('.html')])
        }
    
    def _generate_project_overview(self) -> Dict:
        """生成项目概述"""
        
        project_info = self.report_data.get('project_info', {})
        
        overview = {
            'title': '智能信贷风险评估系统',
            'background': self._get_project_background(),
            'industry_context': self._get_industry_context(),
            'algorithm_advantages': self._get_algorithm_advantages(),
            'key_features': self._get_key_features(),
            'technical_highlights': self._get_technical_highlights()
        }
        
        return overview
    
    def _get_project_background(self) -> str:
        """获取项目背景"""
        return """
        随着金融科技的快速发展和监管要求的不断提高，传统的信贷风险评估方法已无法满足现代金融机构的需求。
        本项目旨在构建一个基于机器学习的智能信贷风险评估系统，通过多维度数据分析和先进算法模型，
        实现对借款人信用风险的精准评估，提高放贷决策的科学性和准确性。
        
        该系统结合了传统金融风控经验与现代数据科学技术，能够处理大规模、多维度的客户数据，
        自动识别潜在风险因素，为金融机构提供可靠的决策支持。
        """
    
    def _get_industry_context(self) -> str:
        """获取行业背景"""
        return """
        **金融行业发展趋势：**
        - 数字化转型加速，传统风控模式面临挑战
        - 监管科技（RegTech）要求更高的风险管理标准
        - 普惠金融发展需要更精准的风险定价能力
        - 大数据和人工智能技术在金融领域广泛应用
        
        **市场需求分析：**
        - 银行业不良贷款率控制压力增大
        - 互联网金融平台需要更智能的风控系统
        - 小微企业融资难题需要创新解决方案
        - 个人消费信贷市场快速增长
        """
    
    def _get_algorithm_advantages(self) -> List[str]:
        """获取算法优势"""
        return [
            "多算法集成：结合随机森林、梯度提升、逻辑回归等多种算法，提高预测准确性",
            "特征工程优化：自动化特征选择和工程，挖掘数据深层价值",
            "实时评估：支持实时风险评估，满足快速放贷需求",
            "可解释性强：提供详细的风险因子分析，满足监管透明度要求",
            "自适应学习：模型可根据新数据持续优化，保持预测准确性",
            "鲁棒性好：对异常数据和噪声具有良好的抗干扰能力"
        ]
    
    def _get_key_features(self) -> List[str]:
        """获取关键特性"""
        return [
            "智能数据预处理：自动处理缺失值、异常值和数据标准化",
            "多维度特征工程：构建衍生特征，提升模型表现",
            "模型自动选择：基于交叉验证自动选择最优模型",
            "风险等级分类：提供明确的风险等级划分和建议",
            "实时监控预警：建立风险监控机制，及时发现异常",
            "可视化分析：丰富的图表展示，直观呈现分析结果"
        ]
    
    def _get_technical_highlights(self) -> List[str]:
        """获取技术亮点"""
        return [
            "采用Ensemble Learning提高模型稳定性和准确性",
            "集成SHAP和LIME可解释性分析工具",
            "实现多种交叉验证策略确保模型泛化能力",
            "支持增量学习和在线模型更新",
            "建立完整的模型监控和性能评估体系",
            "提供RESTful API接口，便于系统集成"
        ]
    
    def _generate_solution_description(self) -> Dict:
        """生成解决方案说明"""
        
        solution = {
            'architecture_overview': self._get_architecture_overview(),
            'data_flow': self._get_data_flow_description(),
            'model_pipeline': self._get_model_pipeline_description(),
            'deployment_strategy': self._get_deployment_strategy(),
            'integration_approach': self._get_integration_approach()
        }
        
        return solution
    
    def _get_architecture_overview(self) -> str:
        """获取架构概述"""
        return """
        **系统架构设计：**
        
        本系统采用分层架构设计，包含数据层、算法层、应用层和展示层：
        
        1. **数据层**：负责数据收集、存储和预处理
           - 支持多种数据源接入（数据库、文件、API）
           - 实现数据质量检查和清洗
           - 提供数据版本管理和备份机制
        
        2. **算法层**：核心机器学习算法实现
           - 多算法模型训练和评估
           - 特征工程和选择
           - 模型优化和调参
        
        3. **应用层**：业务逻辑和服务接口
           - 风险评估服务
           - 模型管理和监控
           - 决策支持系统
        
        4. **展示层**：用户界面和可视化
           - Web界面展示
           - 报告生成和导出
           - 实时监控面板
        """
    
    def _get_data_flow_description(self) -> str:
        """获取数据流描述"""
        return """
        **数据处理流程：**
        
        1. **数据收集**：从多个数据源收集客户信息
           - 基本信息：年龄、性别、教育程度、职业等
           - 财务信息：收入、支出、资产、负债等
           - 信用历史：历史借贷记录、还款情况等
           - 行为数据：消费习惯、社交网络等
        
        2. **数据预处理**：
           - 数据清洗：处理缺失值、异常值、重复值
           - 数据转换：类型转换、编码转换
           - 数据标准化：归一化、标准化处理
        
        3. **特征工程**：
           - 特征构造：创建衍生特征和组合特征
           - 特征选择：基于统计方法和机器学习方法选择重要特征
           - 特征变换：主成分分析、特征缩放等
        
        4. **模型训练**：
           - 数据分割：训练集、验证集、测试集划分
           - 模型训练：多算法并行训练
           - 超参数优化：网格搜索、贝叶斯优化等
        """
    
    def _get_model_pipeline_description(self) -> str:
        """获取模型管道描述"""
        return """
        **机器学习管道：**
        
        1. **数据预处理管道**：
           - 自动化数据清洗和转换
           - 特征工程和选择
           - 数据验证和质量检查
        
        2. **模型训练管道**：
           - 多算法并行训练
           - 交叉验证和模型评估
           - 超参数自动调优
        
        3. **模型评估管道**：
           - 多维度性能评估
           - 模型可解释性分析
           - 业务指标计算
        
        4. **模型部署管道**：
           - 模型版本管理
           - A/B测试支持
           - 在线服务部署
        
        5. **监控反馈管道**：
           - 模型性能监控
           - 数据漂移检测
           - 自动重训练触发
        """
    
    def _get_deployment_strategy(self) -> str:
        """获取部署策略"""
        return """
        **部署策略：**
        
        1. **开发环境**：
           - 本地开发和测试
           - 模型实验和调优
           - 代码版本控制
        
        2. **测试环境**：
           - 集成测试和性能测试
           - 数据安全和隐私测试
           - 用户接受度测试
        
        3. **预生产环境**：
           - 生产数据验证
           - 负载测试
           - 灾难恢复测试
        
        4. **生产环境**：
           - 高可用部署
           - 自动扩缩容
           - 实时监控和告警
        
        **技术栈：**
        - 后端：Python + FastAPI/Flask
        - 前端：Streamlit/React
        - 数据库：PostgreSQL/MongoDB
        - 缓存：Redis
        - 容器化：Docker + Kubernetes
        - 监控：Prometheus + Grafana
        """
    
    def _get_integration_approach(self) -> str:
        """获取集成方法"""
        return """
        **系统集成方案：**
        
        1. **API集成**：
           - RESTful API接口
           - GraphQL查询支持
           - Webhook事件通知
        
        2. **数据集成**：
           - ETL数据管道
           - 实时数据流处理
           - 批量数据同步
        
        3. **安全集成**：
           - OAuth2.0认证
           - JWT令牌管理
           - 数据加密传输
        
        4. **监控集成**：
           - 日志聚合分析
           - 性能指标监控
           - 异常告警机制
        """
    
    def _generate_algorithm_analysis(self) -> Dict:
        """生成算法分析"""
        
        model_results = self.report_data.get('model_results', {})
        
        analysis = {
            'algorithm_selection': self._get_algorithm_selection_rationale(),
            'model_comparison': self._analyze_model_performance(model_results),
            'feature_importance': self._analyze_feature_importance(model_results),
            'model_interpretability': self._analyze_model_interpretability(),
            'optimization_process': self._describe_optimization_process(),
            'validation_strategy': self._describe_validation_strategy()
        }
        
        return analysis
    
    def _get_algorithm_selection_rationale(self) -> str:
        """获取算法选择理由"""
        return """
        **算法选择依据：**
        
        1. **随机森林 (Random Forest)**：
           - 优势：对过拟合鲁棒，处理非线性关系能力强
           - 适用场景：特征维度较高，需要特征重要性分析
           - 在信贷风险评估中表现稳定，可解释性好
        
        2. **梯度提升树 (Gradient Boosting)**：
           - 优势：预测精度高，能够捕捉复杂的特征交互
           - 适用场景：对预测精度要求较高的场景
           - 在处理不平衡数据时表现优异
        
        3. **逻辑回归 (Logistic Regression)**：
           - 优势：模型简单，可解释性强，训练速度快
           - 适用场景：需要概率输出和线性关系建模
           - 符合传统金融风控的理解习惯
        
        4. **支持向量机 (SVM)**：
           - 优势：在高维空间表现良好，内存效率高
           - 适用场景：特征维度高于样本数量的情况
           - 对异常值相对鲁棒
        
        **集成策略**：
        采用Voting Classifier和Stacking方法，结合多个算法的优势，
        提高整体预测性能和模型稳定性。
        """
    
    def _analyze_model_performance(self, model_results: Dict) -> Dict:
        """分析模型性能"""
        
        if not model_results:
            return {'message': '暂无模型结果数据'}
        
        performance_analysis = {}
        
        # 提取各模型的性能指标
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                
                performance_analysis[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'performance_summary': self._generate_performance_summary(metrics)
                }
        
        # 找出最佳模型
        if performance_analysis:
            best_model = max(performance_analysis.keys(), 
                           key=lambda x: performance_analysis[x].get('f1_score', 0))
            performance_analysis['best_model'] = best_model
            performance_analysis['ranking'] = self._rank_models(performance_analysis)
        
        return performance_analysis
    
    def _generate_performance_summary(self, metrics: Dict) -> str:
        """生成性能摘要"""
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        
        if f1 >= 0.9:
            level = "优秀"
        elif f1 >= 0.8:
            level = "良好"
        elif f1 >= 0.7:
            level = "中等"
        else:
            level = "需要改进"
        
        return f"""
        模型整体表现{level}，F1分数为{f1:.3f}。
        准确率{accuracy:.3f}表明模型预测的整体正确性，
        精确率{precision:.3f}反映了正例预测的可靠性，
        召回率{recall:.3f}体现了模型识别正例的能力。
        """
    
    def _rank_models(self, performance_analysis: Dict) -> List[Dict]:
        """对模型进行排名"""
        
        models = [(name, data) for name, data in performance_analysis.items() 
                 if name not in ['best_model', 'ranking'] and isinstance(data, dict)]
        
        # 按F1分数排序
        models.sort(key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        ranking = []
        for i, (model_name, data) in enumerate(models, 1):
            ranking.append({
                'rank': i,
                'model': model_name,
                'f1_score': data.get('f1_score', 0),
                'accuracy': data.get('accuracy', 0),
                'overall_score': (data.get('f1_score', 0) + data.get('accuracy', 0)) / 2
            })
        
        return ranking
    
    def _analyze_feature_importance(self, model_results: Dict) -> Dict:
        """分析特征重要性"""
        
        feature_analysis = {
            'top_features': [],
            'feature_categories': {},
            'business_insights': []
        }
        
        # 从模型结果中提取特征重要性
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'feature_importance' in results:
                importance_data = results['feature_importance']
                
                if isinstance(importance_data, dict):
                    # 获取前10个重要特征
                    sorted_features = sorted(importance_data.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
                    
                    feature_analysis['top_features'].extend([
                        {
                            'feature': feature,
                            'importance': importance,
                            'model': model_name
                        }
                        for feature, importance in sorted_features
                    ])
        
        # 特征分类分析
        feature_analysis['feature_categories'] = self._categorize_features(feature_analysis['top_features'])
        
        # 业务洞察
        feature_analysis['business_insights'] = self._generate_feature_insights(feature_analysis['feature_categories'])
        
        return feature_analysis
    
    def _categorize_features(self, top_features: List[Dict]) -> Dict:
        """对特征进行分类"""
        
        categories = {
            '个人基本信息': [],
            '财务状况': [],
            '信用历史': [],
            '行为特征': [],
            '其他': []
        }
        
        # 特征分类规则
        personal_keywords = ['age', 'gender', 'education', 'marital', '年龄', '性别', '教育', '婚姻']
        financial_keywords = ['income', 'salary', 'amount', 'balance', '收入', '工资', '金额', '余额']
        credit_keywords = ['credit', 'loan', 'debt', 'payment', '信用', '贷款', '债务', '还款']
        behavior_keywords = ['frequency', 'count', 'duration', 'pattern', '频率', '次数', '持续', '模式']
        
        for feature_info in top_features:
            feature_name = feature_info['feature'].lower()
            
            if any(keyword in feature_name for keyword in personal_keywords):
                categories['个人基本信息'].append(feature_info)
            elif any(keyword in feature_name for keyword in financial_keywords):
                categories['财务状况'].append(feature_info)
            elif any(keyword in feature_name for keyword in credit_keywords):
                categories['信用历史'].append(feature_info)
            elif any(keyword in feature_name for keyword in behavior_keywords):
                categories['行为特征'].append(feature_info)
            else:
                categories['其他'].append(feature_info)
        
        return categories
    
    def _generate_feature_insights(self, feature_categories: Dict) -> List[str]:
        """生成特征洞察"""
        
        insights = []
        
        for category, features in feature_categories.items():
            if features:
                avg_importance = np.mean([f['importance'] for f in features])
                feature_count = len(features)
                
                if category == '个人基本信息' and avg_importance > 0.1:
                    insights.append(f"个人基本信息对风险评估影响显著，平均重要性为{avg_importance:.3f}")
                
                elif category == '财务状况' and avg_importance > 0.15:
                    insights.append(f"财务状况是风险评估的关键因素，包含{feature_count}个重要特征")
                
                elif category == '信用历史' and avg_importance > 0.2:
                    insights.append(f"信用历史表现出最高的预测价值，应重点关注相关数据质量")
                
                elif category == '行为特征' and feature_count >= 2:
                    insights.append(f"行为特征提供了{feature_count}个有价值的风险指标")
        
        if not insights:
            insights.append("特征重要性分析显示各类特征对风险预测都有一定贡献")
        
        return insights
    
    def _analyze_model_interpretability(self) -> Dict:
        """分析模型可解释性"""
        
        interpretability = {
            'shap_analysis': {
                'available': False,
                'description': 'SHAP (SHapley Additive exPlanations) 分析提供特征对预测结果的贡献度'
            },
            'lime_analysis': {
                'available': False,
                'description': 'LIME (Local Interpretable Model-agnostic Explanations) 提供局部可解释性'
            },
            'feature_interaction': {
                'description': '特征交互分析揭示特征之间的相互作用对预测的影响'
            },
            'decision_rules': {
                'description': '决策规则提取帮助理解模型的决策逻辑'
            }
        }
        
        return interpretability
    
    def _describe_optimization_process(self) -> str:
        """描述优化过程"""
        return """
        **模型优化过程：**
        
        1. **基线模型建立**：
           - 使用默认参数训练各算法模型
           - 建立性能基线和对比标准
           - 识别初始性能瓶颈
        
        2. **特征工程优化**：
           - 特征选择：移除冗余和无关特征
           - 特征构造：创建有意义的衍生特征
           - 特征变换：标准化、归一化处理
        
        3. **超参数调优**：
           - 网格搜索：系统性搜索参数空间
           - 随机搜索：高效探索参数组合
           - 贝叶斯优化：智能参数优化策略
        
        4. **模型集成**：
           - Voting：多模型投票决策
           - Stacking：分层模型集成
           - Blending：加权平均集成
        
        5. **性能验证**：
           - 交叉验证：确保模型泛化能力
           - 时间序列验证：验证时间稳定性
           - 业务指标验证：确保业务价值
        """
    
    def _describe_validation_strategy(self) -> str:
        """描述验证策略"""
        return """
        **模型验证策略：**
        
        1. **数据分割策略**：
           - 训练集（70%）：用于模型训练
           - 验证集（15%）：用于超参数调优
           - 测试集（15%）：用于最终性能评估
        
        2. **交叉验证方法**：
           - K折交叉验证：评估模型稳定性
           - 分层交叉验证：保持类别分布平衡
           - 时间序列交叉验证：考虑时间因素影响
        
        3. **性能评估指标**：
           - 分类指标：准确率、精确率、召回率、F1分数
           - 概率指标：ROC-AUC、PR-AUC
           - 业务指标：预期损失、风险覆盖率
        
        4. **稳定性测试**：
           - 数据扰动测试：评估对噪声的鲁棒性
           - 时间稳定性测试：验证跨时间段的性能
           - 样本外测试：使用全新数据验证
        """
    
    def _generate_performance_evaluation(self) -> Dict:
        """生成性能评估"""
        
        model_results = self.report_data.get('model_results', {})
        
        evaluation = {
            'overall_performance': self._evaluate_overall_performance(model_results),
            'detailed_metrics': self._calculate_detailed_metrics(model_results),
            'comparison_analysis': self._perform_comparison_analysis(model_results),
            'business_impact': self._assess_business_impact(model_results),
            'recommendations': self._generate_performance_recommendations(model_results)
        }
        
        return evaluation
    
    def _evaluate_overall_performance(self, model_results: Dict) -> Dict:
        """评估整体性能"""
        
        if not model_results:
            return {'status': 'no_data', 'message': '暂无模型结果数据'}
        
        # 计算平均性能指标
        all_metrics = []
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                all_metrics.append(results['metrics'])
        
        if not all_metrics:
            return {'status': 'no_metrics', 'message': '暂无性能指标数据'}
        
        # 计算平均值
        avg_metrics = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metric_names:
            values = [m.get(metric, 0) for m in all_metrics if metric in m]
            if values:
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 性能等级评估
        avg_f1 = avg_metrics.get('f1_score', {}).get('mean', 0)
        if avg_f1 >= 0.9:
            performance_level = '优秀'
            performance_description = '模型性能优秀，可以投入生产使用'
        elif avg_f1 >= 0.8:
            performance_level = '良好'
            performance_description = '模型性能良好，建议进一步优化后使用'
        elif avg_f1 >= 0.7:
            performance_level = '中等'
            performance_description = '模型性能中等，需要显著改进'
        else:
            performance_level = '较差'
            performance_description = '模型性能较差，需要重新设计'
        
        return {
            'status': 'success',
            'performance_level': performance_level,
            'description': performance_description,
            'avg_metrics': avg_metrics,
            'model_count': len(model_results)
        }
    
    def _calculate_detailed_metrics(self, model_results: Dict) -> Dict:
        """计算详细指标"""
        
        detailed_metrics = {}
        
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                
                # 基础指标
                basic_metrics = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'roc_auc': metrics.get('roc_auc', 0)
                }
                
                # 扩展指标
                extended_metrics = {
                    'specificity': metrics.get('specificity', 0),
                    'npv': metrics.get('npv', 0),  # Negative Predictive Value
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                    'matthews_corrcoef': metrics.get('matthews_corrcoef', 0)
                }
                
                # 业务指标
                business_metrics = {
                    'expected_loss': metrics.get('expected_loss', 0),
                    'risk_coverage': metrics.get('risk_coverage', 0),
                    'false_positive_rate': metrics.get('false_positive_rate', 0),
                    'false_negative_rate': metrics.get('false_negative_rate', 0)
                }
                
                detailed_metrics[model_name] = {
                    'basic_metrics': basic_metrics,
                    'extended_metrics': extended_metrics,
                    'business_metrics': business_metrics,
                    'overall_score': self._calculate_overall_score(basic_metrics)
                }
        
        return detailed_metrics
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        
        # 权重设置
        weights = {
            'accuracy': 0.2,
            'precision': 0.25,
            'recall': 0.25,
            'f1_score': 0.3
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0
    
    def _perform_comparison_analysis(self, model_results: Dict) -> Dict:
        """执行对比分析"""
        
        if len(model_results) < 2:
            return {'message': '需要至少2个模型进行对比分析'}
        
        comparison = {
            'model_ranking': [],
            'strength_analysis': {},
            'weakness_analysis': {},
            'recommendation': ''
        }
        
        # 模型排名
        model_scores = []
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                overall_score = self._calculate_overall_score(metrics)
                model_scores.append((model_name, overall_score, metrics))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score, metrics) in enumerate(model_scores, 1):
            comparison['model_ranking'].append({
                'rank': i,
                'model': model_name,
                'overall_score': score,
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0)
            })
        
        # 优势分析
        if model_scores:
            best_model = model_scores[0]
            comparison['strength_analysis'] = {
                'best_model': best_model[0],
                'best_score': best_model[1],
                'advantages': self._identify_model_advantages(best_model[2])
            }
        
        # 劣势分析
        if len(model_scores) > 1:
            worst_model = model_scores[-1]
            comparison['weakness_analysis'] = {
                'worst_model': worst_model[0],
                'worst_score': worst_model[1],
                'weaknesses': self._identify_model_weaknesses(worst_model[2])
            }
        
        # 推荐
        if model_scores:
            comparison['recommendation'] = f"推荐使用{best_model[0]}模型，其综合评分为{best_model[1]:.3f}"
        
        return comparison
    
    def _identify_model_advantages(self, metrics: Dict) -> List[str]:
        """识别模型优势"""
        
        advantages = []
        
        if metrics.get('accuracy', 0) > 0.85:
            advantages.append(f"准确率优秀({metrics['accuracy']:.3f})")
        
        if metrics.get('precision', 0) > 0.85:
            advantages.append(f"精确率高({metrics['precision']:.3f})，误报率低")
        
        if metrics.get('recall', 0) > 0.85:
            advantages.append(f"召回率高({metrics['recall']:.3f})，漏报率低")
        
        if metrics.get('f1_score', 0) > 0.85:
            advantages.append(f"F1分数优秀({metrics['f1_score']:.3f})，平衡性好")
        
        if metrics.get('roc_auc', 0) > 0.9:
            advantages.append(f"ROC-AUC优秀({metrics['roc_auc']:.3f})，区分能力强")
        
        return advantages if advantages else ["整体性能表现良好"]
    
    def _identify_model_weaknesses(self, metrics: Dict) -> List[str]:
        """识别模型劣势"""
        
        weaknesses = []
        
        if metrics.get('accuracy', 0) < 0.7:
            weaknesses.append(f"准确率偏低({metrics['accuracy']:.3f})")
        
        if metrics.get('precision', 0) < 0.7:
            weaknesses.append(f"精确率不足({metrics['precision']:.3f})，误报率较高")
        
        if metrics.get('recall', 0) < 0.7:
            weaknesses.append(f"召回率不足({metrics['recall']:.3f})，漏报率较高")
        
        if metrics.get('f1_score', 0) < 0.7:
            weaknesses.append(f"F1分数偏低({metrics['f1_score']:.3f})，需要改进")
        
        if metrics.get('roc_auc', 0) < 0.8:
            weaknesses.append(f"ROC-AUC偏低({metrics['roc_auc']:.3f})，区分能力不足")
        
        return weaknesses if weaknesses else ["性能有待提升"]
    
    def _assess_business_impact(self, model_results: Dict) -> Dict:
        """评估业务影响"""
        
        business_metrics = self.report_data.get('business_metrics', {})
        
        impact_assessment = {
            'risk_reduction': self._calculate_risk_reduction(),
            'cost_savings': self._calculate_cost_savings(),
            'efficiency_improvement': self._calculate_efficiency_improvement(),
            'decision_quality': self._assess_decision_quality(model_results),
            'roi_estimation': self._estimate_roi()
        }
        
        return impact_assessment
    
    def _calculate_risk_reduction(self) -> Dict:
        """计算风险降低"""
        
        # 模拟风险降低计算
        baseline_default_rate = 0.05  # 基线违约率5%
        model_predicted_reduction = 0.02  # 模型预测可降低2%
        
        return {
            'baseline_default_rate': baseline_default_rate,
            'predicted_reduction': model_predicted_reduction,
            'new_default_rate': baseline_default_rate - model_predicted_reduction,
            'reduction_percentage': (model_predicted_reduction / baseline_default_rate) * 100,
            'description': f"预计可将违约率从{baseline_default_rate*100:.1f}%降低到{(baseline_default_rate-model_predicted_reduction)*100:.1f}%"
        }
    
    def _calculate_cost_savings(self) -> Dict:
        """计算成本节约"""
        
        # 模拟成本节约计算
        annual_loan_volume = 1000000000  # 年放贷额10亿
        risk_reduction_rate = 0.02  # 风险降低2%
        cost_per_default = 0.6  # 每笔违约损失60%
        
        annual_savings = annual_loan_volume * risk_reduction_rate * cost_per_default
        
        return {
            'annual_loan_volume': annual_loan_volume,
            'risk_reduction_rate': risk_reduction_rate,
            'cost_per_default': cost_per_default,
            'annual_savings': annual_savings,
            'description': f"预计年节约成本{annual_savings/10000:.0f}万元"
        }
    
    def _calculate_efficiency_improvement(self) -> Dict:
        """计算效率提升"""
        
        return {
            'processing_time_reduction': 0.8,  # 处理时间减少80%
            'manual_review_reduction': 0.6,   # 人工审核减少60%
            'throughput_increase': 3.0,       # 吞吐量提升3倍
            'description': "自动化风险评估显著提升处理效率"
        }
    
    def _assess_decision_quality(self, model_results: Dict) -> Dict:
        """评估决策质量"""
        
        # 基于模型性能评估决策质量
        if not model_results:
            return {'quality_score': 0, 'description': '无法评估决策质量'}
        
        # 计算平均F1分数作为决策质量指标
        f1_scores = []
        for results in model_results.values():
            if isinstance(results, dict) and 'metrics' in results:
                f1_scores.append(results['metrics'].get('f1_score', 0))
        
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            quality_score = avg_f1 * 100  # 转换为百分制
            
            if quality_score >= 90:
                quality_level = "优秀"
            elif quality_score >= 80:
                quality_level = "良好"
            elif quality_score >= 70:
                quality_level = "中等"
            else:
                quality_level = "需要改进"
            
            return {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'description': f"决策质量{quality_level}，评分{quality_score:.1f}分"
            }
        
        return {'quality_score': 0, 'description': '无法评估决策质量'}
    
    def _estimate_roi(self) -> Dict:
        """估算投资回报率"""
        
        # 模拟ROI计算
        development_cost = 500000      # 开发成本50万
        annual_maintenance = 100000    # 年维护成本10万
        annual_savings = 12000000      # 年节约1200万（来自成本节约计算）
        
        # 3年ROI计算
        total_cost_3years = development_cost + annual_maintenance * 3
        total_savings_3years = annual_savings * 3
        roi_3years = ((total_savings_3years - total_cost_3years) / total_cost_3years) * 100
        
        return {
            'development_cost': development_cost,
            'annual_maintenance': annual_maintenance,
            'annual_savings': annual_savings,
            'roi_3years': roi_3years,
            'payback_period_months': (development_cost / (annual_savings / 12)),
            'description': f"预计3年ROI为{roi_3years:.0f}%，回本周期{development_cost/(annual_savings/12):.1f}个月"
        }
    
    def _generate_performance_recommendations(self, model_results: Dict) -> List[str]:
        """生成性能建议"""
        
        recommendations = []
        
        if not model_results:
            recommendations.append("建议收集更多模型训练和评估数据")
            return recommendations
        
        # 分析整体性能水平
        f1_scores = []
        accuracy_scores = []
        
        for results in model_results.values():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                f1_scores.append(metrics.get('f1_score', 0))
                accuracy_scores.append(metrics.get('accuracy', 0))
        
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            avg_accuracy = np.mean(accuracy_scores)
            
            if avg_f1 < 0.7:
                recommendations.append("模型整体性能偏低，建议重新进行特征工程和算法选择")
            
            if avg_accuracy < 0.8:
                recommendations.append("准确率有待提升，建议增加训练数据或调整模型参数")
            
            if len(set(f1_scores)) > 1:
                f1_std = np.std(f1_scores)
                if f1_std > 0.1:
                    recommendations.append("不同模型性能差异较大，建议进行模型集成以提升稳定性")
            
            if avg_f1 > 0.85:
                recommendations.append("模型性能优秀，建议进行生产环境部署测试")
        
        # 添加通用建议
        recommendations.extend([
            "建议定期监控模型性能，及时发现性能衰减",
            "建议建立A/B测试机制，持续优化模型效果",
            "建议加强模型可解释性分析，提升业务理解度"
        ])
        
        return recommendations
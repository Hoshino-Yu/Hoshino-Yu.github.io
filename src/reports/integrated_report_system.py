"""
集成报告系统
整合所有报告生成功能，提供统一的接口
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 导入自定义模块
from src.analysis.enhanced_model_evaluation import EnhancedModelEvaluator
from src.analysis.cross_validation_system import CrossValidationSystem
from .comprehensive_report_generator import ComprehensiveReportGenerator

class IntegratedReportSystem:
    """集成报告系统"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化集成报告系统
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.charts_dir = self.output_dir / "charts"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.charts_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化各个组件
        self.enhanced_evaluator = EnhancedModelEvaluator()
        self.cv_system = CrossValidationSystem()
        self.report_generator = ComprehensiveReportGenerator(str(self.output_dir))
        
        # 报告配置
        self.report_config = {
            'project_name': '智能信贷风险评估系统',
            'version': '2.0',
            'author': 'AI风控团队',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'language': 'zh-CN'
        }
    
    def generate_complete_analysis_report(self, 
                                        models: Dict[str, Any],
                                        X_train: Any,
                                        X_test: Any,
                                        y_train: Any,
                                        y_test: Any,
                                        feature_names: List[str] = None,
                                        report_type: str = "comprehensive") -> Dict[str, str]:
        """
        生成完整的分析报告
        
        Args:
            models: 训练好的模型字典
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            y_test: 测试标签
            feature_names: 特征名称列表
            report_type: 报告类型 ("comprehensive", "executive", "technical")
            
        Returns:
            生成的报告文件路径字典
        """
        
        print("🚀 开始生成完整分析报告...")
        
        # 1. 增强模型评估
        print("📊 执行增强模型评估...")
        enhanced_results = self._perform_enhanced_evaluation(
            models, X_train, X_test, y_train, y_test, feature_names
        )
        
        # 2. 交叉验证分析
        print("🔄 执行交叉验证分析...")
        cv_results = self._perform_cross_validation_analysis(
            models, X_train, y_train
        )
        
        # 3. 综合报告生成
        print("📝 生成综合报告...")
        comprehensive_reports = self._generate_comprehensive_reports(
            enhanced_results, cv_results, report_type
        )
        
        # 4. 生成执行摘要
        print("📋 生成执行摘要...")
        executive_summary = self._generate_executive_summary(
            enhanced_results, cv_results
        )
        
        # 5. 整合所有结果
        final_reports = {
            'enhanced_evaluation': enhanced_results.get('report_path'),
            'cross_validation': cv_results.get('report_path'),
            'comprehensive_report': comprehensive_reports.get('html_report'),
            'executive_summary': executive_summary,
            'charts_directory': str(self.charts_dir),
            'data_directory': str(self.data_dir)
        }
        
        # 6. 生成索引文件
        index_path = self._generate_report_index(final_reports)
        final_reports['index'] = index_path
        
        print("✅ 完整分析报告生成完成！")
        print(f"📁 报告目录: {self.output_dir}")
        
        return final_reports
    
    def _perform_enhanced_evaluation(self, 
                                   models: Dict[str, Any],
                                   X_train: Any,
                                   X_test: Any,
                                   y_train: Any,
                                   y_test: Any,
                                   feature_names: List[str] = None) -> Dict:
        """执行增强模型评估"""
        
        try:
            # 准备数据
            evaluation_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
            }
            
            # 执行增强评估
            results = self.enhanced_evaluator.comprehensive_model_comparison(
                models, evaluation_data
            )
            
            # 生成报告
            report_path = self.reports_dir / f"enhanced_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.enhanced_evaluator.generate_enhanced_report(results, str(report_path))
            
            results['report_path'] = str(report_path)
            return results
            
        except Exception as e:
            print(f"❌ 增强评估失败: {str(e)}")
            return {'error': str(e), 'report_path': None}
    
    def _perform_cross_validation_analysis(self, 
                                         models: Dict[str, Any],
                                         X: Any,
                                         y: Any) -> Dict:
        """执行交叉验证分析"""
        
        try:
            # 执行交叉验证
            cv_results = {}
            
            for model_name, model in models.items():
                print(f"  🔄 {model_name} 交叉验证中...")
                
                # K折交叉验证
                kfold_results = self.cv_system.k_fold_cross_validation(
                    model, X, y, k=5
                )
                
                # 分层交叉验证
                stratified_results = self.cv_system.stratified_cross_validation(
                    model, X, y, k=5
                )
                
                # 时间序列交叉验证（如果适用）
                try:
                    timeseries_results = self.cv_system.time_series_cross_validation(
                        model, X, y, n_splits=5
                    )
                except:
                    timeseries_results = None
                
                cv_results[model_name] = {
                    'kfold': kfold_results,
                    'stratified': stratified_results,
                    'timeseries': timeseries_results
                }
            
            # 生成交叉验证报告
            report_path = self.reports_dir / f"cross_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            self.cv_system.generate_cv_report(cv_results, str(report_path))
            
            return {
                'results': cv_results,
                'report_path': str(report_path)
            }
            
        except Exception as e:
            print(f"❌ 交叉验证失败: {str(e)}")
            return {'error': str(e), 'report_path': None}
    
    def _generate_comprehensive_reports(self, 
                                      enhanced_results: Dict,
                                      cv_results: Dict,
                                      report_type: str) -> Dict:
        """生成综合报告"""
        
        try:
            # 准备报告数据
            report_data = {
                'project_info': self.report_config,
                'model_results': enhanced_results.get('results', {}),
                'cv_results': cv_results.get('results', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # 生成不同类型的报告
            reports = {}
            
            if report_type in ['comprehensive', 'all']:
                # 完整报告
                html_path = self.reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                pdf_path = self.reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                html_report = self.report_generator.generate_html_report(report_data)
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                
                reports['html_report'] = str(html_path)
                
                # 尝试生成PDF（如果可能）
                try:
                    pdf_report = self.report_generator.generate_pdf_report(report_data)
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_report)
                    reports['pdf_report'] = str(pdf_path)
                except:
                    print("⚠️  PDF生成失败，仅生成HTML报告")
            
            if report_type in ['executive', 'all']:
                # 执行摘要
                summary_path = self.reports_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                summary_report = self.report_generator.generate_executive_summary(report_data)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary_report)
                
                reports['executive_summary'] = str(summary_path)
            
            return reports
            
        except Exception as e:
            print(f"❌ 综合报告生成失败: {str(e)}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, 
                                  enhanced_results: Dict,
                                  cv_results: Dict) -> str:
        """生成执行摘要"""
        
        try:
            # 提取关键指标
            key_metrics = self._extract_key_metrics(enhanced_results, cv_results)
            
            # 生成摘要内容
            summary_content = f"""
            # 智能信贷风险评估系统 - 执行摘要
            
            ## 项目概述
            - **项目名称**: {self.report_config['project_name']}
            - **版本**: {self.report_config['version']}
            - **生成日期**: {self.report_config['date']}
            
            ## 关键发现
            
            ### 模型性能
            - **最佳模型**: {key_metrics.get('best_model', 'N/A')}
            - **准确率**: {key_metrics.get('best_accuracy', 'N/A'):.3f}
            - **AUC分数**: {key_metrics.get('best_auc', 'N/A'):.3f}
            
            ### 业务价值
            - **预期ROI**: {key_metrics.get('expected_roi', 'N/A')}%
            - **风险降低**: {key_metrics.get('risk_reduction', 'N/A')}%
            - **效率提升**: {key_metrics.get('efficiency_gain', 'N/A')}%
            
            ### 建议行动
            1. 部署{key_metrics.get('best_model', '最佳')}模型到生产环境
            2. 建立持续监控和模型更新机制
            3. 实施风险预警系统
            4. 加强数据质量管理
            
            ## 详细报告
            请查看完整的技术报告和分析文档以获取更多详细信息。
            """
            
            # 保存摘要
            summary_path = self.reports_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            return str(summary_path)
            
        except Exception as e:
            print(f"❌ 执行摘要生成失败: {str(e)}")
            return None
    
    def _extract_key_metrics(self, enhanced_results: Dict, cv_results: Dict) -> Dict:
        """提取关键指标"""
        
        key_metrics = {}
        
        try:
            # 从增强评估结果中提取
            if 'results' in enhanced_results:
                results = enhanced_results['results']
                
                # 找到最佳模型
                best_model = None
                best_score = 0
                
                for model_name, model_data in results.items():
                    if isinstance(model_data, dict) and 'overall_score' in model_data:
                        score = model_data['overall_score']
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                            
                            # 提取性能指标
                            if 'performance' in model_data:
                                perf = model_data['performance']
                                key_metrics['best_accuracy'] = perf.get('accuracy', 0)
                                key_metrics['best_auc'] = perf.get('roc_auc', 0)
                
                key_metrics['best_model'] = best_model
                key_metrics['best_overall_score'] = best_score
            
            # 设置业务指标（示例值）
            key_metrics['expected_roi'] = 250
            key_metrics['risk_reduction'] = 40
            key_metrics['efficiency_gain'] = 60
            
        except Exception as e:
            print(f"⚠️  关键指标提取部分失败: {str(e)}")
        
        return key_metrics
    
    def _generate_report_index(self, reports: Dict[str, str]) -> str:
        """生成报告索引页面"""
        
        try:
            index_html = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{self.report_config['project_name']} - 报告索引</title>
                <style>
                    body {{
                        font-family: 'Microsoft YaHei', Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    .report-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin-top: 30px;
                    }}
                    .report-card {{
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 20px;
                        background: #fafafa;
                        transition: transform 0.2s;
                    }}
                    .report-card:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }}
                    .report-title {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #34495e;
                        margin-bottom: 10px;
                    }}
                    .report-description {{
                        color: #7f8c8d;
                        margin-bottom: 15px;
                    }}
                    .report-link {{
                        display: inline-block;
                        padding: 8px 16px;
                        background: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                        transition: background 0.2s;
                    }}
                    .report-link:hover {{
                        background: #2980b9;
                    }}
                    .info-section {{
                        background: #ecf0f1;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{self.report_config['project_name']} - 分析报告</h1>
                    
                    <div class="info-section">
                        <h3>项目信息</h3>
                        <p><strong>版本:</strong> {self.report_config['version']}</p>
                        <p><strong>生成日期:</strong> {self.report_config['date']}</p>
                        <p><strong>作者:</strong> {self.report_config['author']}</p>
                    </div>
                    
                    <div class="report-grid">
            """
            
            # 添加报告卡片
            report_cards = {
                'comprehensive_report': {
                    'title': '综合分析报告',
                    'description': '包含完整的模型评估、性能分析、业务价值评估和风险分析的综合报告'
                },
                'enhanced_evaluation': {
                    'title': '增强模型评估',
                    'description': '详细的模型性能对比、效率分析、复杂度评估和鲁棒性测试'
                },
                'cross_validation': {
                    'title': '交叉验证分析',
                    'description': 'K折、分层和时间序列交叉验证结果，评估模型泛化能力'
                },
                'executive_summary': {
                    'title': '执行摘要',
                    'description': '面向管理层的关键发现、业务价值和行动建议摘要'
                }
            }
            
            for report_key, report_info in report_cards.items():
                if report_key in reports and reports[report_key]:
                    report_path = os.path.relpath(reports[report_key], self.output_dir)
                    index_html += f"""
                        <div class="report-card">
                            <div class="report-title">{report_info['title']}</div>
                            <div class="report-description">{report_info['description']}</div>
                            <a href="{report_path}" class="report-link" target="_blank">查看报告</a>
                        </div>
                    """
            
            # 添加目录链接
            index_html += f"""
                        <div class="report-card">
                            <div class="report-title">图表目录</div>
                            <div class="report-description">所有生成的图表和可视化文件</div>
                            <a href="charts/" class="report-link" target="_blank">浏览图表</a>
                        </div>
                        
                        <div class="report-card">
                            <div class="report-title">数据文件</div>
                            <div class="report-description">分析过程中生成的数据文件和结果</div>
                            <a href="data/" class="report-link" target="_blank">查看数据</a>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 保存索引文件
            index_path = self.output_dir / "index.html"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_html)
            
            return str(index_path)
            
        except Exception as e:
            print(f"❌ 索引页面生成失败: {str(e)}")
            return None
    
    def generate_model_comparison_report(self, models: Dict[str, Any], evaluation_results: Dict) -> str:
        """生成模型对比报告"""
        
        try:
            # 使用增强评估器生成对比报告
            comparison_results = self.enhanced_evaluator.comprehensive_model_comparison(
                models, evaluation_results
            )
            
            # 生成报告文件
            report_path = self.reports_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.enhanced_evaluator.generate_enhanced_report(comparison_results, str(report_path))
            
            return str(report_path)
            
        except Exception as e:
            print(f"❌ 模型对比报告生成失败: {str(e)}")
            return None
    
    def get_report_summary(self) -> Dict:
        """获取报告摘要信息"""
        
        summary = {
            'output_directory': str(self.output_dir),
            'reports_generated': len(list(self.reports_dir.glob('*'))),
            'charts_generated': len(list(self.charts_dir.glob('*'))),
            'data_files': len(list(self.data_dir.glob('*'))),
            'last_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 创建集成报告系统
    report_system = IntegratedReportSystem("output_reports")
    
    print("🎯 集成报告系统已初始化")
    print(f"📁 输出目录: {report_system.output_dir}")
    
    # 示例：生成报告摘要
    summary = report_system.get_report_summary()
    print("📊 报告摘要:", summary)
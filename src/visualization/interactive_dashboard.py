"""
交互式仪表板模块
实现用户友好的界面设计、交互式可视化、动画效果和多样化图表
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入业务模块
from ..analysis.business_intelligence import BusinessIntelligenceSystem
from ..analysis.enhanced_model_evaluation import EnhancedModelEvaluator
from ..analysis.cross_validation_system import CrossValidationSystem
from ..analysis.model_interpretability import ModelInterpretabilityAnalyzer
from ..analysis.performance_monitoring import PerformanceStabilitySystem

class InteractiveDashboard:
    """交互式仪表板"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_custom_css()
        self.bi_system = BusinessIntelligenceSystem()
        
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="信贷风险管理系统",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_custom_css(self):
        """设置自定义CSS样式"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin: 0;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #666;
            margin-top: 0.5rem;
        }
        
        .status-good {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-danger {
            color: #dc3545;
            font-weight: bold;
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .animated-counter {
            animation: countUp 2s ease-out;
        }
        
        @keyframes countUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .progress-bar {
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 2s ease-in-out;
        }
        
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """运行仪表板"""
        
        # 主标题
        st.markdown("""
        <div class="main-header">
            <h1>🏦 信贷风险管理与决策支持系统</h1>
            <p>智能化风险评估 • 自动化决策支持 • 全面合规管控</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 侧边栏导航
        self.setup_sidebar()
        
        # 主内容区域
        page = st.session_state.get('current_page', 'overview')
        
        if page == 'overview':
            self.show_overview_page()
        elif page == 'risk_analysis':
            self.show_risk_analysis_page()
        elif page == 'model_evaluation':
            self.show_model_evaluation_page()
        elif page == 'business_intelligence':
            self.show_business_intelligence_page()
        elif page == 'compliance':
            self.show_compliance_page()
        elif page == 'real_time_monitoring':
            self.show_real_time_monitoring_page()
        elif page == 'settings':
            self.show_settings_page()
    
    def setup_sidebar(self):
        """设置侧边栏"""
        
        with st.sidebar:
            st.markdown("### 🧭 导航菜单")
            
            # 页面选择
            pages = {
                'overview': '📊 系统概览',
                'risk_analysis': '🎯 风险分析',
                'model_evaluation': '🤖 模型评估',
                'business_intelligence': '💼 业务智能',
                'compliance': '⚖️ 合规检查',
                'real_time_monitoring': '📈 实时监控',
                'settings': '⚙️ 系统设置'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # 快速统计
            st.markdown("### 📈 快速统计")
            
            # 模拟实时数据
            if 'dashboard_stats' not in st.session_state:
                st.session_state.dashboard_stats = self.generate_sample_stats()
            
            stats = st.session_state.dashboard_stats
            
            # 今日申请数量
            st.metric(
                label="今日申请",
                value=stats['daily_applications'],
                delta=f"+{stats['daily_applications_delta']}"
            )
            
            # 批准率
            st.metric(
                label="批准率",
                value=f"{stats['approval_rate']:.1%}",
                delta=f"{stats['approval_rate_delta']:+.1%}"
            )
            
            # 风险评分
            st.metric(
                label="平均风险评分",
                value=f"{stats['avg_risk_score']:.3f}",
                delta=f"{stats['risk_score_delta']:+.3f}"
            )
            
            st.markdown("---")
            
            # 系统状态
            st.markdown("### 🔧 系统状态")
            
            status_items = [
                ("模型服务", "🟢 正常", "status-good"),
                ("数据库", "🟢 正常", "status-good"),
                ("API服务", "🟡 警告", "status-warning"),
                ("监控系统", "🟢 正常", "status-good")
            ]
            
            for item, status, css_class in status_items:
                st.markdown(f"**{item}**: <span class='{css_class}'>{status}</span>", 
                           unsafe_allow_html=True)
            
            # 刷新按钮
            if st.button("🔄 刷新数据", use_container_width=True):
                st.session_state.dashboard_stats = self.generate_sample_stats()
                st.rerun()
    
    def generate_sample_stats(self):
        """生成示例统计数据"""
        return {
            'daily_applications': np.random.randint(50, 150),
            'daily_applications_delta': np.random.randint(-10, 20),
            'approval_rate': np.random.uniform(0.6, 0.8),
            'approval_rate_delta': np.random.uniform(-0.05, 0.05),
            'avg_risk_score': np.random.uniform(0.3, 0.7),
            'risk_score_delta': np.random.uniform(-0.05, 0.05)
        }
    
    def show_overview_page(self):
        """显示系统概览页面"""
        
        st.markdown("## 📊 系统概览")
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.create_animated_metric_card(
                "总申请数量", 
                "1,234", 
                "+12%", 
                "📝"
            )
        
        with col2:
            self.create_animated_metric_card(
                "批准率", 
                "72.5%", 
                "+2.1%", 
                "✅"
            )
        
        with col3:
            self.create_animated_metric_card(
                "平均处理时间", 
                "2.3分钟", 
                "-15%", 
                "⏱️"
            )
        
        with col4:
            self.create_animated_metric_card(
                "合规率", 
                "98.7%", 
                "+0.5%", 
                "⚖️"
            )
        
        st.markdown("---")
        
        # 图表区域
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 申请趋势")
            self.create_trend_chart()
        
        with col2:
            st.markdown("### 🎯 风险分布")
            self.create_risk_distribution_chart()
        
        # 实时活动流
        st.markdown("### 🔄 实时活动")
        self.create_activity_stream()
    
    def create_animated_metric_card(self, title, value, delta, icon):
        """创建动画指标卡片"""
        
        delta_color = "green" if delta.startswith("+") else "red" if delta.startswith("-") else "gray"
        
        st.markdown(f"""
        <div class="metric-card animated-counter">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{title}</div>
                    <div style="color: {delta_color}; font-weight: bold; margin-top: 5px;">
                        {delta}
                    </div>
                </div>
                <div style="font-size: 3rem; opacity: 0.3;">
                    {icon}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_trend_chart(self):
        """创建趋势图表"""
        
        # 生成示例数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        applications = np.random.poisson(100, len(dates))
        approvals = np.random.binomial(applications, 0.7)
        
        fig = go.Figure()
        
        # 申请数量线
        fig.add_trace(go.Scatter(
            x=dates,
            y=applications,
            mode='lines+markers',
            name='申请数量',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        # 批准数量线
        fig.add_trace(go.Scatter(
            x=dates,
            y=approvals,
            mode='lines+markers',
            name='批准数量',
            line=dict(color='#28a745', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="",
            xaxis_title="日期",
            yaxis_title="数量",
            hovermode='x unified',
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_risk_distribution_chart(self):
        """创建风险分布图表"""
        
        # 示例数据
        risk_levels = ['极低风险', '低风险', '中等风险', '高风险', '极高风险']
        counts = [45, 120, 80, 35, 15]
        colors = ['#28a745', '#20c997', '#ffc107', '#fd7e14', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_levels,
            values=counts,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_activity_stream(self):
        """创建活动流"""
        
        activities = [
            {"time": "2分钟前", "action": "新申请提交", "user": "张三", "status": "processing"},
            {"time": "5分钟前", "action": "风险评估完成", "user": "李四", "status": "completed"},
            {"time": "8分钟前", "action": "合规检查通过", "user": "王五", "status": "approved"},
            {"time": "12分钟前", "action": "人工审核", "user": "赵六", "status": "review"},
            {"time": "15分钟前", "action": "申请被拒绝", "user": "钱七", "status": "rejected"}
        ]
        
        for activity in activities:
            status_icon = {
                "processing": "🔄",
                "completed": "✅",
                "approved": "👍",
                "review": "👀",
                "rejected": "❌"
            }.get(activity["status"], "ℹ️")
            
            status_color = {
                "processing": "blue",
                "completed": "green",
                "approved": "green",
                "review": "orange",
                "rejected": "red"
            }.get(activity["status"], "gray")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 10px; border-left: 3px solid {status_color}; margin: 5px 0; background: #f8f9fa; border-radius: 5px;">
                <span style="font-size: 1.2rem; margin-right: 10px;">{status_icon}</span>
                <div style="flex: 1;">
                    <strong>{activity['action']}</strong> - {activity['user']}
                    <div style="font-size: 0.8rem; color: #666;">{activity['time']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def show_risk_analysis_page(self):
        """显示风险分析页面"""
        
        st.markdown("## 🎯 风险分析")
        
        # 风险评估工具
        with st.expander("🔍 单笔风险评估", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                loan_amount = st.number_input("贷款金额 (元)", min_value=10000, max_value=5000000, value=200000)
                annual_income = st.number_input("年收入 (元)", min_value=50000, max_value=10000000, value=600000)
                credit_score = st.slider("信用评分", min_value=300, max_value=850, value=720)
            
            with col2:
                total_debt = st.number_input("总债务 (元)", min_value=0, max_value=5000000, value=100000)
                employment_years = st.slider("工作年限", min_value=0, max_value=40, value=5)
                has_collateral = st.checkbox("是否有抵押物")
            
            if st.button("🎯 评估风险", type="primary"):
                # 模拟风险评估
                risk_score = self.calculate_risk_score(
                    loan_amount, annual_income, credit_score, 
                    total_debt, employment_years, has_collateral
                )
                
                # 显示结果
                self.display_risk_assessment_result(risk_score)
        
        # 批量风险分析
        st.markdown("### 📊 批量风险分析")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传申请数据文件 (CSV格式)", 
            type=['csv'],
            help="请上传包含贷款申请信息的CSV文件"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ 成功加载 {len(df)} 条申请记录")
                
                # 显示数据预览
                with st.expander("📋 数据预览"):
                    st.dataframe(df.head())
                
                # 批量分析按钮
                if st.button("🚀 开始批量分析"):
                    self.perform_batch_risk_analysis(df)
            
            except Exception as e:
                st.error(f"❌ 文件加载失败: {str(e)}")
    
    def calculate_risk_score(self, loan_amount, annual_income, credit_score, 
                           total_debt, employment_years, has_collateral):
        """计算风险评分"""
        
        # 简化的风险评分算法
        score = 0.5  # 基础分数
        
        # 收入负债比
        debt_to_income = (total_debt + loan_amount) / annual_income
        if debt_to_income > 0.5:
            score += 0.2
        elif debt_to_income < 0.3:
            score -= 0.1
        
        # 信用评分影响
        if credit_score < 600:
            score += 0.3
        elif credit_score > 750:
            score -= 0.2
        
        # 工作稳定性
        if employment_years < 2:
            score += 0.1
        elif employment_years > 10:
            score -= 0.1
        
        # 抵押物
        if has_collateral:
            score -= 0.15
        
        return max(0, min(1, score))
    
    def display_risk_assessment_result(self, risk_score):
        """显示风险评估结果"""
        
        # 风险等级判定
        if risk_score < 0.2:
            risk_level = "极低风险"
            risk_color = "#28a745"
            recommendation = "✅ 建议批准，可提供优惠利率"
        elif risk_score < 0.4:
            risk_level = "低风险"
            risk_color = "#20c997"
            recommendation = "✅ 建议批准，标准利率"
        elif risk_score < 0.6:
            risk_level = "中等风险"
            risk_color = "#ffc107"
            recommendation = "⚠️ 需要进一步审核，可能需要担保"
        elif risk_score < 0.8:
            risk_level = "高风险"
            risk_color = "#fd7e14"
            recommendation = "🔍 建议人工审核，需要严格风控措施"
        else:
            risk_level = "极高风险"
            risk_color = "#dc3545"
            recommendation = "❌ 建议拒绝申请"
        
        # 创建结果展示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 风险评分仪表盘
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "风险评分"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "lightcoral"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="alert-{'success' if risk_score < 0.4 else 'warning' if risk_score < 0.6 else 'danger'}">
                <h4>评估结果</h4>
                <p><strong>风险等级:</strong> <span style="color: {risk_color};">{risk_level}</span></p>
                <p><strong>风险评分:</strong> {risk_score:.3f}</p>
                <p><strong>建议:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def perform_batch_risk_analysis(self, df):
        """执行批量风险分析"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, row in df.iterrows():
            # 更新进度
            progress = (i + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"正在处理第 {i+1}/{len(df)} 条记录...")
            
            # 模拟处理时间
            time.sleep(0.1)
            
            # 计算风险评分（这里需要根据实际数据列名调整）
            risk_score = np.random.uniform(0.1, 0.9)  # 示例随机评分
            
            results.append({
                'index': i,
                'risk_score': risk_score,
                'risk_level': self.get_risk_level(risk_score)
            })
        
        # 完成处理
        progress_bar.progress(1.0)
        status_text.text("✅ 批量分析完成！")
        
        # 显示结果统计
        self.display_batch_results(results)
    
    def get_risk_level(self, risk_score):
        """获取风险等级"""
        if risk_score < 0.2:
            return "极低风险"
        elif risk_score < 0.4:
            return "低风险"
        elif risk_score < 0.6:
            return "中等风险"
        elif risk_score < 0.8:
            return "高风险"
        else:
            return "极高风险"
    
    def display_batch_results(self, results):
        """显示批量分析结果"""
        
        # 统计信息
        risk_counts = {}
        for result in results:
            level = result['risk_level']
            risk_counts[level] = risk_counts.get(level, 0) + 1
        
        # 显示统计图表
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 风险等级分布")
            
            fig = go.Figure(data=[go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=['#28a745', '#20c997', '#ffc107', '#fd7e14', '#dc3545']
            )])
            
            fig.update_layout(
                title="",
                xaxis_title="风险等级",
                yaxis_title="数量",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 风险评分分布")
            
            scores = [r['risk_score'] for r in results]
            
            fig = go.Figure(data=[go.Histogram(
                x=scores,
                nbinsx=20,
                marker_color='#667eea'
            )])
            
            fig.update_layout(
                title="",
                xaxis_title="风险评分",
                yaxis_title="频次",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 详细结果表格
        st.markdown("#### 📋 详细结果")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # 下载结果
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 下载分析结果",
            data=csv,
            file_name=f"risk_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def show_model_evaluation_page(self):
        """显示模型评估页面"""
        
        st.markdown("## 🤖 模型评估")
        
        # 模型性能对比
        st.markdown("### 📊 模型性能对比")
        
        # 模拟模型数据
        models_data = {
            'Random Forest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85, 'auc': 0.91},
            'XGBoost': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.89, 'f1': 0.86, 'auc': 0.93},
            'Logistic Regression': {'accuracy': 0.79, 'precision': 0.76, 'recall': 0.83, 'f1': 0.79, 'auc': 0.86},
            'Neural Network': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.90, 'f1': 0.87, 'auc': 0.94}
        }
        
        # 创建性能对比图表
        self.create_model_comparison_chart(models_data)
        
        # 模型详细信息
        st.markdown("### 🔍 模型详细信息")
        
        selected_model = st.selectbox("选择模型", list(models_data.keys()))
        
        if selected_model:
            self.display_model_details(selected_model, models_data[selected_model])
    
    def create_model_comparison_chart(self, models_data):
        """创建模型对比图表"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metric_names,
            specs=[[{"type": "bar"}] * len(metrics)]
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            models = list(models_data.keys())
            values = [models_data[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=name,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="模型性能对比",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_model_details(self, model_name, model_data):
        """显示模型详细信息"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 性能指标")
            
            for metric, value in model_data.items():
                metric_names = {
                    'accuracy': '准确率',
                    'precision': '精确率',
                    'recall': '召回率',
                    'f1': 'F1分数',
                    'auc': 'AUC'
                }
                
                st.metric(
                    label=metric_names.get(metric, metric),
                    value=f"{value:.3f}",
                    delta=f"{np.random.uniform(-0.02, 0.02):+.3f}"
                )
        
        with col2:
            st.markdown("#### 🎯 性能雷达图")
            
            # 创建雷达图
            categories = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
            values = list(model_data.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name,
                line_color='#667eea'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_business_intelligence_page(self):
        """显示业务智能页面"""
        
        st.markdown("## 💼 业务智能")
        
        # 业务指标概览
        st.markdown("### 📊 业务指标概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("月度放款额", "¥2.5亿", "+15.2%")
        
        with col2:
            st.metric("客户满意度", "4.8/5.0", "+0.2")
        
        with col3:
            st.metric("处理效率", "95.2%", "+3.1%")
        
        with col4:
            st.metric("风险损失率", "1.2%", "-0.3%")
        
        # 业务趋势分析
        st.markdown("### 📈 业务趋势分析")
        
        # 创建业务趋势图表
        self.create_business_trend_chart()
        
        # 客户分析
        st.markdown("### 👥 客户分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 客户价值分布")
            self.create_customer_value_chart()
        
        with col2:
            st.markdown("#### 地域分布")
            self.create_geographic_distribution_chart()
    
    def create_business_trend_chart(self):
        """创建业务趋势图表"""
        
        # 生成示例数据
        months = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        loan_amount = np.random.normal(200000000, 20000000, len(months))  # 2亿左右
        profit = loan_amount * np.random.uniform(0.05, 0.15, len(months))  # 5-15%利润率
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('月度放款额', '月度利润'),
            vertical_spacing=0.1
        )
        
        # 放款额
        fig.add_trace(
            go.Scatter(
                x=months,
                y=loan_amount,
                mode='lines+markers',
                name='放款额',
                line=dict(color='#667eea', width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # 利润
        fig.add_trace(
            go.Scatter(
                x=months,
                y=profit,
                mode='lines+markers',
                name='利润',
                line=dict(color='#28a745', width=3),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_customer_value_chart(self):
        """创建客户价值图表"""
        
        # 示例数据
        value_segments = ['高价值', '中高价值', '中等价值', '中低价值', '低价值']
        customer_counts = [150, 320, 450, 280, 100]
        colors = ['#28a745', '#20c997', '#ffc107', '#fd7e14', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=value_segments,
            values=customer_counts,
            hole=0.3,
            marker_colors=colors
        )])
        
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    def create_geographic_distribution_chart(self):
        """创建地域分布图表"""
        
        # 示例数据
        cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉']
        loan_amounts = np.random.uniform(50000000, 200000000, len(cities))
        
        fig = go.Figure(data=[go.Bar(
            x=cities,
            y=loan_amounts,
            marker_color='#667eea'
        )])
        
        fig.update_layout(
            title="",
            xaxis_title="城市",
            yaxis_title="放款金额 (元)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_compliance_page(self):
        """显示合规检查页面"""
        
        st.markdown("## ⚖️ 合规检查")
        
        # 合规状态概览
        st.markdown("### 📊 合规状态概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总体合规率", "98.7%", "+0.5%")
        
        with col2:
            st.metric("违规事件", "3", "-2")
        
        with col3:
            st.metric("待处理警告", "12", "+1")
        
        with col4:
            st.metric("合规评分", "A+", "")
        
        # 合规检查项目
        st.markdown("### 📋 合规检查项目")
        
        compliance_items = [
            {"name": "KYC验证", "status": "通过", "score": 100, "color": "green"},
            {"name": "反洗钱筛查", "status": "通过", "score": 98, "color": "green"},
            {"name": "数据隐私保护", "status": "警告", "score": 85, "color": "orange"},
            {"name": "公平放贷", "status": "通过", "score": 95, "color": "green"},
            {"name": "风险管理", "status": "通过", "score": 92, "color": "green"},
            {"name": "监管报告", "status": "通过", "score": 88, "color": "green"}
        ]
        
        for item in compliance_items:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{item['name']}**")
            
            with col2:
                status_color = {"通过": "green", "警告": "orange", "失败": "red"}.get(item['status'], "gray")
                st.markdown(f"<span style='color: {status_color};'>●</span> {item['status']}", 
                           unsafe_allow_html=True)
            
            with col3:
                st.write(f"{item['score']}%")
            
            # 进度条
            progress_color = item['color']
            st.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {item['score']}%; background-color: {progress_color};"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # 合规趋势
        st.markdown("### 📈 合规趋势")
        self.create_compliance_trend_chart()
    
    def create_compliance_trend_chart(self):
        """创建合规趋势图表"""
        
        # 生成示例数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        compliance_scores = np.random.uniform(85, 100, len(dates))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=compliance_scores,
            mode='lines+markers',
            name='合规评分',
            line=dict(color='#667eea', width=3),
            fill='tonexty'
        ))
        
        # 添加合规阈值线
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="最低合规要求")
        
        fig.update_layout(
            title="合规评分趋势",
            xaxis_title="日期",
            yaxis_title="合规评分 (%)",
            yaxis=dict(range=[80, 105]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_real_time_monitoring_page(self):
        """显示实时监控页面"""
        
        st.markdown("## 📈 实时监控")
        
        # 自动刷新控制
        auto_refresh = st.checkbox("🔄 自动刷新 (30秒)", value=True)
        
        if auto_refresh:
            # 使用 st.empty() 创建占位符，实现自动刷新
            placeholder = st.empty()
            
            # 模拟实时数据更新
            with placeholder.container():
                self.display_real_time_metrics()
        else:
            self.display_real_time_metrics()
    
    def display_real_time_metrics(self):
        """显示实时指标"""
        
        # 系统状态
        st.markdown("### 🔧 系统状态")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = np.random.uniform(20, 80)
            st.metric("CPU使用率", f"{cpu_usage:.1f}%", 
                     f"{np.random.uniform(-5, 5):+.1f}%")
        
        with col2:
            memory_usage = np.random.uniform(40, 90)
            st.metric("内存使用率", f"{memory_usage:.1f}%", 
                     f"{np.random.uniform(-3, 3):+.1f}%")
        
        with col3:
            response_time = np.random.uniform(100, 500)
            st.metric("响应时间", f"{response_time:.0f}ms", 
                     f"{np.random.uniform(-50, 50):+.0f}ms")
        
        with col4:
            throughput = np.random.uniform(800, 1200)
            st.metric("吞吐量", f"{throughput:.0f}/min", 
                     f"{np.random.uniform(-100, 100):+.0f}/min")
        
        # 实时图表
        st.markdown("### 📊 实时数据")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 申请处理量")
            self.create_real_time_processing_chart()
        
        with col2:
            st.markdown("#### 系统负载")
            self.create_real_time_load_chart()
        
        # 告警信息
        st.markdown("### 🚨 系统告警")
        
        alerts = [
            {"level": "warning", "message": "API响应时间超过阈值", "time": "2分钟前"},
            {"level": "info", "message": "定时任务执行完成", "time": "5分钟前"},
            {"level": "error", "message": "数据库连接异常", "time": "10分钟前"}
        ]
        
        for alert in alerts:
            alert_color = {"error": "red", "warning": "orange", "info": "blue"}.get(alert['level'], "gray")
            alert_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(alert['level'], "ℹ️")
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {alert_color}; margin: 5px 0; background: #f8f9fa;">
                {alert_icon} <strong>{alert['message']}</strong>
                <div style="font-size: 0.8rem; color: #666;">{alert['time']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    def create_real_time_processing_chart(self):
        """创建实时处理量图表"""
        
        # 生成最近1小时的数据
        times = pd.date_range(end=datetime.now(), periods=60, freq='1min')
        processing_counts = np.random.poisson(20, len(times))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=processing_counts,
            mode='lines',
            name='处理量',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            title="",
            xaxis_title="时间",
            yaxis_title="处理量",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_real_time_load_chart(self):
        """创建实时负载图表"""
        
        # 生成最近1小时的数据
        times = pd.date_range(end=datetime.now(), periods=60, freq='1min')
        cpu_load = np.random.uniform(20, 80, len(times))
        memory_load = np.random.uniform(30, 70, len(times))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=cpu_load,
            mode='lines',
            name='CPU',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=memory_load,
            mode='lines',
            name='内存',
            line=dict(color='#4ecdc4', width=2)
        ))
        
        fig.update_layout(
            title="",
            xaxis_title="时间",
            yaxis_title="使用率 (%)",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_settings_page(self):
        """显示系统设置页面"""
        
        st.markdown("## ⚙️ 系统设置")
        
        # 风险阈值设置
        with st.expander("🎯 风险阈值设置", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.slider("自动批准阈值", 0.0, 1.0, 0.3, 0.01, key="auto_approve_threshold")
                st.slider("自动拒绝阈值", 0.0, 1.0, 0.8, 0.01, key="auto_reject_threshold")
            
            with col2:
                st.slider("人工审核下限", 0.0, 1.0, 0.3, 0.01, key="manual_review_lower")
                st.slider("人工审核上限", 0.0, 1.0, 0.8, 0.01, key="manual_review_upper")
        
        # 业务规则设置
        with st.expander("📋 业务规则设置"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input("最大贷款金额", min_value=100000, max_value=10000000, 
                               value=5000000, step=100000, key="max_loan_amount")
                st.number_input("最低信用评分", min_value=300, max_value=850, 
                               value=600, step=10, key="min_credit_score")
            
            with col2:
                st.slider("最大债务收入比", 0.0, 1.0, 0.43, 0.01, key="max_debt_ratio")
                st.slider("最大贷款收入比", 0.0, 1.0, 0.5, 0.01, key="max_loan_ratio")
        
        # 系统配置
        with st.expander("🔧 系统配置"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], 
                           index=1, key="log_level")
                st.number_input("会话超时时间(分钟)", min_value=5, max_value=120, 
                               value=30, key="session_timeout")
            
            with col2:
                st.checkbox("启用自动备份", value=True, key="auto_backup")
                st.checkbox("启用邮件通知", value=True, key="email_notification")
        
        # 保存设置按钮
        if st.button("💾 保存设置", type="primary"):
            st.success("✅ 设置已保存！")
        
        # 系统信息
        st.markdown("### ℹ️ 系统信息")
        
        system_info = {
            "系统版本": "v2.1.0",
            "数据库版本": "PostgreSQL 13.4",
            "Python版本": "3.9.7",
            "最后更新": "2024-01-15 10:30:00",
            "运行时间": "15天 8小时 32分钟"
        }
        
        for key, value in system_info.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)

# 主函数
def main():
    """主函数"""
    dashboard = InteractiveDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
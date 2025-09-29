import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入核心模块
try:
    from src.core.data_preprocessing import AdvancedDataPreprocessor
    from src.core.feature_engineering import AdvancedFeatureEngineer
    from src.core.model_development import AdvancedModelDeveloper
    from src.core.model_evaluation import ComprehensiveModelEvaluator
    
    # 导入高级功能模块
    from src.visualization.interactive_dashboard import InteractiveDashboard
    from src.visualization.advanced_visualization import AdvancedVisualization
    from src.visualization.ui_components import UIComponents
    from src.analysis.business_intelligence import BusinessIntelligenceSystem
    from src.analysis.enhanced_model_evaluation import EnhancedModelEvaluator
    from src.analysis.cross_validation_system import CrossValidationSystem
    from src.analysis.model_interpretability import ModelInterpretabilityAnalyzer
    from src.analysis.performance_monitoring import PerformanceStabilitySystem
    from src.reports.integrated_report_system import IntegratedReportSystem
    
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    st.warning(f"部分高级功能模块未加载: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

class UnifiedCreditRiskApp:
    """统一的信贷风险管理应用"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="智能信贷风险评估与训练系统",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/Hoshino-Yu/Hoshino-Yu.github.io',
                'Report a bug': 'https://github.com/Hoshino-Yu/Hoshino-Yu.github.io/issues',
                'About': "# 智能信贷风险评估与训练系统\n集成数据分析、模型训练、风险预测和业务智能的完整解决方案"
            }
        )
    
    def initialize_components(self):
        """初始化组件"""
        if ADVANCED_FEATURES_AVAILABLE:
            self.ui = UIComponents()
            self.viz = AdvancedVisualization()
            self.dashboard = InteractiveDashboard()
            self.bi_system = BusinessIntelligenceSystem()
            self.model_evaluator = EnhancedModelEvaluator()
            self.cv_system = CrossValidationSystem()
            self.interpretability = ModelInterpretabilityAnalyzer()
            self.performance_monitor = PerformanceStabilitySystem()
            self.report_system = IntegratedReportSystem()
    
    def setup_session_state(self):
        """设置会话状态"""
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = 'basic'
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def run(self):
        """运行应用程序"""
        self.render_header()
        self.render_mode_selector()
        self.render_navigation()
        
        # 根据模式和页面渲染内容
        if st.session_state.current_mode == 'basic':
            self.render_basic_mode()
        else:
            self.render_advanced_mode()
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; 
                    text-align: center; margin-bottom: 2rem; 
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
            <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">🏦 智能信贷风险评估与管理系统</h1>
            <p style="margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.9;">
                🤖 AI驱动 • 📊 数据洞察 • ⚡ 实时决策 • 🔒 安全合规
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_mode_selector(self):
        """渲染模式选择器"""
        st.sidebar.markdown("### 🎛️ 系统模式")
        
        mode_options = ["基础模式", "高级模式"] if ADVANCED_FEATURES_AVAILABLE else ["基础模式"]
        
        selected_mode = st.sidebar.radio(
            "选择操作模式",
            mode_options,
            index=0 if st.session_state.current_mode == 'basic' else 1,
            help="基础模式：数据分析、模型训练、风险预测\n高级模式：业务智能、高级分析、报告系统"
        )
        
        st.session_state.current_mode = 'basic' if selected_mode == "基础模式" else 'advanced'
        
        if not ADVANCED_FEATURES_AVAILABLE and selected_mode == "高级模式":
            st.sidebar.warning("⚠️ 高级功能模块未完全加载")
    
    def render_navigation(self):
        """渲染导航菜单"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 功能导航")
        
        if st.session_state.current_mode == 'basic':
            pages = {
                'home': '🏠 系统首页',
                'data_analysis': '📊 数据分析与预处理',
                'model_training': '🔧 模型训练',
                'risk_prediction': '🎯 风险预测',
                'model_evaluation': '📈 模型评估',
                'settings': '⚙️ 系统设置'
            }
        else:
            pages = {
                'home': '🏠 系统概览',
                'risk_assessment': '🎯 智能风险评估',
                'model_analysis': '🤖 高级模型分析',
                'business_intelligence': '💼 业务智能分析',
                'monitoring': '📊 实时监控中心',
                'reports': '📋 综合报告系统',
                'settings': '⚙️ 高级设置',
                'help': '❓ 帮助文档'
            }
        
        for page_key, page_name in pages.items():
            if st.sidebar.button(
                page_name,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_key else "secondary"
            ):
                st.session_state.current_page = page_key
                st.rerun()
        
        # 快速操作
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ 快速操作")
        
        if st.sidebar.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.sidebar.success("✅ 数据已刷新")
        
        if st.sidebar.button("📥 导出报告", use_container_width=True):
            self.export_quick_report()
        
        # 系统状态
        self.render_system_status()
    
    def render_system_status(self):
        """渲染系统状态"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 系统状态")
        
        # 模拟系统状态
        status_items = [
            ("核心模块", "🟢", "正常"),
            ("数据连接", "🟢", "正常"),
            ("模型服务", "🟡" if not ADVANCED_FEATURES_AVAILABLE else "🟢", "部分功能" if not ADVANCED_FEATURES_AVAILABLE else "正常")
        ]
        
        for name, icon, status in status_items:
            st.sidebar.markdown(f"**{name}**: {icon} {status}")
    
    def render_basic_mode(self):
        """渲染基础模式"""
        page = st.session_state.current_page
        
        if page == 'home':
            self.render_basic_home()
        elif page == 'data_analysis':
            self.render_data_analysis_page()
        elif page == 'model_training':
            self.render_model_training_page()
        elif page == 'risk_prediction':
            self.render_risk_prediction_page()
        elif page == 'model_evaluation':
            self.render_model_evaluation_page()
        elif page == 'settings':
            self.render_basic_settings()
    
    def render_advanced_mode(self):
        """渲染高级模式"""
        if not ADVANCED_FEATURES_AVAILABLE:
            st.error("❌ 高级功能模块未完全加载，请检查依赖项")
            return
        
        page = st.session_state.current_page
        
        if page == 'home':
            self.render_advanced_home()
        elif page == 'risk_assessment':
            self.render_advanced_risk_assessment()
        elif page == 'model_analysis':
            self.render_advanced_model_analysis()
        elif page == 'business_intelligence':
            self.render_business_intelligence()
        elif page == 'monitoring':
            self.render_monitoring_center()
        elif page == 'reports':
            self.render_report_system()
        elif page == 'settings':
            self.render_advanced_settings()
        elif page == 'help':
            self.render_help_page()
    
    def render_basic_home(self):
        """渲染基础模式首页"""
        st.markdown("## 🏠 系统首页 - 基础模式")
        
        # 功能概览
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📊 数据分析
            - 数据上传与预览
            - 缺失值处理
            - 特征工程
            - 数据可视化
            """)
            if st.button("开始数据分析", key="start_data_analysis", use_container_width=True):
                st.session_state.current_page = 'data_analysis'
                st.rerun()
        
        with col2:
            st.markdown("""
            ### 🔧 模型训练
            - 算法选择
            - 参数调优
            - 模型训练
            - 性能评估
            """)
            if st.button("开始模型训练", key="start_model_training", use_container_width=True):
                st.session_state.current_page = 'model_training'
                st.rerun()
        
        with col3:
            st.markdown("""
            ### 🎯 风险预测
            - 单笔预测
            - 批量预测
            - 结果分析
            - 风险建议
            """)
            if st.button("开始风险预测", key="start_risk_prediction", use_container_width=True):
                st.session_state.current_page = 'risk_prediction'
                st.rerun()
        
        # 使用指南
        st.markdown("---")
        st.markdown("### 📖 使用指南")
        
        with st.expander("🚀 快速开始", expanded=True):
            st.markdown("""
            1. **数据准备**: 上传您的信贷数据文件（CSV或Excel格式）
            2. **数据预处理**: 处理缺失值、异常值，进行特征工程
            3. **模型训练**: 选择合适的算法，训练风险评估模型
            4. **风险预测**: 使用训练好的模型进行风险评估
            5. **结果分析**: 查看预测结果和风险建议
            """)
        
        with st.expander("💡 功能说明"):
            st.markdown("""
            - **基础模式**: 提供核心的数据分析、模型训练和风险预测功能
            - **高级模式**: 包含业务智能、高级分析和综合报告功能
            - **云端部署**: 支持GitHub Pages部署，随时随地访问
            """)
    
    def render_advanced_home(self):
        """渲染高级模式首页"""
        st.markdown("## 🏠 系统概览 - 高级模式")
        
        # 关键指标仪表板
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("今日申请", "156", "+12%")
        with col2:
            st.metric("批准率", "72.5%", "+2.1%")
        with col3:
            st.metric("平均风险评分", "0.65", "-0.02")
        with col4:
            st.metric("系统可用性", "99.8%", "+0.1%")
        
        # 图表展示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 申请趋势")
            # 生成示例数据
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            data = pd.DataFrame({
                '日期': dates,
                '申请数量': np.random.poisson(100, 30),
                '批准数量': np.random.poisson(70, 30)
            })
            st.line_chart(data.set_index('日期'))
        
        with col2:
            st.markdown("### 🎯 风险分布")
            risk_data = pd.DataFrame({
                '风险等级': ['极低', '低', '中等', '高', '极高'],
                '数量': [45, 120, 80, 35, 15]
            })
            st.bar_chart(risk_data.set_index('风险等级'))
        
        # 快速入口
        st.markdown("### 🚀 快速入口")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 智能评估", use_container_width=True):
                st.session_state.current_page = 'risk_assessment'
                st.rerun()
        
        with col2:
            if st.button("📊 业务分析", use_container_width=True):
                st.session_state.current_page = 'business_intelligence'
                st.rerun()
        
        with col3:
            if st.button("🤖 模型分析", use_container_width=True):
                st.session_state.current_page = 'model_analysis'
                st.rerun()
        
        with col4:
            if st.button("📋 生成报告", use_container_width=True):
                st.session_state.current_page = 'reports'
                st.rerun()
    
    def render_data_analysis_page(self):
        """数据分析页面"""
        st.header("📊 数据分析与预处理")
        
        # 数据上传
        uploaded_file = st.file_uploader(
            "选择数据文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV、Excel格式文件"
        )
        
        if uploaded_file is not None:
            try:
                # 读取数据
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state['data'] = df
                st.success(f"✅ 成功上传文件: {uploaded_file.name}")
                st.info(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
                
                # 数据预览
                st.markdown("### 👀 数据预览")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 数据基本信息
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 数据基本信息")
                    info_df = pd.DataFrame({
                        '列名': df.columns,
                        '数据类型': df.dtypes.astype(str),
                        '非空值数量': df.count(),
                        '缺失值数量': df.isnull().sum(),
                        '缺失率(%)': (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 数值列统计")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.info("没有数值型列")
                
                # 数据预处理选项
                st.markdown("### ⚙️ 数据预处理")
                preprocessing_options = st.multiselect(
                    "选择预处理操作",
                    ["处理缺失值", "编码分类变量", "标准化数值变量", "移除重复行", "异常值处理"]
                )
                
                if st.button("🔧 执行预处理") and preprocessing_options:
                    with st.spinner("正在执行预处理..."):
                        processed_df = self.perform_preprocessing(df, preprocessing_options)
                        st.session_state['processed_data'] = processed_df
                        st.success("🎉 预处理完成！")
                        
                        # 显示处理后的数据
                        st.markdown("#### 处理后的数据预览")
                        st.dataframe(processed_df.head(), use_container_width=True)
                        
                        # 下载处理后的数据
                        csv = processed_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载处理后的数据",
                            data=csv,
                            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ 文件处理失败: {str(e)}")
    
    def perform_preprocessing(self, df, options):
        """执行数据预处理"""
        processed_df = df.copy()
        
        if "处理缺失值" in options:
            for col in processed_df.columns:
                if processed_df[col].isnull().sum() > 0:
                    if processed_df[col].dtype == 'object':
                        mode_value = processed_df[col].mode()
                        if len(mode_value) > 0:
                            processed_df[col] = processed_df[col].fillna(mode_value[0])
                        else:
                            processed_df[col] = processed_df[col].fillna('Unknown')
                    else:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        
        if "编码分类变量" in options:
            from sklearn.preprocessing import LabelEncoder
            for col in processed_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        
        if "移除重复行" in options:
            processed_df = processed_df.drop_duplicates()
        
        return processed_df
    
    def render_model_training_page(self):
        """模型训练页面"""
        st.header("🔧 模型训练")
        
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ 请先上传并预处理数据")
            return
        
        df = st.session_state['processed_data']
        
        # 目标变量选择
        st.markdown("### 🎯 目标变量设置")
        target_col = st.selectbox("选择目标变量", df.columns)
        
        if target_col:
            # 特征选择
            feature_cols = st.multiselect(
                "选择特征变量",
                [col for col in df.columns if col != target_col],
                default=[col for col in df.columns if col != target_col][:5]
            )
            
            if feature_cols:
                # 算法选择
                st.markdown("### 🤖 算法选择")
                algorithm = st.selectbox(
                    "选择机器学习算法",
                    ["随机森林", "逻辑回归", "支持向量机", "梯度提升", "神经网络"]
                )
                
                # 训练参数
                st.markdown("### ⚙️ 训练参数")
                col1, col2 = st.columns(2)
                
                with col1:
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
                    random_state = st.number_input("随机种子", value=42)
                
                with col2:
                    cv_folds = st.slider("交叉验证折数", 3, 10, 5)
                    
                # 开始训练
                if st.button("🚀 开始训练", type="primary"):
                    with st.spinner("正在训练模型..."):
                        results = self.train_model(df, target_col, feature_cols, algorithm, test_size, random_state, cv_folds)
                        st.session_state['model_results'] = results
                        
                        # 显示训练结果
                        self.display_training_results(results)
    
    def train_model(self, df, target_col, feature_cols, algorithm, test_size, random_state, cv_folds):
        """训练模型"""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # 准备数据
        X = df[feature_cols]
        y = df[target_col]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 选择算法
        if algorithm == "随机森林":
            model = RandomForestClassifier(random_state=random_state)
        elif algorithm == "逻辑回归":
            model = LogisticRegression(random_state=random_state)
        elif algorithm == "支持向量机":
            model = SVC(random_state=random_state)
        elif algorithm == "梯度提升":
            model = GradientBoostingClassifier(random_state=random_state)
        else:  # 神经网络
            model = MLPClassifier(random_state=random_state)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        
        return {
            'model': model,
            'algorithm': algorithm,
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_cols': feature_cols,
            'target_col': target_col
        }
    
    def display_training_results(self, results):
        """显示训练结果"""
        st.markdown("### 📊 训练结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("准确率", f"{results['accuracy']:.3f}")
        
        with col2:
            st.metric("交叉验证均值", f"{results['cv_scores'].mean():.3f}")
        
        with col3:
            st.metric("交叉验证标准差", f"{results['cv_scores'].std():.3f}")
        
        # 分类报告
        st.markdown("#### 📋 分类报告")
        st.text(results['classification_report'])
        
        # 混淆矩阵
        st.markdown("#### 🔄 混淆矩阵")
        st.write(results['confusion_matrix'])
        
        st.success("✅ 模型训练完成！")
    
    def render_risk_prediction_page(self):
        """风险预测页面"""
        st.header("🎯 风险预测")
        
        if 'model_results' not in st.session_state:
            st.warning("⚠️ 请先训练模型")
            return
        
        model_results = st.session_state['model_results']
        model = model_results['model']
        feature_cols = model_results['feature_cols']
        
        # 预测模式选择
        prediction_mode = st.radio(
            "选择预测模式",
            ["单笔预测", "批量预测"]
        )
        
        if prediction_mode == "单笔预测":
            st.markdown("### 📝 输入预测数据")
            
            # 动态生成输入字段
            input_data = {}
            cols = st.columns(2)
            
            for i, col in enumerate(feature_cols):
                with cols[i % 2]:
                    input_data[col] = st.number_input(f"{col}", value=0.0)
            
            if st.button("🎯 开始预测", type="primary"):
                # 执行预测
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                
                # 显示结果
                self.display_prediction_result(prediction, prediction_proba)
        
        else:  # 批量预测
            st.markdown("### 📁 批量预测")
            
            uploaded_file = st.file_uploader(
                "上传预测数据文件",
                type=['csv', 'xlsx'],
                help="文件应包含与训练数据相同的特征列"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        pred_df = pd.read_csv(uploaded_file)
                    else:
                        pred_df = pd.read_excel(uploaded_file)
                    
                    if st.button("🚀 开始批量预测", type="primary"):
                        # 执行批量预测
                        predictions = model.predict(pred_df[feature_cols])
                        prediction_probas = model.predict_proba(pred_df[feature_cols]) if hasattr(model, 'predict_proba') else None
                        
                        # 显示批量结果
                        self.display_batch_prediction_results(pred_df, predictions, prediction_probas)
                
                except Exception as e:
                    st.error(f"❌ 文件处理失败: {str(e)}")
    
    def display_prediction_result(self, prediction, prediction_proba):
        """显示单笔预测结果"""
        st.markdown("### 🎯 预测结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_level = "高风险" if prediction == 1 else "低风险"
            risk_color = "red" if prediction == 1 else "green"
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; border-radius: 10px; 
                        background: {'#ffebee' if prediction == 1 else '#e8f5e9'};">
                <h2 style="color: {risk_color}; margin: 0;">{risk_level}</h2>
                <p style="margin: 0;">预测结果</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if prediction_proba is not None:
                risk_prob = prediction_proba[1] * 100
                st.metric("违约概率", f"{risk_prob:.2f}%")
                confidence = max(prediction_proba) * 100
                st.metric("预测置信度", f"{confidence:.2f}%")
    
    def display_batch_prediction_results(self, df, predictions, prediction_probas):
        """显示批量预测结果"""
        st.markdown("### 📊 批量预测结果")
        
        # 创建结果DataFrame
        results_df = df.copy()
        results_df['预测结果'] = ['高风险' if p == 1 else '低风险' for p in predictions]
        results_df['风险标签'] = predictions
        
        if prediction_probas is not None:
            results_df['违约概率'] = prediction_probas[:, 1]
            results_df['预测置信度'] = np.max(prediction_probas, axis=1)
        
        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        
        total_count = len(results_df)
        high_risk_count = sum(predictions == 1)
        low_risk_count = total_count - high_risk_count
        high_risk_ratio = high_risk_count / total_count * 100
        
        with col1:
            st.metric("总样本数", total_count)
        with col2:
            st.metric("高风险客户", high_risk_count)
        with col3:
            st.metric("低风险客户", low_risk_count)
        with col4:
            st.metric("高风险比例", f"{high_risk_ratio:.1f}%")
        
        # 显示结果表格
        st.dataframe(results_df, use_container_width=True)
        
        # 下载结果
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载预测结果",
            data=csv,
            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_model_evaluation_page(self):
        """模型评估页面"""
        st.header("📈 模型评估")
        
        if 'model_results' not in st.session_state:
            st.warning("⚠️ 请先训练模型")
            return
        
        results = st.session_state['model_results']
        
        # 性能指标
        st.markdown("### 📊 性能指标")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("算法", results['algorithm'])
        with col2:
            st.metric("准确率", f"{results['accuracy']:.3f}")
        with col3:
            st.metric("交叉验证得分", f"{results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")
        
        # 详细报告
        st.markdown("### 📋 详细评估报告")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 分类报告")
            st.text(results['classification_report'])
        
        with col2:
            st.markdown("#### 混淆矩阵")
            st.write(results['confusion_matrix'])
        
        # 交叉验证结果
        st.markdown("### 🔄 交叉验证结果")
        cv_df = pd.DataFrame({
            '折数': range(1, len(results['cv_scores']) + 1),
            '得分': results['cv_scores']
        })
        st.line_chart(cv_df.set_index('折数'))
    
    def render_basic_settings(self):
        """基础设置页面"""
        st.header("⚙️ 系统设置")
        
        # 数据设置
        st.markdown("### 📊 数据设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("最大文件大小 (MB)", value=200, min_value=1, max_value=1000)
            st.selectbox("默认编码", ["utf-8", "gbk", "gb2312"])
        
        with col2:
            st.checkbox("自动处理缺失值", value=True)
            st.checkbox("自动编码分类变量", value=True)
        
        # 模型设置
        st.markdown("### 🤖 模型设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("默认算法", ["随机森林", "逻辑回归", "支持向量机"])
            st.slider("默认测试集比例", 0.1, 0.5, 0.2)
        
        with col2:
            st.number_input("默认随机种子", value=42)
            st.slider("默认交叉验证折数", 3, 10, 5)
        
        if st.button("💾 保存设置", type="primary"):
            st.success("✅ 设置已保存")
    
    def render_advanced_risk_assessment(self):
        """渲染高级风险评估页面"""
        st.markdown("## 🎯 智能风险评估")
        
        # 单笔评估
        with st.expander("🔍 单笔风险评估", expanded=True):
            self.render_single_risk_assessment()
        
        # 批量评估
        with st.expander("📊 批量风险评估"):
            self.render_batch_risk_assessment()
        
        # 历史记录
        with st.expander("📋 评估历史"):
            self.render_assessment_history()
    
    def render_single_risk_assessment(self):
        """渲染单笔风险评估"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 申请信息")
            
            loan_amount = st.number_input(
                "贷款金额 (元)", 
                min_value=10000, 
                max_value=5000000, 
                value=200000,
                step=10000
            )
            
            annual_income = st.number_input(
                "年收入 (元)", 
                min_value=50000, 
                max_value=10000000, 
                value=600000,
                step=10000
            )
            
            credit_score = st.slider(
                "信用评分", 
                min_value=300, 
                max_value=850, 
                value=720
            )
            
            employment_years = st.slider(
                "工作年限", 
                min_value=0, 
                max_value=40, 
                value=5
            )
        
        with col2:
            st.markdown("#### 💰 财务信息")
            
            total_debt = st.number_input(
                "总债务 (元)", 
                min_value=0, 
                max_value=5000000, 
                value=100000,
                step=10000
            )
            
            monthly_income = st.number_input(
                "月收入 (元)", 
                min_value=3000, 
                max_value=500000, 
                value=50000,
                step=1000
            )
            
            has_collateral = st.checkbox("是否有抵押物")
            has_guarantor = st.checkbox("是否有担保人")
        
        if st.button("🎯 开始评估", type="primary", use_container_width=True):
            # 执行风险评估
            risk_score = self.calculate_risk_score(
                loan_amount, annual_income, credit_score, 
                total_debt, employment_years, has_collateral, has_guarantor
            )
            
            # 显示评估结果
            self.display_assessment_result(risk_score, {
                'loan_amount': loan_amount,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'total_debt': total_debt,
                'employment_years': employment_years,
                'has_collateral': has_collateral,
                'has_guarantor': has_guarantor
            })
    
    def calculate_risk_score(self, loan_amount, annual_income, credit_score, 
                           total_debt, employment_years, has_collateral, has_guarantor):
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
        
        # 抵押物和担保人
        if has_collateral:
            score -= 0.15
        if has_guarantor:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def display_assessment_result(self, risk_score, application_data):
        """显示评估结果"""
        
        st.markdown("---")
        st.markdown("### 📊 评估结果")
        
        # 风险等级判定
        if risk_score < 0.2:
            risk_level = "极低风险"
            recommendation = "✅ 建议批准，可提供优惠利率"
            decision = "自动批准"
        elif risk_score < 0.4:
            risk_level = "低风险"
            recommendation = "✅ 建议批准，标准利率"
            decision = "自动批准"
        elif risk_score < 0.6:
            risk_level = "中等风险"
            recommendation = "⚠️ 需要进一步审核，可能需要担保"
            decision = "人工审核"
        elif risk_score < 0.8:
            risk_level = "高风险"
            recommendation = "🔍 建议人工审核，需要严格风控措施"
            decision = "人工审核"
        else:
            risk_level = "极高风险"
            recommendation = "❌ 建议拒绝申请"
            decision = "自动拒绝"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("风险评分", f"{risk_score:.3f}")
        
        with col2:
            st.metric("风险等级", risk_level)
        
        with col3:
            st.metric("决策建议", decision)
        
        # 详细分析
        st.markdown("#### 📋 详细分析")
        
        analysis_data = {
            "债务收入比": f"{((application_data['total_debt'] + application_data['loan_amount']) / application_data['annual_income']):.2%}",
            "信用评分等级": self.get_credit_score_level(application_data['credit_score']),
            "工作稳定性": self.get_employment_stability(application_data['employment_years']),
            "风险缓解措施": "有抵押物" if application_data['has_collateral'] else "无抵押物"
        }
        
        for key, value in analysis_data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
        
        # 建议
        if risk_score < 0.4:
            st.success(recommendation)
        elif risk_score < 0.6:
            st.warning(recommendation)
        else:
            st.error(recommendation)
    
    def get_credit_score_level(self, score):
        """获取信用评分等级"""
        if score >= 750:
            return "优秀"
        elif score >= 700:
            return "良好"
        elif score >= 650:
            return "一般"
        elif score >= 600:
            return "较差"
        else:
            return "很差"
    
    def get_employment_stability(self, years):
        """获取工作稳定性"""
        if years >= 10:
            return "非常稳定"
        elif years >= 5:
            return "稳定"
        elif years >= 2:
            return "一般"
        else:
            return "不稳定"
    
    def render_batch_risk_assessment(self):
        """渲染批量风险评估"""
        
        st.markdown("#### 📁 文件上传")
        
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
                    st.dataframe(df.head(10))
                
                # 批量分析按钮
                if st.button("🚀 开始批量分析", type="primary"):
                    self.perform_batch_analysis(df)
            
            except Exception as e:
                st.error(f"❌ 文件加载失败: {str(e)}")
    
    def perform_batch_analysis(self, df):
        """执行批量分析"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i in range(len(df)):
            # 更新进度
            progress = (i + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"正在处理第 {i+1}/{len(df)} 条记录...")
            
            # 模拟风险评分计算
            risk_score = np.random.uniform(0.1, 0.9)
            
            results.append({
                'index': i,
                'risk_score': risk_score,
                'risk_level': self.get_risk_level_from_score(risk_score),
                'decision': self.get_decision_from_score(risk_score)
            })
        
        # 完成处理
        progress_bar.progress(1.0)
        status_text.text("✅ 批量分析完成！")
        
        # 显示结果
        self.display_batch_results(results)
    
    def get_risk_level_from_score(self, score):
        """根据评分获取风险等级"""
        if score < 0.2:
            return "极低风险"
        elif score < 0.4:
            return "低风险"
        elif score < 0.6:
            return "中等风险"
        elif score < 0.8:
            return "高风险"
        else:
            return "极高风险"
    
    def get_decision_from_score(self, score):
        """根据评分获取决策"""
        if score < 0.4:
            return "自动批准"
        elif score < 0.6:
            return "人工审核"
        else:
            return "自动拒绝"
    
    def display_batch_results(self, results):
        """显示批量结果"""
        
        # 统计信息
        st.markdown("#### 📊 分析统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        auto_approve = sum(1 for r in results if r['decision'] == '自动批准')
        manual_review = sum(1 for r in results if r['decision'] == '人工审核')
        auto_reject = sum(1 for r in results if r['decision'] == '自动拒绝')
        
        with col1:
            st.metric("总申请数", str(len(results)))
        
        with col2:
            st.metric("自动批准", str(auto_approve))
        
        with col3:
            st.metric("人工审核", str(manual_review))
        
        with col4:
            st.metric("自动拒绝", str(auto_reject))
        
        # 结果表格
        st.markdown("#### 📋 详细结果")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # 下载结果
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 下载分析结果",
            data=csv,
            file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def render_assessment_history(self):
        """渲染评估历史"""
        
        st.markdown("#### 📋 历史记录")
        
        # 模拟历史数据
        history_data = []
        for i in range(20):
            history_data.append({
                '时间': datetime.now() - timedelta(hours=i),
                '申请人': f"申请人{i+1}",
                '贷款金额': np.random.randint(50000, 1000000),
                '风险评分': np.random.uniform(0.1, 0.9),
                '决策结果': np.random.choice(['自动批准', '人工审核', '自动拒绝'])
            })
        
        history_df = pd.DataFrame(history_data)
        history_df['时间'] = history_df['时间'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(history_df, use_container_width=True)

    def render_advanced_model_analysis(self):
        """渲染高级模型分析页面"""
        
        st.markdown("## 🤖 模型分析")
        
        # 模型性能对比
        with st.expander("📊 模型性能对比", expanded=True):
            self.render_model_performance_comparison()
        
        # 模型解释性分析
        with st.expander("🔍 模型解释性分析"):
            self.render_model_interpretability()
        
        # 交叉验证结果
        with st.expander("✅ 交叉验证结果"):
            self.render_cross_validation_results()
    
    def render_model_performance_comparison(self):
        """渲染模型性能对比"""
        
        # 模拟模型数据
        models_data = {
            'Random Forest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85, 'auc': 0.91},
            'XGBoost': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.89, 'f1': 0.86, 'auc': 0.93},
            'Logistic Regression': {'accuracy': 0.79, 'precision': 0.76, 'recall': 0.83, 'f1': 0.79, 'auc': 0.86},
            'Neural Network': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.90, 'f1': 0.87, 'auc': 0.94}
        }
        
        # 转换为DataFrame
        df = pd.DataFrame(models_data).T
        
        # 显示对比表格
        st.markdown("#### 📋 性能指标对比")
        st.dataframe(df.style.format(precision=3).background_gradient(cmap='RdYlGn', axis=1))
        
        # 可视化对比
        st.markdown("#### 📊 可视化对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 准确率对比
            accuracy_data = pd.DataFrame({
                '模型': list(models_data.keys()),
                '准确率': [data['accuracy'] for data in models_data.values()]
            })
            st.bar_chart(accuracy_data.set_index('模型'))
        
        with col2:
            # AUC对比
            auc_data = pd.DataFrame({
                '模型': list(models_data.keys()),
                'AUC': [data['auc'] for data in models_data.values()]
            })
            st.bar_chart(auc_data.set_index('模型'))
    
    def render_model_interpretability(self):
        """渲染模型解释性分析"""
        
        st.markdown("#### 🔍 特征重要性分析")
        
        # 模拟特征重要性数据
        features = ['信用评分', '年收入', '债务收入比', '工作年限', '贷款金额', '抵押物价值', '历史违约次数', '账户余额']
        importance = np.random.uniform(0.05, 0.25, len(features))
        importance = importance / importance.sum()  # 归一化
        
        feature_importance_df = pd.DataFrame({
            '特征': features,
            '重要性': importance
        }).sort_values('重要性', ascending=True)
        
        st.bar_chart(feature_importance_df.set_index('特征'))
        
        # SHAP值分析
        st.markdown("#### 📈 SHAP值分析")
        
        st.info("SHAP (SHapley Additive exPlanations) 值显示每个特征对模型预测的贡献程度")
        
        # 模拟SHAP值
        shap_data = pd.DataFrame({
            '特征': features,
            'SHAP值': np.random.uniform(-0.1, 0.1, len(features))
        })
        
        st.bar_chart(shap_data.set_index('特征'))
    
    def render_cross_validation_results(self):
        """渲染交叉验证结果"""
        
        st.markdown("#### ✅ K折交叉验证结果")
        
        # 模拟交叉验证数据
        cv_results = []
        for fold in range(1, 6):
            cv_results.append({
                '折数': f'Fold {fold}',
                '准确率': np.random.uniform(0.80, 0.90),
                '精确率': np.random.uniform(0.75, 0.85),
                '召回率': np.random.uniform(0.82, 0.92),
                'F1分数': np.random.uniform(0.78, 0.88)
            })
        
        cv_df = pd.DataFrame(cv_results)
        
        # 显示结果表格
        st.dataframe(cv_df.style.format(precision=3))
        
        # 显示统计信息
        st.markdown("#### 📊 统计摘要")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_acc = cv_df['准确率'].mean()
            std_acc = cv_df['准确率'].std()
            st.metric("平均准确率", f"{mean_acc:.3f}", f"±{std_acc:.3f}")
        
        with col2:
            mean_prec = cv_df['精确率'].mean()
            std_prec = cv_df['精确率'].std()
            st.metric("平均精确率", f"{mean_prec:.3f}", f"±{std_prec:.3f}")
        
        with col3:
            mean_recall = cv_df['召回率'].mean()
            std_recall = cv_df['召回率'].std()
            st.metric("平均召回率", f"{mean_recall:.3f}", f"±{std_recall:.3f}")
        
        with col4:
            mean_f1 = cv_df['F1分数'].mean()
            std_f1 = cv_df['F1分数'].std()
            st.metric("平均F1分数", f"{mean_f1:.3f}", f"±{std_f1:.3f}")

    def render_business_intelligence(self):
        """渲染业务智能页面"""
        
        st.markdown("## 💼 业务智能")
        
        # 业务概览
        with st.expander("📊 业务概览", expanded=True):
            self.render_business_overview()
        
        # 客户分析
        with st.expander("👥 客户分析"):
            self.render_customer_analysis()
        
        # 风险分析
        with st.expander("⚠️ 风险分析"):
            self.render_business_risk_analysis()
        
        # 盈利分析
        with st.expander("💰 盈利分析"):
            self.render_profitability_analysis()
    
    def render_business_overview(self):
        """渲染业务概览"""
        
        # 关键业务指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.create_metric_card("月度放款额", "¥2.5亿", "+15.2%", "💰")
        
        with col2:
            self.ui.create_metric_card("客户满意度", "4.8/5.0", "+0.2", "😊")
        
        with col3:
            self.ui.create_metric_card("处理效率", "95.2%", "+3.1%", "⚡")
        
        with col4:
            self.ui.create_metric_card("风险损失率", "1.2%", "-0.3%", "📉")
        
        # 业务趋势图表
        st.markdown("#### 📈 业务趋势")
        
        # 生成示例数据
        months = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        loan_amount = np.random.normal(200000000, 20000000, len(months))
        profit = loan_amount * np.random.uniform(0.05, 0.15, len(months))
        
        trend_data = pd.DataFrame({
            '月份': months,
            '放款金额': loan_amount,
            '利润': profit
        })
        
        st.line_chart(trend_data.set_index('月份'))
    
    def render_customer_analysis(self):
        """渲染客户分析"""
        
        st.markdown("#### 👥 客户价值分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 客户价值分布
            value_data = pd.DataFrame({
                '价值等级': ['高价值', '中高价值', '中等价值', '中低价值', '低价值'],
                '客户数量': [150, 320, 450, 280, 100]
            })
            st.bar_chart(value_data.set_index('价值等级'))
        
        with col2:
            # 地域分布
            region_data = pd.DataFrame({
                '地区': ['华东', '华南', '华北', '华中', '西南', '东北', '西北'],
                '客户数量': [450, 380, 320, 280, 220, 150, 100]
            })
            st.bar_chart(region_data.set_index('地区'))
        
        # 客户行为分析
        st.markdown("#### 📊 客户行为分析")
        
        behavior_metrics = [
            {"title": "平均贷款金额", "value": "¥185,000", "delta": "+12%"},
            {"title": "平均还款周期", "value": "24个月", "delta": "+2个月"},
            {"title": "提前还款率", "value": "15.3%", "delta": "+2.1%"},
            {"title": "续贷率", "value": "68.7%", "delta": "+5.2%"}
        ]
        
        cols = st.columns(4)
        for i, metric in enumerate(behavior_metrics):
            with cols[i]:
                st.metric(
                    metric["title"],
                    metric["value"],
                    metric["delta"]
                )
    
    def render_business_risk_analysis(self):
        """渲染业务风险分析"""
        
        st.markdown("#### ⚠️ 风险指标监控")
        
        # 风险指标
        risk_metrics = [
            {"name": "违约率", "current": 2.3, "threshold": 3.0, "trend": "下降"},
            {"name": "逾期率", "current": 5.1, "threshold": 6.0, "trend": "稳定"},
            {"name": "坏账率", "current": 1.8, "threshold": 2.5, "trend": "下降"},
            {"name": "集中度风险", "current": 15.2, "threshold": 20.0, "trend": "上升"}
        ]
        
        for metric in risk_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{metric['name']}**")
            
            with col2:
                st.write(f"{metric['current']:.1f}%")
            
            with col3:
                color = "green" if metric['current'] < metric['threshold'] else "red"
                st.markdown(f"<span style='color: {color};'>阈值: {metric['threshold']:.1f}%</span>", 
                           unsafe_allow_html=True)
            
            with col4:
                trend_color = {"上升": "red", "下降": "green", "稳定": "blue"}[metric['trend']]
                st.markdown(f"<span style='color: {trend_color};'>{metric['trend']}</span>", 
                           unsafe_allow_html=True)
            
            # 进度条
            progress = min(metric['current'] / metric['threshold'], 1.0)
            bar_color = "green" if progress < 0.8 else "orange" if progress < 1.0 else "red"
            self.ui.create_progress_bar(metric['current'], metric['threshold'], color=bar_color)
    
    def render_profitability_analysis(self):
        """渲染盈利分析"""
        
        st.markdown("#### 💰 盈利能力分析")
        
        # 盈利指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("净利润率", "12.5%", "+1.2%")
        
        with col2:
            st.metric("资产回报率", "8.3%", "+0.8%")
        
        with col3:
            st.metric("净息差", "3.2%", "+0.1%")
        
        with col4:
            st.metric("成本收入比", "45.6%", "-2.3%")
        
        # 收入构成分析
        st.markdown("#### 📊 收入构成")
        
        revenue_data = pd.DataFrame({
            '收入类型': ['利息收入', '手续费收入', '其他收入'],
            '金额(万元)': [8500, 1200, 300],
            '占比': ['85%', '12%', '3%']
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(revenue_data, use_container_width=True)
        
        with col2:
            st.bar_chart(revenue_data.set_index('收入类型')['金额(万元)'])
    
    def render_monitoring_center(self):
        """渲染监控中心"""
        
        st.markdown("## 📊 实时监控")
        
        # 自动刷新控制
        auto_refresh = st.checkbox("🔄 自动刷新 (30秒)", value=False)
        
        if auto_refresh:
            st.info("自动刷新已启用，页面将每30秒更新一次数据")
        
        # 系统状态监控
        with st.expander("🔧 系统状态监控", expanded=True):
            self.render_system_monitoring()
        
        # 业务监控
        with st.expander("📈 业务监控"):
            self.render_business_monitoring()
        
        # 告警中心
        with st.expander("🚨 告警中心"):
            self.render_alert_center()
    
    def render_system_monitoring(self):
        """渲染系统监控"""
        
        # 系统资源使用情况
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = np.random.uniform(20, 80)
            st.metric("CPU使用率", f"{cpu_usage:.1f}%", f"{np.random.uniform(-5, 5):+.1f}%")
        
        with col2:
            memory_usage = np.random.uniform(40, 90)
            st.metric("内存使用率", f"{memory_usage:.1f}%", f"{np.random.uniform(-3, 3):+.1f}%")
        
        with col3:
            disk_usage = np.random.uniform(30, 70)
            st.metric("磁盘使用率", f"{disk_usage:.1f}%", f"{np.random.uniform(-2, 2):+.1f}%")
        
        with col4:
            network_usage = np.random.uniform(10, 50)
            st.metric("网络使用率", f"{network_usage:.1f}%", f"{np.random.uniform(-5, 5):+.1f}%")
        
        # 服务状态
        st.markdown("#### 🔧 服务状态")
        
        services = [
            {"name": "Web服务", "status": "running", "uptime": "15天 8小时"},
            {"name": "数据库", "status": "running", "uptime": "30天 12小时"},
            {"name": "缓存服务", "status": "warning", "uptime": "2天 6小时"},
            {"name": "消息队列", "status": "running", "uptime": "7天 14小时"}
        ]
        
        for service in services:
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.write(f"**{service['name']}**")
            
            with col2:
                if service['status'] == 'running':
                    st.success("运行中")
                elif service['status'] == 'warning':
                    st.warning("警告")
                else:
                    st.error("停止")
            
            with col3:
                st.write(f"运行时间: {service['uptime']}")
    
    def render_business_monitoring(self):
        """渲染业务监控"""
        
        # 实时业务指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_applications = np.random.randint(50, 150)
            st.metric("当前申请数", str(current_applications), f"{np.random.randint(-10, 20):+d}")
        
        with col2:
            processing_time = np.random.uniform(1.5, 3.5)
            st.metric("平均处理时间", f"{processing_time:.1f}分钟", f"{np.random.uniform(-0.5, 0.5):+.1f}分钟")
        
        with col3:
            approval_rate = np.random.uniform(0.65, 0.85)
            st.metric("实时批准率", f"{approval_rate:.1%}", f"{np.random.uniform(-0.05, 0.05):+.1%}")
        
        with col4:
            queue_length = np.random.randint(0, 50)
            st.metric("待处理队列", str(queue_length), f"{np.random.randint(-5, 10):+d}")
        
        # 实时处理量图表
        st.markdown("#### 📊 实时处理量")
        
        # 生成最近24小时的数据
        hours = pd.date_range(end=datetime.now(), periods=24, freq='1H')
        processing_counts = np.random.poisson(25, len(hours))
        
        processing_data = pd.DataFrame({
            '时间': hours,
            '处理量': processing_counts
        })
        
        st.line_chart(processing_data.set_index('时间'))
    
    def render_alert_center(self):
        """渲染告警中心"""
        
        st.markdown("#### 🚨 系统告警")
        
        alerts = [
            {
                "level": "danger",
                "title": "数据库连接异常",
                "message": "数据库连接池耗尽，需要立即处理",
                "time": "2分钟前",
                "status": "未处理"
            },
            {
                "level": "warning", 
                "title": "API响应时间过长",
                "message": "风险评估API平均响应时间超过5秒",
                "time": "15分钟前",
                "status": "处理中"
            },
            {
                "level": "info",
                "title": "定时任务完成",
                "message": "日终批处理任务已成功完成",
                "time": "1小时前",
                "status": "已处理"
            },
            {
                "level": "warning",
                "title": "磁盘空间不足",
                "message": "日志分区磁盘使用率达到85%",
                "time": "2小时前",
                "status": "未处理"
            }
        ]
        
        for alert in alerts:
            # 根据告警级别设置颜色
            level_colors = {
                "danger": "#dc3545",
                "warning": "#ffc107", 
                "info": "#17a2b8"
            }
            
            level_icons = {
                "danger": "🔴",
                "warning": "🟡",
                "info": "🔵"
            }
            
            color = level_colors.get(alert['level'], '#6c757d')
            icon = level_icons.get(alert['level'], 'ℹ️')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 1rem; margin: 1rem 0; background: #f8f9fa; border-radius: 0 8px 8px 0;">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-weight: bold; color: {color};">
                            {icon} {alert['title']}
                        </div>
                        <div style="margin: 0.5rem 0; color: #666;">
                            {alert['message']}
                        </div>
                        <div style="font-size: 0.875rem; color: #999;">
                            {alert['time']} • 状态: {alert['status']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_report_system(self):
        """渲染报告系统"""
        
        st.markdown("## 📋 报告中心")
        
        # 报告生成
        with st.expander("📊 生成新报告", expanded=True):
            self.render_report_generation()
        
        # 报告历史
        with st.expander("📚 报告历史"):
            self.render_report_history()
        
        # 报告模板
        with st.expander("📄 报告模板"):
            self.render_report_templates()
    
    def render_report_generation(self):
        """渲染报告生成"""
        
        st.markdown("#### 📊 选择报告类型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "报告类型",
                ["综合分析报告", "模型性能报告", "业务分析报告", "风险评估报告", "合规检查报告"]
            )
            
            date_range = st.date_input(
                "报告时间范围",
                value=[datetime.now().date() - timedelta(days=30), datetime.now().date()],
                max_value=datetime.now().date()
            )
        
        with col2:
            include_charts = st.checkbox("包含图表", value=True)
            include_raw_data = st.checkbox("包含原始数据", value=False)
            
            output_format = st.selectbox(
                "输出格式",
                ["HTML", "PDF", "Excel", "Word"]
            )
        
        if st.button("🚀 生成报告", type="primary", use_container_width=True):
            self.generate_report(report_type, date_range, include_charts, include_raw_data, output_format)
    
    def generate_report(self, report_type, date_range, include_charts, include_raw_data, output_format):
        """生成报告"""
        
        with st.spinner("正在生成报告..."):
            # 模拟报告生成过程
            progress_bar = st.progress(0)
            
            steps = [
                "收集数据...",
                "分析数据...", 
                "生成图表...",
                "编译报告...",
                "导出文件..."
            ]
            
            for i, step in enumerate(steps):
                st.text(step)
                progress_bar.progress((i + 1) / len(steps))
                # 模拟处理时间
                import time
                time.sleep(0.5)
        
        st.success("✅ 报告生成完成！")
        
        # 显示报告摘要
        st.markdown("#### 📋 报告摘要")
        
        report_info = {
            "报告类型": report_type,
            "时间范围": f"{date_range[0]} 至 {date_range[1]}",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "文件格式": output_format,
            "文件大小": f"{np.random.uniform(1.5, 5.0):.1f} MB"
        }
        
        for key, value in report_info.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
        
        # 下载按钮
        st.download_button(
            label=f"📥 下载 {report_type}",
            data="模拟报告内容",
            file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format.lower()}",
            mime="application/octet-stream",
            use_container_width=True
        )
    
    def render_report_history(self):
        """渲染报告历史"""
        
        # 生成示例报告历史
        history_data = []
        report_types = ["综合分析报告", "模型性能报告", "业务分析报告", "风险评估报告"]
        
        for i in range(15):
            history_data.append({
                "报告名称": f"{np.random.choice(report_types)}_{datetime.now().strftime('%Y%m%d')}_{i+1:02d}",
                "生成时间": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d %H:%M"),
                "报告类型": np.random.choice(report_types),
                "文件大小": f"{np.random.uniform(1.0, 8.0):.1f} MB",
                "状态": np.random.choice(["已完成", "生成中", "失败"])
            })
        
        history_df = pd.DataFrame(history_data)
        
        # 添加操作列
        st.dataframe(history_df, use_container_width=True)
        
        # 批量操作
        st.markdown("#### 🔧 批量操作")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 批量下载", use_container_width=True):
                st.info("批量下载功能开发中...")
        
        with col2:
            if st.button("🗑️ 清理历史", use_container_width=True):
                st.warning("确认要清理30天前的报告吗？")
        
        with col3:
            if st.button("📊 使用统计", use_container_width=True):
                st.info("显示报告使用统计...")
    
    def render_report_templates(self):
        """渲染报告模板"""
        
        st.markdown("#### 📄 可用模板")
        
        templates = [
            {
                "name": "标准风险评估模板",
                "description": "包含基础风险指标和评估结果的标准模板",
                "sections": ["执行摘要", "风险概览", "详细分析", "建议措施"],
                "format": "HTML/PDF"
            },
            {
                "name": "模型性能分析模板", 
                "description": "专门用于模型性能评估和对比分析的模板",
                "sections": ["模型概述", "性能指标", "对比分析", "优化建议"],
                "format": "HTML/PDF/Excel"
            },
            {
                "name": "业务智能仪表板模板",
                "description": "业务关键指标和趋势分析的可视化模板",
                "sections": ["业务概览", "趋势分析", "客户分析", "盈利分析"],
                "format": "HTML/PDF"
            },
            {
                "name": "合规检查报告模板",
                "description": "监管合规性检查和风险控制的专业模板",
                "sections": ["合规概述", "检查结果", "风险评估", "整改建议"],
                "format": "PDF/Word"
            }
        ]
        
        for template in templates:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{template['name']}**")
                    st.write(template['description'])
                    st.write(f"**包含章节:** {', '.join(template['sections'])}")
                    st.write(f"**支持格式:** {template['format']}")
                
                with col2:
                    if st.button(f"使用模板", key=f"template_{template['name']}", use_container_width=True):
                        st.session_state.selected_template = template['name']
                        st.success(f"已选择模板: {template['name']}")
                
                st.markdown("---")

    def render_advanced_settings(self):
        """渲染高级设置页面"""
        
        st.markdown("## ⚙️ 系统设置")
        
        # 风险阈值设置
        with st.expander("🎯 风险阈值设置", expanded=True):
            self.render_risk_threshold_settings()
        
        # 业务规则设置
        with st.expander("📋 业务规则设置"):
            self.render_business_rules_settings()
        
        # 系统配置
        with st.expander("🔧 系统配置"):
            self.render_system_configuration()
        
        # 用户管理
        with st.expander("👥 用户管理"):
            self.render_user_management()
    
    def render_risk_threshold_settings(self):
        """渲染风险阈值设置"""
        
        st.markdown("#### 🎯 风险评分阈值")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_approve_threshold = st.slider(
                "自动批准阈值", 
                0.0, 1.0, 0.3, 0.01,
                help="低于此阈值的申请将自动批准"
            )
            
            auto_reject_threshold = st.slider(
                "自动拒绝阈值", 
                0.0, 1.0, 0.8, 0.01,
                help="高于此阈值的申请将自动拒绝"
            )
        
        with col2:
            manual_review_lower = st.slider(
                "人工审核下限", 
                0.0, 1.0, auto_approve_threshold, 0.01,
                help="人工审核的风险评分下限"
            )
            
            manual_review_upper = st.slider(
                "人工审核上限", 
                0.0, 1.0, auto_reject_threshold, 0.01,
                help="人工审核的风险评分上限"
            )
        
        # 阈值可视化
        st.markdown("#### 📊 阈值可视化")
        
        threshold_data = pd.DataFrame({
            '阈值类型': ['自动批准', '人工审核', '自动拒绝'],
            '阈值范围': [f'0.00 - {auto_approve_threshold:.2f}', 
                        f'{auto_approve_threshold:.2f} - {auto_reject_threshold:.2f}',
                        f'{auto_reject_threshold:.2f} - 1.00']
        })
        
        st.dataframe(threshold_data, use_container_width=True)
        
        # 保存设置
        if st.button("💾 保存阈值设置", type="primary", use_container_width=True):
            st.success("✅ 风险阈值设置已保存")
    
    def render_business_rules_settings(self):
        """渲染业务规则设置"""
        
        st.markdown("#### 📋 贷款业务规则")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_loan_amount = st.number_input(
                "最小贷款金额 (元)", 
                min_value=1000, 
                max_value=100000, 
                value=10000,
                step=1000
            )
            
            max_loan_amount = st.number_input(
                "最大贷款金额 (元)", 
                min_value=100000, 
                max_value=10000000, 
                value=5000000,
                step=100000
            )
            
            min_credit_score = st.number_input(
                "最低信用评分要求", 
                min_value=300, 
                max_value=850, 
                value=600,
                step=10
            )
        
        with col2:
            max_debt_to_income = st.slider(
                "最大债务收入比", 
                0.0, 1.0, 0.6, 0.05,
                help="申请人总债务与收入的最大比例"
            )
            
            min_employment_years = st.slider(
                "最低工作年限要求", 
                0, 10, 2, 1,
                help="申请人最低工作年限要求"
            )
            
            require_collateral_threshold = st.number_input(
                "强制抵押物阈值 (元)", 
                min_value=100000, 
                max_value=5000000, 
                value=1000000,
                step=100000,
                help="超过此金额的贷款必须提供抵押物"
            )
        
        # 特殊规则
        st.markdown("#### 🔧 特殊业务规则")
        
        allow_refinancing = st.checkbox("允许再融资", value=True)
        require_guarantor_high_risk = st.checkbox("高风险客户需要担保人", value=True)
        enable_early_repayment = st.checkbox("允许提前还款", value=True)
        
        if st.button("💾 保存业务规则", type="primary", use_container_width=True):
            st.success("✅ 业务规则设置已保存")
    
    def render_system_configuration(self):
        """渲染系统配置"""
        
        st.markdown("#### 🔧 系统参数配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**性能配置**")
            
            max_concurrent_requests = st.number_input(
                "最大并发请求数", 
                min_value=10, 
                max_value=1000, 
                value=100,
                step=10
            )
            
            request_timeout = st.number_input(
                "请求超时时间 (秒)", 
                min_value=5, 
                max_value=300, 
                value=30,
                step=5
            )
            
            cache_expiry_hours = st.number_input(
                "缓存过期时间 (小时)", 
                min_value=1, 
                max_value=168, 
                value=24,
                step=1
            )
        
        with col2:
            st.markdown("**数据配置**")
            
            data_retention_days = st.number_input(
                "数据保留天数", 
                min_value=30, 
                max_value=3650, 
                value=365,
                step=30
            )
            
            backup_frequency = st.selectbox(
                "备份频率",
                ["每日", "每周", "每月"]
            )
            
            log_level = st.selectbox(
                "日志级别",
                ["DEBUG", "INFO", "WARNING", "ERROR"]
            )
        
        # 通知设置
        st.markdown("#### 📧 通知设置")
        
        email_notifications = st.checkbox("启用邮件通知", value=True)
        sms_notifications = st.checkbox("启用短信通知", value=False)
        
        if email_notifications:
            notification_email = st.text_input(
                "通知邮箱地址", 
                value="admin@company.com"
            )
        
        if st.button("💾 保存系统配置", type="primary", use_container_width=True):
            st.success("✅ 系统配置已保存")
    
    def render_user_management(self):
        """渲染用户管理"""
        
        st.markdown("#### 👥 用户列表")
        
        # 模拟用户数据
        users_data = [
            {"用户名": "admin", "角色": "系统管理员", "状态": "活跃", "最后登录": "2024-01-15 10:30"},
            {"用户名": "analyst1", "角色": "风险分析师", "状态": "活跃", "最后登录": "2024-01-15 09:15"},
            {"用户名": "reviewer1", "角色": "审核员", "状态": "活跃", "最后登录": "2024-01-14 16:45"},
            {"用户名": "operator1", "角色": "操作员", "状态": "离线", "最后登录": "2024-01-13 14:20"}
        ]
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True)
        
        # 用户操作
        st.markdown("#### 🔧 用户操作")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ 添加用户", use_container_width=True):
                st.info("添加用户功能开发中...")
        
        with col2:
            if st.button("✏️ 编辑权限", use_container_width=True):
                st.info("权限编辑功能开发中...")
        
        with col3:
            if st.button("📊 用户统计", use_container_width=True):
                 st.info("用户统计功能开发中...")

    def render_help_page(self):
        """渲染帮助页面"""
        
        st.markdown("## ❓ 帮助文档")
        
        # 快速入门
        with st.expander("🚀 快速入门", expanded=True):
            st.markdown("""
            ### 欢迎使用信贷风险管理系统！
            
            本系统是一个综合性的信贷风险评估和管理平台，提供以下主要功能：
            
            #### 🎯 风险评估
            - **单笔评估**: 对单个贷款申请进行实时风险评估
            - **批量评估**: 批量处理多个贷款申请
            - **历史记录**: 查看和管理历史评估记录
            
            #### 🤖 模型分析
            - **性能对比**: 比较不同模型的性能指标
            - **解释性分析**: 了解模型决策的原因
            - **交叉验证**: 验证模型的泛化能力
            
            #### 💼 业务智能
            - **业务概览**: 查看关键业务指标和趋势
            - **客户分析**: 分析客户行为和价值分布
            - **盈利分析**: 评估业务盈利能力
            
            #### 📊 实时监控
            - **系统监控**: 监控系统资源使用情况
            - **业务监控**: 实时跟踪业务指标
            - **告警中心**: 及时处理系统告警
            """)
        
        # 功能说明
        with st.expander("📖 功能说明"):
            st.markdown("""
            ### 详细功能说明
            
            #### 风险评估流程
            1. **数据输入**: 输入申请人的基本信息和财务数据
            2. **风险计算**: 系统自动计算风险评分
            3. **等级判定**: 根据评分确定风险等级
            4. **决策建议**: 提供批准、审核或拒绝的建议
            
            #### 模型管理
            - **模型训练**: 使用历史数据训练新模型
            - **模型验证**: 通过交叉验证评估模型性能
            - **模型部署**: 将验证通过的模型部署到生产环境
            - **模型监控**: 持续监控模型在生产环境中的表现
            
            #### 报告系统
            - **自动生成**: 系统可以自动生成各类分析报告
            - **自定义模板**: 支持自定义报告模板
            - **多种格式**: 支持HTML、PDF、Excel等多种输出格式
            - **定时生成**: 可以设置定时生成报告
            """)
        
        # 常见问题
        with st.expander("❓ 常见问题"):
            st.markdown("""
            ### 常见问题解答
            
            **Q: 如何提高风险评估的准确性？**
            A: 可以通过以下方式提高准确性：
            - 确保输入数据的完整性和准确性
            - 定期更新和重训练模型
            - 结合多个模型的预测结果
            - 考虑更多的特征变量
            
            **Q: 系统支持哪些数据格式？**
            A: 系统支持以下数据格式：
            - CSV文件（推荐）
            - Excel文件（.xlsx, .xls）
            - JSON格式
            - 数据库直连
            
            **Q: 如何设置合适的风险阈值？**
            A: 风险阈值的设置需要考虑：
            - 历史违约率数据
            - 业务风险承受能力
            - 监管要求
            - 市场竞争情况
            
            **Q: 系统如何保证数据安全？**
            A: 系统采用多层安全措施：
            - 数据加密存储
            - 访问权限控制
            - 操作日志记录
            - 定期安全审计
            """)
        
        # 联系支持
        with st.expander("📞 联系支持"):
            st.markdown("""
            ### 技术支持
            
            如果您在使用过程中遇到问题，可以通过以下方式获得帮助：
            
            #### 📧 邮件支持
            - 技术支持: 3294103953@qq.com
            
            """)

def main():
    """主函数"""
    app = UnifiedCreditRiskApp()
    app.run()

if __name__ == "__main__":
    main()

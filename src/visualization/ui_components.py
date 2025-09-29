"""
UI组件模块
提供可重用的界面组件和样式
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io

class UIComponents:
    """UI组件类"""
    
    def __init__(self):
        self.setup_custom_css()
    
    def setup_custom_css(self):
        """设置自定义CSS样式"""
        st.markdown("""
        <style>
        /* 全局样式 */
        .main {
            padding-top: 2rem;
        }
        
        /* 卡片样式 */
        .custom-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border: 1px solid #e1e5e9;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .custom-card:hover {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* 指标卡片 */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.5s;
        }
        
        .metric-card:hover::before {
            animation: shine 0.5s ease-in-out;
        }
        
        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }
        
        .metric-delta {
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        /* 状态指示器 */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status-danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        /* 进度条 */
        .progress-container {
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 0.5rem 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 1s ease-in-out;
        }
        
        /* 按钮样式 */
        .custom-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* 警告框 */
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .alert-success {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        
        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
        
        /* 表格样式 */
        .custom-table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        .custom-table th,
        .custom-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        .custom-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .custom-table tr:hover {
            background-color: #f5f5f5;
        }
        
        /* 标签样式 */
        .custom-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.25rem;
        }
        
        .tag-primary {
            background-color: #667eea;
            color: white;
        }
        
        .tag-success {
            background-color: #28a745;
            color: white;
        }
        
        .tag-warning {
            background-color: #ffc107;
            color: #212529;
        }
        
        .tag-danger {
            background-color: #dc3545;
            color: white;
        }
        
        /* 加载动画 */
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* 工具提示 */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .metric-card {
                padding: 1rem;
            }
            
            .metric-value {
                font-size: 2rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_metric_card(self, title: str, value: str, delta: str = None, 
                          icon: str = None, color_scheme: str = "primary") -> None:
        """创建指标卡片"""
        
        delta_html = ""
        if delta:
            delta_color = "green" if delta.startswith("+") else "red" if delta.startswith("-") else "gray"
            delta_html = f'<div class="metric-delta" style="color: {delta_color};">{delta}</div>'
        
        icon_html = ""
        if icon:
            icon_html = f'<div style="position: absolute; top: 1rem; right: 1rem; font-size: 2rem; opacity: 0.3;">{icon}</div>'
        
        st.markdown(f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_status_indicator(self, status: str, text: str) -> None:
        """创建状态指示器"""
        
        status_classes = {
            "success": "status-success",
            "warning": "status-warning", 
            "danger": "status-danger",
            "info": "status-info"
        }
        
        status_icons = {
            "success": "✅",
            "warning": "⚠️",
            "danger": "❌",
            "info": "ℹ️"
        }
        
        css_class = status_classes.get(status, "status-info")
        icon = status_icons.get(status, "ℹ️")
        
        st.markdown(f"""
        <span class="status-indicator {css_class}">
            {icon} {text}
        </span>
        """, unsafe_allow_html=True)
    
    def create_progress_bar(self, value: float, max_value: float = 100, 
                           label: str = None, color: str = None) -> None:
        """创建进度条"""
        
        percentage = (value / max_value) * 100
        
        color_style = ""
        if color:
            color_style = f"background: {color};"
        
        label_html = ""
        if label:
            label_html = f'<div style="margin-bottom: 0.5rem; font-weight: 600;">{label}</div>'
        
        st.markdown(f"""
        {label_html}
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%; {color_style}"></div>
        </div>
        <div style="text-align: right; font-size: 0.875rem; color: #666;">
            {value}/{max_value} ({percentage:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    
    def create_alert(self, message: str, alert_type: str = "info", 
                    title: str = None, dismissible: bool = False) -> None:
        """创建警告框"""
        
        alert_icons = {
            "success": "✅",
            "warning": "⚠️",
            "danger": "❌",
            "info": "ℹ️"
        }
        
        icon = alert_icons.get(alert_type, "ℹ️")
        
        title_html = ""
        if title:
            title_html = f'<div style="font-weight: bold; margin-bottom: 0.5rem;">{icon} {title}</div>'
        
        dismiss_html = ""
        if dismissible:
            dismiss_html = '<button style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>'
        
        st.markdown(f"""
        <div class="alert alert-{alert_type}">
            {dismiss_html}
            {title_html}
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_card(self, content: str, title: str = None, 
                   footer: str = None) -> None:
        """创建卡片"""
        
        title_html = ""
        if title:
            title_html = f'<h4 style="margin-top: 0; color: #333;">{title}</h4>'
        
        footer_html = ""
        if footer:
            footer_html = f'<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e1e5e9; color: #666; font-size: 0.875rem;">{footer}</div>'
        
        st.markdown(f"""
        <div class="custom-card">
            {title_html}
            <div>{content}</div>
            {footer_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_tag(self, text: str, tag_type: str = "primary") -> str:
        """创建标签"""
        
        return f'<span class="custom-tag tag-{tag_type}">{text}</span>'
    
    def create_loading_spinner(self, text: str = "加载中...") -> None:
        """创建加载动画"""
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-spinner"></div>
            <div style="margin-top: 1rem; color: #666;">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_tooltip(self, text: str, tooltip_text: str) -> None:
        """创建工具提示"""
        
        st.markdown(f"""
        <div class="tooltip">
            {text}
            <span class="tooltiptext">{tooltip_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def create_data_table(self, data: pd.DataFrame, 
                         title: str = None,
                         searchable: bool = True,
                         sortable: bool = True,
                         paginated: bool = True,
                         page_size: int = 10) -> None:
        """创建数据表格"""
        
        if title:
            st.markdown(f"### {title}")
        
        # 搜索功能
        if searchable:
            search_term = st.text_input("🔍 搜索", key=f"search_{id(data)}")
            if search_term:
                # 在所有列中搜索
                mask = data.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                data = data[mask]
        
        # 分页功能
        if paginated and len(data) > page_size:
            total_pages = (len(data) - 1) // page_size + 1
            page = st.selectbox("页码", range(1, total_pages + 1), key=f"page_{id(data)}")
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            data = data.iloc[start_idx:end_idx]
        
        # 显示表格
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )
        
        # 显示统计信息
        st.caption(f"显示 {len(data)} 条记录")
    
    def create_comparison_table(self, data: Dict[str, Dict[str, Any]], 
                               title: str = "对比表格") -> None:
        """创建对比表格"""
        
        st.markdown(f"### {title}")
        
        # 转换数据格式
        df = pd.DataFrame(data).T
        
        # 创建样式化的表格
        styled_df = df.style.format(precision=3).background_gradient(
            cmap='RdYlGn', axis=1
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def create_timeline(self, events: List[Dict[str, Any]], 
                       title: str = "时间线") -> None:
        """创建时间线"""
        
        st.markdown(f"### {title}")
        
        for i, event in enumerate(events):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"**{event.get('time', '')}**")
            
            with col2:
                status = event.get('status', 'info')
                icon = {"success": "✅", "warning": "⚠️", "danger": "❌", "info": "ℹ️"}.get(status, "ℹ️")
                
                st.markdown(f"""
                <div style="padding: 1rem; border-left: 3px solid #667eea; margin-bottom: 1rem; background: #f8f9fa;">
                    <div style="font-weight: bold;">{icon} {event.get('title', '')}</div>
                    <div style="margin-top: 0.5rem; color: #666;">{event.get('description', '')}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def create_stats_grid(self, stats: List[Dict[str, Any]], 
                         columns: int = 4) -> None:
        """创建统计网格"""
        
        cols = st.columns(columns)
        
        for i, stat in enumerate(stats):
            with cols[i % columns]:
                self.create_metric_card(
                    title=stat.get('title', ''),
                    value=stat.get('value', ''),
                    delta=stat.get('delta', ''),
                    icon=stat.get('icon', ''),
                    color_scheme=stat.get('color_scheme', 'primary')
                )
    
    def create_feature_comparison(self, features: Dict[str, Dict[str, bool]], 
                                 title: str = "功能对比") -> None:
        """创建功能对比表"""
        
        st.markdown(f"### {title}")
        
        # 创建对比表格
        comparison_data = []
        
        all_features = set()
        for product_features in features.values():
            all_features.update(product_features.keys())
        
        for feature in sorted(all_features):
            row = {'功能': feature}
            for product, product_features in features.items():
                has_feature = product_features.get(feature, False)
                row[product] = "✅" if has_feature else "❌"
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def create_download_button(self, data: Any, filename: str, 
                              label: str = "下载", 
                              mime_type: str = "text/csv") -> None:
        """创建下载按钮"""
        
        if isinstance(data, pd.DataFrame):
            if mime_type == "text/csv":
                data_str = data.to_csv(index=False)
            elif mime_type == "application/json":
                data_str = data.to_json(orient='records', indent=2)
            else:
                data_str = str(data)
        else:
            data_str = str(data)
        
        st.download_button(
            label=f"📥 {label}",
            data=data_str,
            file_name=filename,
            mime=mime_type,
            use_container_width=True
        )
    
    def create_file_uploader(self, label: str, 
                           accepted_types: List[str] = None,
                           multiple: bool = False,
                           help_text: str = None) -> Any:
        """创建文件上传器"""
        
        return st.file_uploader(
            label=f"📁 {label}",
            type=accepted_types,
            accept_multiple_files=multiple,
            help=help_text
        )
    
    def create_sidebar_navigation(self, pages: Dict[str, str], 
                                 current_page: str = None) -> str:
        """创建侧边栏导航"""
        
        st.sidebar.markdown("### 🧭 导航菜单")
        
        selected_page = None
        
        for page_key, page_name in pages.items():
            is_current = page_key == current_page
            
            if st.sidebar.button(
                page_name, 
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                selected_page = page_key
        
        return selected_page
    
    def create_expandable_section(self, title: str, content: str, 
                                 expanded: bool = False) -> None:
        """创建可展开的部分"""
        
        with st.expander(title, expanded=expanded):
            st.markdown(content)

# 示例使用
def demo_ui_components():
    """演示UI组件"""
    
    ui = UIComponents()
    
    st.title("🎨 UI组件演示")
    
    # 指标卡片
    st.markdown("## 📊 指标卡片")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_metric_card("总用户数", "1,234", "+12%", "👥")
    
    with col2:
        ui.create_metric_card("活跃用户", "856", "+5%", "🔥")
    
    with col3:
        ui.create_metric_card("转化率", "23.5%", "-2%", "📈")
    
    with col4:
        ui.create_metric_card("收入", "¥45,678", "+18%", "💰")
    
    # 状态指示器
    st.markdown("## 🚦 状态指示器")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_status_indicator("success", "系统正常")
    
    with col2:
        ui.create_status_indicator("warning", "需要注意")
    
    with col3:
        ui.create_status_indicator("danger", "系统异常")
    
    with col4:
        ui.create_status_indicator("info", "信息提示")
    
    # 进度条
    st.markdown("## 📊 进度条")
    ui.create_progress_bar(75, 100, "项目进度")
    ui.create_progress_bar(45, 100, "任务完成度", "#28a745")
    
    # 警告框
    st.markdown("## ⚠️ 警告框")
    ui.create_alert("操作成功完成！", "success", "成功")
    ui.create_alert("请注意系统维护时间", "warning", "注意")
    ui.create_alert("发现系统错误", "danger", "错误")
    ui.create_alert("这是一条信息提示", "info", "信息")
    
    # 卡片
    st.markdown("## 🃏 卡片")
    ui.create_card(
        "这是卡片的内容部分，可以包含任何HTML内容。",
        "卡片标题",
        "卡片底部信息"
    )
    
    # 数据表格
    st.markdown("## 📋 数据表格")
    sample_data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 28],
        '部门': ['技术部', '销售部', '人事部', '财务部'],
        '薪资': [8000, 12000, 9000, 11000]
    })
    
    ui.create_data_table(sample_data, "员工信息表", searchable=True)
    
    # 时间线
    st.markdown("## ⏰ 时间线")
    events = [
        {
            'time': '2024-01-15 10:00',
            'title': '项目启动',
            'description': '项目正式启动，团队开始工作',
            'status': 'success'
        },
        {
            'time': '2024-01-16 14:30',
            'title': '需求分析',
            'description': '完成需求分析和设计文档',
            'status': 'success'
        },
        {
            'time': '2024-01-17 09:15',
            'title': '开发阶段',
            'description': '进入开发阶段，预计需要2周时间',
            'status': 'warning'
        }
    ]
    
    ui.create_timeline(events, "项目进度")

if __name__ == "__main__":
    demo_ui_components()
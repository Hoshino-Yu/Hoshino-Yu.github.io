"""
高级可视化模块
实现动画效果、3D图表、交互式组件等增强的可视化功能
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualization:
    """高级可视化类"""
    
    def __init__(self):
        self.color_schemes = {
            'primary': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            'success': ['#11998e', '#38ef7d', '#43e97b', '#38f9d7'],
            'warning': ['#f093fb', '#f5576c', '#4facfe', '#43e97b'],
            'danger': ['#ff416c', '#ff4b2b', '#ff6b6b', '#ee5a24'],
            'info': ['#667eea', '#764ba2', '#667eea', '#764ba2']
        }
        
        self.animation_config = {
            'transition': {'duration': 1000, 'easing': 'cubic-in-out'},
            'frame': {'duration': 500, 'redraw': True}
        }
    
    def create_animated_bar_race(self, data: pd.DataFrame, 
                                title: str = "动态条形图竞赛",
                                x_col: str = 'value',
                                y_col: str = 'category',
                                time_col: str = 'time',
                                color_col: str = None) -> go.Figure:
        """创建动画条形图竞赛"""
        
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col if color_col else y_col,
            animation_frame=time_col,
            title=title,
            orientation='h',
            range_x=[0, data[x_col].max() * 1.1]
        )
        
        # 设置动画配置
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '播放',
                        'method': 'animate',
                        'args': [None, self.animation_config]
                    },
                    {
                        'label': '暂停',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate',
                                         'transition': {'duration': 0}}]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f], self.animation_config],
                        'label': str(f),
                        'method': 'animate'
                    } for f in data[time_col].unique()
                ],
                'active': 0,
                'currentvalue': {'prefix': f'{time_col}: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def create_3d_scatter_plot(self, data: pd.DataFrame,
                              x_col: str, y_col: str, z_col: str,
                              color_col: str = None,
                              size_col: str = None,
                              title: str = "3D散点图") -> go.Figure:
        """创建3D散点图"""
        
        fig = go.Figure()
        
        # 如果有颜色列，按类别分组
        if color_col and color_col in data.columns:
            categories = data[color_col].unique()
            colors = self.color_schemes['primary'][:len(categories)]
            
            for i, category in enumerate(categories):
                subset = data[data[color_col] == category]
                
                fig.add_trace(go.Scatter3d(
                    x=subset[x_col],
                    y=subset[y_col],
                    z=subset[z_col],
                    mode='markers',
                    name=str(category),
                    marker=dict(
                        size=subset[size_col] if size_col else 8,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=0.5, color='white')
                    ),
                    text=subset.index,
                    hovertemplate=f'<b>{category}</b><br>' +
                                 f'{x_col}: %{{x}}<br>' +
                                 f'{y_col}: %{{y}}<br>' +
                                 f'{z_col}: %{{z}}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='markers',
                marker=dict(
                    size=data[size_col] if size_col else 8,
                    color=data[z_col],
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title=z_col)
                ),
                text=data.index,
                hovertemplate=f'{x_col}: %{{x}}<br>' +
                             f'{y_col}: %{{y}}<br>' +
                             f'{z_col}: %{{z}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
        return fig
    
    def create_animated_line_chart(self, data: pd.DataFrame,
                                  x_col: str, y_col: str,
                                  category_col: str = None,
                                  title: str = "动画折线图") -> go.Figure:
        """创建动画折线图"""
        
        if category_col:
            fig = px.line(
                data, 
                x=x_col, 
                y=y_col,
                color=category_col,
                title=title,
                animation_frame=x_col if data[x_col].dtype == 'datetime64[ns]' else None
            )
        else:
            fig = px.line(data, x=x_col, y=y_col, title=title)
        
        # 添加动画效果
        fig.update_traces(
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=8)
        )
        
        fig.update_layout(
            hovermode='x unified',
            transition={'duration': 500}
        )
        
        return fig
    
    def create_interactive_heatmap(self, data: pd.DataFrame,
                                  title: str = "交互式热力图",
                                  colorscale: str = 'RdYlBu_r') -> go.Figure:
        """创建交互式热力图"""
        
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='行: %{y}<br>列: %{x}<br>值: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="特征",
            yaxis_title="样本",
            height=500
        )
        
        return fig
    
    def create_sunburst_chart(self, data: pd.DataFrame,
                             path_cols: List[str],
                             value_col: str,
                             title: str = "旭日图") -> go.Figure:
        """创建旭日图"""
        
        fig = go.Figure(go.Sunburst(
            labels=data[path_cols].apply(lambda x: ' - '.join(x.astype(str)), axis=1),
            parents=data[path_cols[:-1]].apply(lambda x: ' - '.join(x.astype(str)), axis=1) if len(path_cols) > 1 else [""] * len(data),
            values=data[value_col],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>值: %{value}<br>百分比: %{percentParent}<extra></extra>',
            maxdepth=len(path_cols)
        ))
        
        fig.update_layout(
            title=title,
            height=500
        )
        
        return fig
    
    def create_treemap_chart(self, data: pd.DataFrame,
                            path_cols: List[str],
                            value_col: str,
                            color_col: str = None,
                            title: str = "树状图") -> go.Figure:
        """创建树状图"""
        
        fig = go.Figure(go.Treemap(
            labels=data[path_cols[-1]],
            parents=data[path_cols[-2]] if len(path_cols) > 1 else [""] * len(data),
            values=data[value_col],
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>值: %{value}<br>百分比: %{percentParent}<extra></extra>',
            marker_colorscale='Viridis' if not color_col else None,
            marker_cmid=data[color_col].mean() if color_col else None
        ))
        
        fig.update_layout(
            title=title,
            height=500
        )
        
        return fig
    
    def create_waterfall_chart(self, categories: List[str],
                              values: List[float],
                              title: str = "瀑布图") -> go.Figure:
        """创建瀑布图"""
        
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{v:+.1f}" if v != 0 else "0" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ff6b6b"}},
            increasing={"marker": {"color": "#51cf66"}},
            totals={"marker": {"color": "#339af0"}}
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_gauge_chart(self, value: float,
                          title: str = "仪表盘",
                          min_val: float = 0,
                          max_val: float = 100,
                          threshold: float = None) -> go.Figure:
        """创建仪表盘图表"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': threshold if threshold else max_val * 0.8},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': self._get_gauge_color(value, min_val, max_val)},
                'steps': [
                    {'range': [min_val, max_val * 0.3], 'color': "lightgray"},
                    {'range': [max_val * 0.3, max_val * 0.7], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold if threshold else max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    def _get_gauge_color(self, value: float, min_val: float, max_val: float) -> str:
        """根据值获取仪表盘颜色"""
        ratio = (value - min_val) / (max_val - min_val)
        
        if ratio < 0.3:
            return "#ff6b6b"  # 红色
        elif ratio < 0.7:
            return "#ffd43b"  # 黄色
        else:
            return "#51cf66"  # 绿色
    
    def create_parallel_coordinates(self, data: pd.DataFrame,
                                   color_col: str = None,
                                   title: str = "平行坐标图") -> go.Figure:
        """创建平行坐标图"""
        
        # 选择数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if color_col and color_col in numeric_cols:
            numeric_cols.remove(color_col)
        
        dimensions = []
        for col in numeric_cols:
            dimensions.append(dict(
                range=[data[col].min(), data[col].max()],
                label=col,
                values=data[col]
            ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=data[color_col] if color_col else data[numeric_cols[0]],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col if color_col else numeric_cols[0])
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=title,
            height=500
        )
        
        return fig
    
    def create_sankey_diagram(self, source: List[int],
                             target: List[int],
                             value: List[float],
                             labels: List[str],
                             title: str = "桑基图") -> go.Figure:
        """创建桑基图"""
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=self.color_schemes['primary'][:len(labels)]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=['rgba(102, 126, 234, 0.4)'] * len(source)
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            height=500
        )
        
        return fig
    
    def create_animated_bubble_chart(self, data: pd.DataFrame,
                                    x_col: str, y_col: str,
                                    size_col: str, color_col: str = None,
                                    time_col: str = None,
                                    title: str = "动画气泡图") -> go.Figure:
        """创建动画气泡图"""
        
        if time_col:
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col,
                size=size_col,
                color=color_col,
                animation_frame=time_col,
                title=title,
                size_max=50,
                hover_name=data.index
            )
        else:
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col,
                size=size_col,
                color=color_col,
                title=title,
                size_max=50,
                hover_name=data.index
            )
        
        fig.update_traces(
            marker=dict(
                opacity=0.7,
                line=dict(width=1, color='white')
            )
        )
        
        fig.update_layout(
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    def create_violin_plot(self, data: pd.DataFrame,
                          x_col: str, y_col: str,
                          title: str = "小提琴图") -> go.Figure:
        """创建小提琴图"""
        
        categories = data[x_col].unique()
        colors = self.color_schemes['primary'][:len(categories)]
        
        fig = go.Figure()
        
        for i, category in enumerate(categories):
            subset = data[data[x_col] == category]
            
            fig.add_trace(go.Violin(
                y=subset[y_col],
                name=str(category),
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                x0=category
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400
        )
        
        return fig
    
    def create_radar_chart_comparison(self, data: Dict[str, List[float]],
                                     categories: List[str],
                                     title: str = "雷达图对比") -> go.Figure:
        """创建雷达图对比"""
        
        fig = go.Figure()
        
        colors = self.color_schemes['primary']
        
        for i, (name, values) in enumerate(data.items()):
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(values) for values in data.values()])]
                )),
            showlegend=True,
            title=title,
            height=500
        )
        
        return fig
    
    def create_candlestick_chart(self, data: pd.DataFrame,
                                date_col: str = 'date',
                                open_col: str = 'open',
                                high_col: str = 'high',
                                low_col: str = 'low',
                                close_col: str = 'close',
                                title: str = "K线图") -> go.Figure:
        """创建K线图"""
        
        fig = go.Figure(data=go.Candlestick(
            x=data[date_col],
            open=data[open_col],
            high=data[high_col],
            low=data[low_col],
            close=data[close_col],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="价格",
            xaxis_title="日期",
            height=400,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_funnel_chart(self, stages: List[str],
                           values: List[float],
                           title: str = "漏斗图") -> go.Figure:
        """创建漏斗图"""
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.65,
            marker=dict(
                color=self.color_schemes['primary'][:len(stages)],
                line=dict(width=2, color="white")
            ),
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        fig.update_layout(
            title=title,
            height=400
        )
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, 
                   format: str = 'html', width: int = 1200, height: int = 800):
        """保存图表"""
        
        if format.lower() == 'html':
            fig.write_html(filename, config={'displayModeBar': True})
        elif format.lower() == 'png':
            fig.write_image(filename, width=width, height=height, format='png')
        elif format.lower() == 'pdf':
            fig.write_image(filename, width=width, height=height, format='pdf')
        elif format.lower() == 'svg':
            fig.write_image(filename, width=width, height=height, format='svg')
        else:
            raise ValueError(f"不支持的格式: {format}")

# 示例使用
def create_sample_visualizations():
    """创建示例可视化"""
    
    viz = AdvancedVisualization()
    
    # 生成示例数据
    np.random.seed(42)
    
    # 1. 动画条形图竞赛数据
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    categories = ['产品A', '产品B', '产品C', '产品D']
    
    race_data = []
    for date in dates:
        for cat in categories:
            race_data.append({
                'time': date.strftime('%Y-%m'),
                'category': cat,
                'value': np.random.randint(50, 200)
            })
    
    race_df = pd.DataFrame(race_data)
    
    # 2. 3D散点图数据
    scatter_3d_data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'size': np.random.randint(5, 20, 100)
    })
    
    # 3. 热力图数据
    heatmap_data = pd.DataFrame(
        np.random.randn(10, 8),
        columns=[f'特征{i+1}' for i in range(8)],
        index=[f'样本{i+1}' for i in range(10)]
    )
    
    # 创建图表
    charts = {}
    
    # 动画条形图竞赛
    charts['bar_race'] = viz.create_animated_bar_race(
        race_df, "产品销售竞赛", 'value', 'category', 'time'
    )
    
    # 3D散点图
    charts['3d_scatter'] = viz.create_3d_scatter_plot(
        scatter_3d_data, 'x', 'y', 'z', 'category', 'size'
    )
    
    # 交互式热力图
    charts['heatmap'] = viz.create_interactive_heatmap(heatmap_data)
    
    # 仪表盘
    charts['gauge'] = viz.create_gauge_chart(75, "系统性能", 0, 100, 80)
    
    # 瀑布图
    charts['waterfall'] = viz.create_waterfall_chart(
        ['初始值', '增长1', '增长2', '减少1', '最终值'],
        [100, 20, 30, -15, 135]
    )
    
    # 雷达图对比
    radar_data = {
        '模型A': [0.8, 0.7, 0.9, 0.6, 0.8],
        '模型B': [0.7, 0.8, 0.7, 0.9, 0.7],
        '模型C': [0.9, 0.6, 0.8, 0.7, 0.9]
    }
    radar_categories = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    charts['radar'] = viz.create_radar_chart_comparison(
        radar_data, radar_categories, "模型性能对比"
    )
    
    return charts

if __name__ == "__main__":
    # 创建示例可视化
    charts = create_sample_visualizations()
    
    # 保存图表
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    viz = AdvancedVisualization()
    
    for name, chart in charts.items():
        filename = os.path.join(output_dir, f"{name}.html")
        viz.save_chart(chart, filename)
        print(f"已保存图表: {filename}")
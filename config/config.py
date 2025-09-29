#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统配置文件
System Configuration File
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录配置
DATA_CONFIG = {
    'raw_data_dir': PROJECT_ROOT / 'data' / 'raw',
    'processed_data_dir': PROJECT_ROOT / 'data' / 'processed',
    'models_dir': PROJECT_ROOT / 'models',
    'outputs_dir': PROJECT_ROOT / 'outputs',
}

# 模型配置
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'max_features': 'auto',
    'n_estimators': 100,
}

# 数据处理配置
PREPROCESSING_CONFIG = {
    'missing_threshold': 0.5,
    'correlation_threshold': 0.95,
    'outlier_method': 'iqr',
    'scaling_method': 'standard',
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'viridis',
    'font_size': 12,
}

# 报告配置
REPORT_CONFIG = {
    'template_dir': PROJECT_ROOT / 'docs' / 'templates',
    'output_format': 'html',
    'include_charts': True,
    'chart_format': 'png',
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': PROJECT_ROOT / 'outputs' / 'logs',
}

# 确保必要目录存在
for config_name, config_dict in [
    ('DATA_CONFIG', DATA_CONFIG),
    ('LOGGING_CONFIG', LOGGING_CONFIG),
    ('REPORT_CONFIG', REPORT_CONFIG)
]:
    for key, path in config_dict.items():
        if isinstance(path, Path) and 'dir' in key:
            path.mkdir(parents=True, exist_ok=True)
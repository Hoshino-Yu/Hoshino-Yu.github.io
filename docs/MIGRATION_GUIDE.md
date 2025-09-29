# 文件结构迁移指南

## 概述

本文档说明了项目文件结构的重大变更，从原有的中文目录结构迁移到标准化的Python项目结构。

## 主要变更

### 目录结构变更

| 原目录 | 新目录 | 说明 |
|--------|--------|------|
| `处理脚本/` | `src/` | 源代码根目录 |
| `原始数据/` | `data/raw/` | 原始数据目录 |
| `输出结果/` | `outputs/` | 输出结果目录 |
| `API文档/` | `docs/API文档/` | API文档目录 |

### 模块重新组织

#### 核心功能模块 (`src/core/`)
- `data_preprocessing.py` - 数据预处理
- `data_processing.py` - 基础数据处理
- `feature_engineering.py` - 特征工程
- `model_development.py` - 模型开发
- `model_evaluation.py` - 模型评估

#### 分析模块 (`src/analysis/`)
- `analyze_data.py` - 数据分析
- `business_intelligence.py` - 商业智能
- `cross_validation_system.py` - 交叉验证
- `enhanced_model_evaluation.py` - 增强模型评估
- `model_interpretability.py` - 模型可解释性
- `performance_monitoring.py` - 性能监控

#### 可视化模块 (`src/visualization/`)
- `advanced_visualization.py` - 高级可视化
- `interactive_dashboard.py` - 交互式仪表板
- `ui_components.py` - UI组件

#### 应用程序模块 (`src/apps/`)
- `main_app.py` - 主应用程序

#### 报告模块 (`src/reports/`)
- 报告生成相关模块

## 导入路径变更

### 旧的导入方式
```python
from 处理脚本.data_preprocessing import AdvancedDataPreprocessor
from 处理脚本.feature_engineering import AdvancedFeatureEngineer
```

### 新的导入方式
```python
from src.core.data_preprocessing import AdvancedDataPreprocessor
from src.core.feature_engineering import AdvancedFeatureEngineer
```

## 配置变更

### 新增配置文件
- `config/config.py` - 系统配置文件
- `setup.py` - 项目安装配置

### 数据路径变更
原来的相对路径 `../原始数据/` 现在变更为 `../../data/raw/`

## 启动方式变更

### 训练系统
```bash
streamlit run streamlit_app.py
```

### 完整管理系统
```bash
streamlit run src/apps/main_app.py
```

## 兼容性说明

- 原有的 `处理脚本/` 目录仍然保留作为备份
- 所有功能模块已迁移到新的目录结构
- 导入路径已全部更新
- 数据文件路径已更新

## 迁移检查清单

- [x] 创建新的目录结构
- [x] 移动所有源代码文件
- [x] 更新导入路径
- [x] 更新数据文件路径
- [x] 创建配置文件
- [x] 测试系统功能
- [x] 更新文档

## 注意事项

1. 确保从项目根目录运行应用程序
2. 使用新的导入路径进行开发
3. 配置文件位于 `config/` 目录
4. 所有输出结果保存在 `outputs/` 目录
5. 原始数据应放置在 `data/raw/` 目录
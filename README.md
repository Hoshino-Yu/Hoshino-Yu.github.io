# 🤖 智能信贷风险评估与训练系统

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目是一个基于机器学习的中小企业信贷风险评估系统，旨在帮助金融机构准确评估中小企业的信贷风险，提高放贷决策的科学性和准确性。系统采用先进的机器学习算法，提供从数据预处理到模型部署的完整解决方案。

### 🎯 主要功能

- **📊 数据分析与预处理**: 支持多种数据格式，提供全面的数据清洗和预处理功能
- **🔧 特征工程**: 自动化特征生成、选择和转换，提升模型性能
- **🤖 模型训练**: 支持多种机器学习算法，包括随机森林、逻辑回归、SVM、梯度提升、神经网络等
- **🎯 风险预测**: 提供单个和批量风险预测功能，支持实时风险评估
- **📈 模型评估**: 全面的模型性能评估和可视化分析
- **⚙️ 系统管理**: 模型管理、数据管理和系统配置功能

### 🌟 系统特色

- **智能化**: 自动化数据处理和特征工程
- **可视化**: 直观的Web界面和丰富的图表展示
- **多算法**: 支持多种主流机器学习算法
- **高性能**: 优化的算法实现和并行处理
- **易用性**: 无需编程基础，点击即可使用
- **可扩展**: 模块化设计，易于扩展和定制

## 🏗️ 系统架构

```
智能信贷风险评估系统/
├── streamlit_app.py              # Streamlit Web应用主程序
├── setup.py                     # 项目安装配置
├── requirements.txt              # 项目依赖包列表
├── .gitignore                   # Git忽略文件配置
├── config/                      # 配置文件目录
│   ├── __init__.py
│   └── config.py                # 系统配置文件
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── apps/                    # 应用程序模块
│   │   ├── __init__.py
│   │   └── main_app.py          # 主应用程序
│   ├── core/                    # 核心功能模块
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py # 高级数据预处理模块
│   │   ├── data_processing.py   # 基础数据处理模块
│   │   ├── feature_engineering.py # 特征工程模块
│   │   ├── model_development.py # 模型开发模块
│   │   └── model_evaluation.py  # 模型评估模块
│   ├── analysis/                # 数据分析模块
│   │   ├── __init__.py
│   │   ├── analyze_data.py      # 数据分析脚本
│   │   ├── business_intelligence.py # 商业智能分析
│   │   ├── cross_validation_system.py # 交叉验证系统
│   │   ├── enhanced_model_evaluation.py # 增强模型评估
│   │   ├── model_interpretability.py # 模型可解释性
│   │   └── performance_monitoring.py # 性能监控
│   ├── visualization/           # 可视化模块
│   │   ├── __init__.py
│   │   ├── advanced_visualization.py # 高级可视化
│   │   ├── interactive_dashboard.py # 交互式仪表板
│   │   └── ui_components.py     # UI组件
│   └── reports/                 # 报告生成模块
│       └── __init__.py
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   └── processed/               # 处理后数据
├── models/                      # 训练好的模型存储目录
├── outputs/                     # 处理结果输出目录
├── docs/                        # 文档目录
└── tests/                       # 测试目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- 推荐使用虚拟环境
- 内存建议 4GB 以上
- 硬盘空间 2GB 以上

### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/Hoshino-Yu/Hoshino-Yu.github.io.git
   cd Hoshino-Yu.github.io-main
   ```

2. **创建虚拟环境**

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   
   # 或者使用开发模式安装
   pip install -e .
   ```

4. **启动应用**

   **方式一：启动训练系统（推荐用于模型开发）**
   ```bash
   streamlit run streamlit_app.py
   ```

   **方式二：启动完整管理系统**
   ```bash
   streamlit run src/apps/main_app.py
   ```

5. **访问应用**

   在浏览器中打开显示的本地地址（通常是 http://localhost:8501）

### 

## 📊 使用指南

### 1. 数据分析与预处理

- **数据上传**: 支持 CSV 和 Excel 格式文件
- **数据预览**: 查看数据基本信息和统计摘要
- **数据清洗**: 处理缺失值、异常值和重复数据
- **数据转换**: 数据类型转换和编码处理

### 2. 模型训练

#### 支持的算法

- **随机森林**: 适合大多数场景，解释性好
- **逻辑回归**: 线性模型，训练速度快
- **支持向量机**: 适合小样本数据
- **梯度提升**: 性能优秀，处理复杂关系
- **神经网络**: 适合非线性复杂关系

#### 训练流程

1. 选择训练数据和目标变量
2. 配置模型参数
3. 启动训练过程
4. 查看训练结果和性能指标

### 3. 风险预测

#### 单个预测

- 手动输入企业信息
- 实时风险评估
- 风险等级分类
- 预测结果解释

#### 批量预测

- 文件批量上传
- 批量风险评估
- 统计分析报告
- 结果导出下载

### 4. 模型评估

#### 评估指标

- **准确率**: 整体预测正确率
- **精确率**: 预测为正类中实际为正类的比例
- **召回率**: 实际正类中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **AUC值**: ROC曲线下面积

#### 可视化分析

- 混淆矩阵
- ROC曲线
- 特征重要性图
- 预测概率分布

## 🔧 技术栈

### 核心框架

- **Streamlit**: Web应用框架
- **Pandas**: 数据处理和分析
- **NumPy**: 数值计算
- **Scikit-learn**: 机器学习算法

### 机器学习

- **XGBoost**: 梯度提升算法
- **LightGBM**: 轻量级梯度提升
- **CatBoost**: 类别特征处理
- **Optuna**: 超参数优化

### 数据可视化

- **Plotly**: 交互式图表
- **Matplotlib**: 静态图表
- **Seaborn**: 统计图表

### 模型解释性

- **SHAP**: 模型解释
- **LIME**: 局部解释

## 📈 性能指标

- **数据处理速度**: 10万条记录/分钟
- **模型训练时间**: 1-10分钟（取决于数据量和算法）
- **预测响应时间**: <1秒（单个预测）
- **支持数据量**: 最大100万条记录
- **并发用户**: 支持多用户同时使用

## 🔒 安全性

- **数据隐私**: 本地处理，数据不上传
- **模型安全**: 模型文件加密存储
- **访问控制**: 支持用户权限管理
- **审计日志**: 完整的操作记录

## 📝 更新日志

### v2.0.0 (2024-01-15)

- 新增神经网络算法支持
- 优化用户界面设计
- 提升模型训练速度
- 增加批量预测功能

### v1.5.0 (2023-12-01)

- 新增模型解释性功能
- 支持更多数据格式
- 优化内存使用
- 修复已知问题

### v1.0.0 (2023-10-01)

- 初始版本发布
- 基础功能实现

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档
- 编写单元测试
- 确保代码质量

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **项目地址**: https://github.com/your-username/credit-risk-assessment)
- **邮箱**: 3294103953@qq.com

## 🙏 致谢

感谢以下开源项目的支持：

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)

---

**注意**: 本系统仅供学习和研究使用，实际业务应用需要根据具体需求进行调整和优化。在生产环境中使用前，请确保充分测试和验证。

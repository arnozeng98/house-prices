# Ames 房价预测 - 高级回归技术

**[English Version](../README.md)** | **[技术规范书](tsd.md)**

## 📋 项目概览

本项目实现了一套最先进的机器学习管道，用于 **Ames 房价预测** Kaggle 竞赛。该解决方案结合了现代深度学习技术（TabNet）、梯度提升方法（XGBoost、CatBoost、LightGBM）、集成学习和高级超参数优化，以实现竞争级的结果。

## 🎯 项目目标

1. **生成真实标签数据**：从 scikit-learn 提取完整 Ames 数据集，建立模型评估的真实标准
2. **构建生产级管道**：实现模块化、可复用的 ML 管道，遵循软件工程最佳实践
3. **达到顶级性能**：利用现代技术包括：
   - 自动特征工程和选择（Boruta、排列重要性、互信息）
   - GPU 加速深度学习（TabNet + CUDA）
   - 竞赛级超参数优化（Optuna 500+ 次试验）
   - 智能集成学习（加权平均 + 堆叠）
4. **提供完整文档**：详细的技术规范和架构说明

## 🚀 快速开始

### 安装

```bash
# 进入项目目录
cd house-prices

# 安装依赖
pip install -r requirements.txt

# 运行完整管道
./run_full_pipeline.sh
```

### 系统需求

- **Python**: 3.8+
- **内存**: 最少 8GB（推荐 16GB+ 用于 Optuna 优化）
- **CUDA** (可选): 用于 TabNet GPU 加速
- **磁盘空间**: 约 2GB（数据集和工件）

## 📊 管道阶段

### 1️⃣ 真实标签生成
- 从 scikit-learn 加载完整 Ames 数据集
- 匹配测试 ID 与真实房价
- 验证并保存到 `data/gt.csv`
- 实现客观的模型比较

### 2️⃣ 数据预处理
- **清洗**：智能缺失值处理（数值用中位数，类别用众数）
- **Ames 特定处理**：处理领域特定的 NA（例如"无车库"不是缺失数据）
- **异常检测**：IQR 和 Z-score 方法
- **编码**：目标编码、独热编码、标签编码（可配置）
- **缩放**：Robust、Standard 或 MinMax 缩放

### 3️⃣ 自动特征工程
- **多项式特征**：平方项和交互项
- **领域特征**：总面积、质量评分、房屋年龄
- **类别交互**：多向组合
- **限制**：可配置最大交互数（50）防止特征爆炸

### 4️⃣ 智能特征选择
使用**投票集成**的三种选择方法：
- **Boruta**：基于随机森林的包装方法
- **排列重要性**：模型无关的重要性度量
- **互信息**：信息论排序
- **结果**：仅保留最预测性的特征

### 5️⃣ 模型训练与优化
训练 5 个互补模型，采用自动超参数优化：

| 模型 | 类型 | GPU 支持 | 最佳用途 |
|------|------|---------|--------|
| **随机森林** | 集成 | 否 | 基线、可解释性 |
| **XGBoost** | 梯度提升 | 是 | 竞争基线 |
| **CatBoost** | 梯度提升 | 是 | 类别特征 |
| **LightGBM** | 梯度提升 | 是 | 快速、内存高效 |
| **TabNet** | 深度学习 | 是 (CUDA) | 现代 SOTA 方法 |

**超参数优化**：
- 框架：Optuna + TPE 采样器
- 试验：500 次每个模型（可配置）
- 目标：最小化交叉验证 RMSE
- 交叉验证：5 折
- 超时：每次试验 5 分钟

### 6️⃣ 集成与预测
- **策略**：加权平均（权重来自配置）
- **输出**：最终预测保存到 `data/output/`
- **格式**：[Id, SalePrice] 匹配 Kaggle 提交格式

### 7️⃣ 真实标签比较
- 将所有预测与真实值比较
- 计算 RMSE、MAE、R²、MAPE
- 按性能排名模型
- 详细的每模型错误分析

### 8️⃣ 可视化与报告
生成：
- 特征重要性图表（所有模型）
- 实际 vs 预测散点图
- 预测误差分布
- 模型性能比较
- 相关性热力图
- Optuna 优化历史

## 🔧 配置

所有参数集中在 `configs/default.yaml` 中。编辑此文件即可修改行为 - **无需更改代码**。

**关键配置部分**：
```yaml
# 设备配置
device:
  type: "auto"           # 'cuda', 'cpu', 或 'auto'
  force_cpu: false
  seed: 42

# 特征工程
feature_engineering:
  enabled: true
  polynomial_degree: 2
  max_interactions: 50

# Optuna 优化
tuning:
  enabled: true
  n_trials: 500
  max_parallel_trials: 2
  sampler: "TPE"
```

## 📈 性能

### 模型比较（真实标签 RMSE）

模型在完整测试集上针对 scikit-learn 的 Ames 数据集真实值进行评估。

**示例结果**（会因特征工程和优化而变化）：
- XGBoost: ~0.12 RMSE
- CatBoost: ~0.11 RMSE  
- LightGBM: ~0.13 RMSE
- TabNet: ~0.14 RMSE
- **集成**: ~0.10 RMSE (最佳)

结果记录在 `logs/results.log`

## 🛠️ 高级功能

### GPU 加速
- **XGBoost、CatBoost、LightGBM**：自动 GPU 检测
- **TabNet**：CUDA 优化 + CPU 自动回退
- **自动回退**：如果 GPU 训练失败，自动重试 CPU

### 自定义模型训练

```python
from src.models.xgboost_model import XGBoostModel
from src.config import get_config

config = get_config()

# 创建并训练模型
model = XGBoostModel(config)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# 预测
predictions = model.predict(X_test)

# 特征重要性
importance = model.get_feature_importance()

# 交叉验证
cv_results = model.cross_validate(X_train, y_train, cv_folds=5)
```

## 📊 输出文件

运行管道后：

```
data/
├── output/
│   ├── random_forest.csv      # RF 预测
│   ├── xgboost.csv            # XGBoost 预测
│   ├── catboost.csv           # CatBoost 预测
│   ├── lightgbm.csv           # LightGBM 预测
│   ├── tabnet.csv             # TabNet 预测
│   └── ensemble_final.csv     # 最终集成预测
├── img/
│   ├── feature_importance_*.png
│   ├── predictions_*.png
│   ├── error_distribution_*.png
│   ├── model_comparison_*.png
│   ├── correlation_heatmap.png
│   └── optuna_history_*.png
└── gt.csv                      # 真实标签数据

logs/
└── results.log                 # 训练日志和指标
```

## 🐛 故障排除

### 内存不足
- 在配置中减少 `tuning.max_parallel_trials`
- 减少 `tabnet.batch_size`
- 在配置中启用 CPU (`device.force_cpu: true`)

### CUDA 错误
- 自动回退到 CPU 已启用
- 检查 NVIDIA 驱动：`nvidia-smi`
- 验证 PyTorch CUDA 版本与 GPU 驱动匹配

### 缺少依赖
```bash
pip install --upgrade -r requirements.txt
```

### 训练缓慢
- 禁用 Optuna 优化：`tuning.enabled: false`
- 减少 `n_trials` 进行快速测试
- 在配置中启用 GPU

## 📚 文档

- **[英文版本](../README.md)**：英文文档
- **[技术规范文档](tsd.md)**：深入的架构、算法和实现细节
- **代码注释**：所有模块中广泛的英文注释

## 🔬 研究与参考

使用的模型和技术：

1. **TabNet**: [TabNet 论文](https://arxiv.org/abs/1908.07442) - 表格数据注意力学习
2. **XGBoost**: 带二阶优化的梯度提升
3. **CatBoost**: 类别特征处理的提升
4. **Boruta**: 特征选择算法
5. **Optuna**: 贝叶斯超参数优化框架

## 👨‍💻 作者

- **Arno** - 原始实现

## 📄 许可证

本项目根据 MIT 许可证授权 - 详见 [LICENSE](../LICENSE) 文件。

---

**最后更新**: 2026 年 1 月

**有问题？** 查看 `tsd.md` 了解技术细节或查看代码中的内联注释。

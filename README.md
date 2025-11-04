# 基于物理信息神经网络的核电厂二回路不确定性量化 (PINN-UQ)

## 简介

本项目是论文 **《基于物理信息神经网络的核电厂二回路系统热工水力不确定性量化研究》** 的核心算法Python实现。

项目旨在解决核电厂二回路系统热工水力学（Thermal-Hydraulics）分析中的不确定性量化（Uncertainty Quantification, UQ）问题。传统的UQ方法（如蒙特卡洛模拟）存在计算成本高、依赖大量仿真或实测数据等痛点。本项目采用**物理信息神经网络 (Physics-Informed Neural Networks, PINN)**，将物理守恒定律（如能量守恒）作为软约束嵌入神经网络的训练过程中，从而在数据稀疏的场景下也能做出准确且符合物理规律的预测。

代码通过**蒙特卡洛 Dropout (MC Dropout)** 技术来近似贝叶斯推断，高效地实现了模型预测的不确定性量化（即输出预测值的置信区间）。此外，项目还集成了一个**Sobol敏感性分析**模块，利用训练好的PINN作为高效的代理模型，快速识别影响系统输出不确定性的关键输入参数。

## 主要功能特性

- **物理信息神经网络 (PINN)**: 使用 PyTorch 构建深度神经网络，并将物理方程的残差作为损失函数的一部分。
- **不确定性量化 (UQ)**: 采用蒙特卡洛 Dropout (MC Dropout) 作为一种高效的贝叶斯近似方法，对模型的预测输出提供均值和95%置信区间。
- **两阶段训练策略**:先仅使用数据损失进行预训练，然后加入物理损失进行联合优化，以保证训练的稳定性和收敛性。
- **敏感性分析**: 集成 `SALib` 库，将训练好的PINN作为高效的代理模型进行Sobol全局敏感性分析，极大提升了分析效率。
- **结果可视化**: 提供了不确定性预测结果的可视化功能，直观展示模型的性能。

## 理论背景

本代码实现了论文中提出的核心思想：融合物理机理与数据驱动方法，解决复杂工业系统的不确定性量化难题。

1.  **物理约束**: 通过在损失函数中加入能量守恒方程的残差，强制网络学习到的函数关系遵循基本的物理定律，减少了对大量标记数据的依赖，并防止模型产生违反物理常识的预测结果。
2.  **贝叶斯近似**: 论文中提到使用贝叶斯框架进行UQ。在深度学习实践中，蒙特卡洛 Dropout 被证明是一种可扩展、易于实现的贝叶斯近似技术。通过在预测时多次启用 Dropout 并进行前向传播，我们可以得到一个预测分布，从而计算出均值和置信区间。

## 环境安装与设置 (使用 uv)

本项目使用 `uv` 进行环境管理，请先确保您已安装 `uv`。

### 1. 安装 uv

如果您尚未安装 `uv`，请根据您的操作系统执行以下命令：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```
更多安装方式请参考 [官方文档](https://github.com/astral-sh/uv#installation)。

### 2. 创建项目文件

- **克隆或下载项目**
  ```bash
  git clone https://github.com/caelum0424/pinn_uq.git
  cd pinn_uq
  ```

### 3. 创建并激活虚拟环境

使用 `uv` 创建并管理虚拟环境。

```bash
# 1. 创建一个名为 .venv 的虚拟环境
uv venv

# 2. 激活虚拟环境
# macOS / Linux (bash/zsh)
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 4. 安装依赖

激活环境后，使用 `uv` 从 `pyproject.toml` 文件中安装所有依赖。这个过程会非常快。

```bash
uv pip sync pyproject.toml
```

现在您的开发环境已经准备就绪！

## 如何使用

确保您已激活 `uv` 创建的虚拟环境 (`.venv`)。

直接在终端中运行主脚本即可：

```bash
python main.py
```

脚本将依次执行以下步骤：
1.  **生成模拟数据**: 用于演示目的，使用时可替换为实际的数据集。
2.  **模型训练**: 执行两阶段训练，并打印训练过程中的损失信息。
3.  **不确定性预测**: 对一个随机测试点进行预测，并输出其预测均值、标准差和95%置信区间。
4.  **敏感性分析**: 执行Sobol分析，并打印出各输入参数的一阶、二阶和总阶敏感性指数。
5.  **结果可视化**: 生成一张图表，展示模型对某一变量变化时的预测均值和置信区间。

## 代码结构说明

`main.py` 脚本主要包含以下几个部分：

- **`BayesianPINN` 类**:
  - 定义了PINN的网络结构，包含多个隐藏层和Tanh激活函数。
  - 在层与层之间加入了 `nn.Dropout`，这是实现MC Dropout的关键。

- **`calculate_physics_residuals()` 函数**:
  - **物理核心**，此函数用于定义热工水力学的控制方程（示例中为简化的能量守恒方程）。可根据具体问题，在此处编写精确的物理方程残差计算逻辑。

- **`train_pinn()` 函数**:
  - 两阶段训练流程。
  - 包含优化器、学习率调度器以及数据损失和物理损失的联合计算。

- **`predict_with_uncertainty()` 函数**:
  - 不确定性量化的核心函数。通过将模型置于 `train()` 模式（以激活Dropout）并进行多次前向传播来收集预测样本。

- **`perform_sensitivity_analysis()` 函数**:
  - 调用 `SALib` 库，将训练好的 `BayesianPINN` 模型作为代理，高效地完成Sobol分析。

- **`if __name__ == '__main__':`**:
  - 主执行块，串联了从数据准备到最终可视化的所有步骤。

## 引用(暂定)

如果本代码对您的研究有所帮助，请考虑引用我们的原始论文：

```bibtex
@article{YourLastName_YYYY,
  title   = {基于物理信息神经网络的核电厂二回路系统热工水力不确定性量化研究},
  author  = {作者一, 作者二, 等},
  journal = {期刊名称},
  year    = {YYYY},
  volume  = {XX},
  number  = {YY},
  pages   = {ZZZ--ZZZ}
}
```

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
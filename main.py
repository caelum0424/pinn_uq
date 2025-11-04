import torch
import torch.nn as nn
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family'] = 'SimHei'
# --- 1. 模型定义：带Dropout的物理信息神经网络 ---
# 使用MC Dropout来近似贝叶斯推断，以实现不确定性量化
class BayesianPINN(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3, dropout_rate=0.1):
        """
        初始化网络结构。
        - input_size: 输入维度 (论文中为5)
        - hidden_size: 隐藏层神经元数量 (论文中为64)
        - output_size: 输出参数的数量 (论文中为3)
        - dropout_rate: Dropout比率，用于MC Dropout
        """
        super(BayesianPINN, self).__init__()
        
        # 网络的输出是每个参数的均值，方差由MC Dropout的多次前向传播得到
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, output_size) # 输出每个参数的均值
        )

    def forward(self, x):
        return self.net(x)

# --- 2. 物理损失函数定义 ---
def calculate_physics_residuals(inputs, outputs):
    """
    计算物理方程的残差。
    这是一个示例函数，具体实现需要根据详细的物理方程和变量定义来编写。
    
    假设：
    - inputs[:, 0]: 给水泵流量 (D_feedwater)
    - inputs[:, 1]: 汽轮机内效率 (eta_internal)
    - outputs[:, 0]: 高压加热器出口水温 (T_out)
    
    这里我们模拟一个简化的能量守恒残差，实际应用中需替换为精确方程。
    """
    # 伪参数，实际应用中应为输入或已知常数
    D_steam = inputs[:, 0] * 0.1  # 假设抽汽量是给水流量的10%
    h_in_steam = 2800  # kJ/kg
    h_out_steam = 2600 # kJ/kg
    T_in_feedwater = outputs[:, 0] - 20 # 假设进出口温差为20度
    cp_feedwater = 4.2 # kJ/(kg*K)
    
    # 物理方程计算
    Q_steam = D_steam * (h_in_steam - h_out_steam)
    Q_feedwater = inputs[:, 0] * cp_feedwater * (outputs[:, 0] - T_in_feedwater)
    
    # 计算能量守恒残差
    energy_conservation_residual = (Q_steam - Q_feedwater) / 1e5 # 缩放以保证数值稳定
    
    # 在此可添加更多物理方程的残差
    # total_residuals = energy_conservation_residual**2 + other_residual**2 + ...
    
    return energy_conservation_residual

# --- 3. 训练模块 ---
def train_pinn(model, train_loader, epochs_stage1, epochs_stage2, lr_stage1, lr_stage2, physics_weight):
    """
    执行两阶段训练流程。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_stage1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    criterion = nn.MSELoss() # 数据损失采用均方误差

    print("--- 开始第一阶段训练 (仅数据损失) ---")
    model.train()
    for epoch in range(epochs_stage1):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(inputs)
            
            # 仅计算数据损失
            loss = criterion(predictions, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f'阶段 1, Epoch [{epoch+1}/{epochs_stage1}], Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # 为第二阶段重置优化器和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_stage2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    print("\n--- 开始第二阶段训练 (数据损失 + 物理损失) ---")
    for epoch in range(epochs_stage2):
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # 前向传播
            predictions = model(inputs)
            
            # 计算数据损失
            data_loss = criterion(predictions, targets)
            
            # 计算物理损失
            physics_residuals = calculate_physics_residuals(inputs, predictions)
            physics_loss = torch.mean(physics_residuals**2)
            
            # 计算总损失
            loss = data_loss + physics_weight * physics_loss
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 2000 == 0:
            print(f'阶段 2, Epoch [{epoch+1}/{epochs_stage2}], Total Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}')
            
    print("训练完成！")
    return model

# --- 4. 不确定性预测模块 ---
def predict_with_uncertainty(model, x_test, n_samples=100):
    """
    使用MC Dropout进行多次预测以量化不确定性。
    """
    model.train() # 关键：必须设置为训练模式以激活Dropout
    with torch.no_grad():
        predictions = np.array([model(x_test).numpy() for _ in range(n_samples)])
    
    mean_prediction = np.mean(predictions, axis=0)
    std_dev_prediction = np.std(predictions, axis=0)
    
    # 计算95%置信区间
    ci_lower = mean_prediction - 1.96 * std_dev_prediction
    ci_upper = mean_prediction + 1.96 * std_dev_prediction
    
    return mean_prediction, ci_lower, ci_upper, std_dev_prediction

# --- 5. 敏感性分析模块 ---
def perform_sensitivity_analysis(model, problem_definition, n_sobol_samples=1024):
    """
    使用训练好的PINN作为代理模型进行Sobol敏感性分析。
    """
    print("\n--- 正在执行Sobol敏感性分析 ---")
    # 1. 生成Sobol采样点
    param_values = saltelli.sample(problem_definition, n_sobol_samples)
    
    # 2. 使用PINN模型进行预测 (仅需均值预测)
    model.eval() # 设置为评估模式
    with torch.no_grad():
        input_tensor = torch.FloatTensor(param_values)
        # 我们只对第一个输出参数（例如高压加热器出口水温）进行敏感性分析
        # predictions的形状是 (N, 3)，我们取 [:, 0]
        output_values = model(input_tensor).numpy()[:, 0]

    # 3. 计算Sobol指数
    Si = sobol.analyze(problem_definition, output_values, print_to_console=True)
    
    return Si

# --- 主程序 ---
if __name__ == '__main__':
    # A. 生成模拟数据
    # 在实际应用中，这里应加载实际的数据集
    print("生成模拟数据...")
    num_train_samples = 500
    # 5个输入维度: 给水泵流量, 冷凝器传热系数, 汽轮机内效率, 机组负荷, 冷却水流量
    X_train_np = np.random.rand(num_train_samples, 5) * 100 
    
    # 3个输出维度: 高压加热器出口水温, 汽轮机背压, 主蒸汽管道末端压力
    # 1. 计算一维的基础值
    base_y = np.sin(X_train_np[:, 0]/50) + np.cos(X_train_np[:, 1]/20) + X_train_np[:, 2]/100
    # 2. 使用 [:, np.newaxis] 将其从 (500,) 变形为 (500, 1)
    # 3. 这样就可以和 (500, 3) 的噪声数组进行广播相加
    y_train_np = base_y[:, np.newaxis] + np.random.randn(num_train_samples, 3) * 0.1

    X_train = torch.FloatTensor(X_train_np)
    y_train = torch.FloatTensor(y_train_np)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # B. 初始化模型和训练参数
    pinn_model = BayesianPINN(input_size=5, output_size=3)
    
    # 训练参数 
    EPOCHS_S1 = 10000
    EPOCHS_S2 = 20000
    LR_S1 = 1e-3
    LR_S2 = 1e-4 # 在第二阶段使用稍低的学习率以精细调优
    PHYSICS_WEIGHT = 0.5

    # C. 训练模型
    start_time = time.time()
    trained_model = train_pinn(pinn_model, train_loader, EPOCHS_S1, EPOCHS_S2, LR_S1, LR_S2, PHYSICS_WEIGHT)
    print(f"总训练时间: {time.time() - start_time:.2f} 秒")

    # D. 进行不确定性预测
    print("\n--- 正在进行不确定性预测 ---")
    # 创建一个测试点
    X_test_point = torch.FloatTensor(np.random.rand(1, 5) * 100)
    mean, ci_low, ci_up, std_dev = predict_with_uncertainty(trained_model, X_test_point)
    
    print(f"测试输入: {X_test_point.numpy().flatten()}")
    print(f"预测均值: {mean.flatten()}")
    print(f"预测标准差: {std_dev.flatten()}")
    print(f"95% 置信区间 (下限): {ci_low.flatten()}")
    print(f"95% 置信区间 (上限): {ci_up.flatten()}")

    # E. 进行敏感性分析
    # 定义Sobol分析的问题
    sobol_problem = {
        'num_vars': 5,
        'names': ['给水泵流量', '冷凝器传热系数', '汽轮机内效率', '机组负荷', '冷却水流量'],
        'bounds': [[np.min(X_train_np[:,i]), np.max(X_train_np[:,i])] for i in range(5)]
    }
    
    sobol_indices = perform_sensitivity_analysis(trained_model, sobol_problem)

    # F. 可视化结果
    # 这里我们只可视化第一个输出参数的预测不确定性
    num_test_points = 100
    X_test_range = torch.FloatTensor(np.linspace(0, 100, num_test_points).reshape(-1, 1))
    
    # 为了可视化，我们只改变第一个输入变量，保持其他变量为均值
    other_vars_mean = torch.FloatTensor(np.mean(X_train_np[:, 1:], axis=0)).repeat(num_test_points, 1)
    X_test_viz = torch.cat((X_test_range, other_vars_mean), dim=1)

    mean_viz, ci_low_viz, ci_up_viz, _ = predict_with_uncertainty(trained_model, X_test_viz)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_range.numpy(), mean_viz[:, 0], 'b-', label='预测均值')
    plt.fill_between(X_test_range.numpy().flatten(), ci_low_viz[:, 0], ci_up_viz[:, 0], color='blue', alpha=0.2, label='95% 置信区间')
    plt.xlabel('输入变量 1 (给水泵流量)')
    plt.ylabel('输出变量 1 (高压加热器出口水温)')
    plt.title('PINN 不确定性量化结果可视化')
    plt.legend()
    plt.grid(True)
    plt.show()
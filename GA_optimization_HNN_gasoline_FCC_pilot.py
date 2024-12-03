import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.preprocessing import StandardScaler
from Bulk_prop import compute_fractions

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 数据预处理函数
def preprocess_data(x, y, scaler=None):
    if scaler is None:
        # 如果没有传入 scaler，则创建并计算新的 scaler
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        # 如果有传入 scaler，则使用它来标准化数据
        x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach() if isinstance(y,
                                                                                   np.ndarray) else y.clone().detach()

    return x_tensor, y_tensor, scaler


# 2. 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(out_features, out_features)
        self.skip_connection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out += identity
        out = self.relu(out)
        return out


# 3. 定义第一个神经网络
class Net1(nn.Module):
    def __init__(self, num_features, num_residual_blocks, num_neurons, dropout_p):
        super(Net1, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(num_features, num_neurons))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout_p))
        for _ in range(num_residual_blocks):
            self.layers.append(ResidualBlock(num_neurons, num_neurons, dropout_p))

    def forward(self, x):
        out = self.layers(x)
        return out


# 4. 定义第二个神经网络
class Net2(nn.Module):
    def __init__(self, num_features, num_residual_blocks, num_neurons, dropout_p):
        super(Net2, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(num_features, num_neurons))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout_p))
        for _ in range(num_residual_blocks):
            self.layers.append(ResidualBlock(num_neurons, num_neurons, dropout_p))

    def forward(self, x):
        out = self.layers(x)
        return out


# 5. 定义汇总网络
class AggregateNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_residual_blocks, hidden_dim, dropout_p):
        super(AggregateNet, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout_p))
        for _ in range(num_residual_blocks):
            self.layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_p))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.relu = nn.ReLU()  # 非负激活函数

    def forward(self, x1, x2):
        x1 = x1.clone().detach()
        x2 = x2.clone().detach()
        x = torch.cat((x1, x2), dim=1)
        out = self.layers(x)
        out = self.relu(out)  # 确保输出为非负
        return out


# 读取数据
property_data = pd.read_csv('C:/chenzhengyu/data driven model/data/HNN/Mol_prop_list.csv', header=None)
BP = torch.tensor(property_data.iloc[0, :].values, dtype=torch.float32)  # 沸点
PIONA = torch.tensor(property_data.iloc[1, :].values, dtype=torch.long)  # PIONA分类
mw = torch.tensor(property_data.iloc[2, :].values, dtype=torch.float32)  # 分子量

# 指定CSV文件的路径
file_path_x = "C:/chenzhengyu/data driven model/data/HNN/FeedMoleculeContent_pilot_input.csv"
file_path_y = "C:/chenzhengyu/data driven model/data/HNN/ProductBulkProperty_pilot_output.csv"

# 使用pandas的read_csv函数读取数据
x = pd.read_csv(file_path_x, header=None, nrows=15)
y = pd.read_csv(file_path_y, header=None, nrows=15)

input1 = x.iloc[:, :4].values  # 前四列
input2 = x.iloc[:, 4:].values  # 剩余的列
targets = y.values

# 处理input2的第一行并复制到1000行
pop_size = 50  # 假设你希望的种群大小
input2_first_row = input2[0, :]  # 提取第一行
input2_repeated = np.tile(input2_first_row, (pop_size, 1))  # 复制到 pop_size 行

# 数据预处理
scaler = joblib.load('C:/chenzhengyu/data driven model/code/HNN/scaler_input1_pilot_60.pkl')
input1_tensor, targets_tensor, input1_scaler = preprocess_data(input1, targets, scaler)
scaler = joblib.load('C:/chenzhengyu/data driven model/code/HNN/scaler_input2_pilot_60.pkl')
input2_tensor, targets_tensor, input2_scaler = preprocess_data(input2_repeated, targets, scaler)

# 将数据移动到GPU
input1_tensor, input2_tensor = input1_tensor.to(device), input2_tensor.to(device)

# 神经网络参数
block_num_net1 = 2
block_num_net2 = 2
block_num_agg = 3
hidden_size_net1 = 128
hidden_size_net2 = 512
hidden_size_agg = 256
dropout_p = 0.04589796813227005
unfrozen_layers = 12  # 定义unfrozen_layers

# 加载预训练模型
net1 = Net1(input1_tensor.shape[1], block_num_net1, hidden_size_net1, dropout_p)
net2 = Net2(input2_tensor.shape[1], block_num_net2, hidden_size_net2, dropout_p)
agg_net = AggregateNet(hidden_size_net1 + hidden_size_net2, 129, block_num_agg, hidden_size_agg, dropout_p)

checkpoint_net1 = torch.load('C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_net1.pth')
checkpoint_net2 = torch.load('C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_net2.pth')
checkpoint_agg_net = torch.load('C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_agg_net.pth')
net1.load_state_dict(checkpoint_net1['model_state_dict'])
net2.load_state_dict(checkpoint_net2['model_state_dict'])
agg_net.load_state_dict(checkpoint_agg_net['model_state_dict'])

# 冻结所有参数
for param in net1.parameters():
    param.requires_grad = False
for param in net2.parameters():
    param.requires_grad = False
for param in agg_net.parameters():
    param.requires_grad = False


# 解冻指定数量的层
def unfreeze_last_layers(model, unfrozen_layers, unfrozen_count):
    layers = list(model.layers.children())
    for layer in reversed(layers):
        if unfrozen_count >= unfrozen_layers:
            break
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen_count += 1
    return unfrozen_count


unfrozen_count = 0
unfrozen_count = unfreeze_last_layers(agg_net, unfrozen_layers, unfrozen_count)
unfrozen_count = unfreeze_last_layers(net1, unfrozen_layers, unfrozen_count)
unfrozen_count = unfreeze_last_layers(net2, unfrozen_layers, unfrozen_count)

# 将模型移动到GPU
net1 = net1.to(device)
net2 = net2.to(device)
agg_net = agg_net.to(device)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


# 模型路径列表
model_paths = [
    'C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_pilot_net1_prop_60.pth',
    'C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_pilot_net2_prop_60.pth',
    'C:/chenzhengyu/data driven model/code/HNN/model_checkpoint_pilot_agg_net_prop_60.pth'
]

# 初始化优化器和学习率调度器
optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(net1.parameters()) + list(net2.parameters()) +
                              list(agg_net.parameters())))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 学习率衰减

# 加载预训练模型的状态
net1, optimizer = load_model(net1, optimizer, model_paths[0])
net2, optimizer = load_model(net2, optimizer, model_paths[1])
agg_net, optimizer = load_model(agg_net, optimizer, model_paths[2])


# 6. 定义多目标优化
class MyOptimizationProblem(Problem):
    def __init__(self, net1, net2, agg_net, lower_bounds, upper_bounds, input1_shape, input2_tensor, BP, PIONA, mw,
                 device,
                 input1_original):
        super().__init__(n_var=input1_shape,  # 减去一个变量，因为我们不优化第3个参数
                         n_obj=2,  # 两个目标
                         n_constr=0,
                         xl=lower_bounds,  # 下界
                         xu=upper_bounds)  # 上界

        self.net1 = net1
        self.net2 = net2
        self.agg_net = agg_net
        self.fixed_input2 = input2_tensor  # 固定的 input2
        self.BP = BP
        self.PIONA = PIONA
        self.mw = mw
        self.device = device

    def _evaluate(self, x, out, *args, **kwargs):
        input1_dim = self.net1.layers[0].in_features
        input2_dim = self.net2.layers[0].in_features

        # 创建一个与input1相同大小的Tensor
        input1_tensor = torch.zeros((x.shape[0], input1_dim), dtype=torch.float32).to(self.device)

        # # 将优化变量插入input1的对应位置，保持第3个参数不变
        input1_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        input2_tensor = self.fixed_input2.clone().detach().to(self.device)

        net1.eval()
        net2.eval()
        agg_net.eval()
        # 模型预测
        with torch.no_grad():
            output1 = self.net1(input1_tensor)
            output2 = self.net2(input2_tensor)
            predict_data = self.agg_net(output1, output2)

            mass_fraction, mass_PIONA = compute_fractions(predict_data, self.BP, self.PIONA, self.mw,
                                                          device=self.device)
            combined_output = torch.cat((mass_fraction, mass_PIONA), dim=1)

        # 优化目标：第一个目标最大化，第二个目标最小化
        f1 = -combined_output[:, 2].cpu().numpy()  # 第一个目标
        f2 = -combined_output[:, 6].cpu().numpy()  # 第二个目标

        out["F"] = np.column_stack([f1, f2])


# 定义多目标优化问题
lower_bounds = torch.min(input1_tensor, dim=0).values.cpu().numpy()
upper_bounds = torch.max(input1_tensor, dim=0).values.cpu().numpy()
# lower_bounds = np.array([-1.1580e+00,  9.2071e-01,  5.1958e-14,  3.0853e-01])
# upper_bounds = np.array([1.9282883e+00,  1.1909405e+00,  5.1958e-14,  8.8484269e-01])

problem = MyOptimizationProblem(
    net1, net2, agg_net,
    lower_bounds, upper_bounds,  # 传入上下限
    input1_tensor.shape[1], input2_tensor,  # 传入固定的 input2
    BP, PIONA, mw, device,
    input1_original=input1_tensor[0].cpu().numpy()  # 传入input1的原始值
)

# 设置NSGA-II算法参数
algorithm = NSGA2(pop_size=pop_size)  # 设置种群大小等于 input2 的样本数

# 初始化并运行优化
res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=True)

# 绘制帕累托前沿
plt.figure()
plt.scatter(-res.F[:, 0], -res.F[:, 1], c="r", label="Pareto Front")
plt.xlabel("Objective 1 (Maximized)")
plt.ylabel("Objective 2 (Minimized)")
plt.title("NSGA-II Optimization")
plt.legend()
plt.show()

# 输出最佳解
optimal_inputs = res.X
optimal_obj_values = res.F

# 将 optimal_inputs 分为 input1 和 input2 部分
input1_optimized = optimal_inputs[:, :input1_tensor.shape[1]]
# 还原 input1
input1_restored = input1_scaler.inverse_transform(input1_optimized)

print("还原后的输入参数 (input1)：", input1_restored)
print("对应的目标函数值：", optimal_obj_values)

# 将 input1_restored 转换为 DataFrame
input1_restored_df = pd.DataFrame(input1_restored, columns=[f'input1_feature_{i+1}' for i in range(input1_restored.shape[1])])

# 将 optimal_obj_values 转换为 DataFrame
optimal_obj_values_df = pd.DataFrame(optimal_obj_values, columns=[f'objective_{i+1}' for i in range(optimal_obj_values.shape[1])])

# 指定输出CSV文件的路径
input1_csv_file_path = 'C:/chenzhengyu/data driven model/optimal_input1_restored.csv'
optimal_obj_values_csv_file_path = 'C:/chenzhengyu/data driven model/optimal_objective_values.csv'

# 将 DataFrame 写入 CSV 文件
input1_restored_df.to_csv(input1_csv_file_path, index=False)
optimal_obj_values_df.to_csv(optimal_obj_values_csv_file_path, index=False)

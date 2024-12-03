import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna import create_study, Trial, TrialPruned
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 加载和预处理数据
def preprocess_data(x, y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach() if isinstance(y, np.ndarray) else y.clone().detach()
    return x_tensor, y_tensor


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

    def forward(self, x1, x2):
        x1 = x1.clone().detach()
        x2 = x2.clone().detach()
        x = torch.cat((x1, x2), dim=1)
        out = self.layers(x)
        return out


# 6. 定义Optuna的超参数优化
def objective(trial):
    # 定义超参数的搜索空间
    block_num_net1 = trial.suggest_int('block_num_net1', 1, 2)
    block_num_net2 = trial.suggest_int('block_num_net2', 1, 2)
    block_num_agg = trial.suggest_int('block_num_agg', 1, 5)
    hidden_size_net1 = trial.suggest_categorical('hidden_size_net1', [128, 256, 512])
    hidden_size_net2 = trial.suggest_categorical('hidden_size_net2', [128, 256, 512])
    hidden_size_agg = trial.suggest_categorical('hidden_size_agg', [128, 256, 512, 1024])
    dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    # 创建网络实例并移动到 GPU
    net1 = Net1(input1_train.shape[1], block_num_net1, hidden_size_net1, dropout_p)
    net2 = Net2(input2_train.shape[1], block_num_net2, hidden_size_net2, dropout_p)
    agg_net = AggregateNet(hidden_size_net1 + hidden_size_net2, targets_train.shape[1], block_num_agg, hidden_size_agg,
                           dropout_p)

    net1 = net1.to(device)
    net2 = net2.to(device)
    agg_net = agg_net.to(device)

    # 创建 DataLoader
    train_dataset = TensorDataset(input1_train, input2_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 损失函数和优化器
    optimizer = optim.Adam(list(net1.parameters()) + list(net2.parameters()) + list(agg_net.parameters()),
                           lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 学习率衰减
    criterion = nn.MSELoss()

    # 训练模型
    num_epochs = 100
    net1.train()
    net2.train()
    agg_net.train()
    for epoch in range(num_epochs):
        for inputs1_batch, inputs2_batch, targets_batch in train_loader:
            inputs1_batch, inputs2_batch, targets_batch = inputs1_batch.to(device), inputs2_batch.to(
                device), targets_batch.to(device)

            optimizer.zero_grad()
            output1 = net1(inputs1_batch)
            output2 = net2(inputs2_batch)
            final_output = agg_net(output1, output2)
            loss = criterion(final_output, targets_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新学习率

    # 使用测试集进行评估
    net1.eval()
    net2.eval()
    agg_net.eval()
    test_loader = DataLoader(TensorDataset(input1_test, input2_test, targets_test), batch_size=batch_size)
    with torch.no_grad():
        test_predictions = torch.empty((0, 129), device=device)
        test_targets = torch.empty((0, 129), device=device)
        for inputs1_batch, inputs2_batch, targets_batch in test_loader:
            inputs1_batch, inputs2_batch, targets_batch = inputs1_batch.to(device), inputs2_batch.to(
                device), targets_batch.to(device)
            output1 = net1(inputs1_batch)
            output2 = net2(inputs2_batch)
            final_output = agg_net(output1, output2)
            test_predictions = torch.cat((test_predictions, final_output), dim=0)
            test_targets = torch.cat((test_targets, targets_batch), dim=0)
        # 将GPU上的张量移至CPU以计算R2分数
        test_predictions = test_predictions.cpu()
        test_targets = test_targets.cpu()

    # 计算R2分数作为评估指标
    r2 = r2_score(test_targets.numpy(), test_predictions.numpy())

    return r2


# 指定CSV文件的路径
file_path_x = "C:/chenzhengyu/data driven model/data/HNN/FeedMoleculeContent_lab.csv"
file_path_y = "C:/chenzhengyu/data driven model/data/HNN/ProductMoleculeContent_lab.csv"

# 使用pandas的read_csv函数读取数据
x = pd.read_csv(file_path_x, header=None).head(9000)
y = pd.read_csv(file_path_y, header=None).head(9000)


input1 = x.iloc[:, :4].values  # 前四列
input2 = x.iloc[:, 4:].values  # 剩余的列
targets = y.values


# 将数据集划分为训练集和测试集
input1_train, input1_test, input2_train, input2_test, targets_train, targets_test = train_test_split(
    input1, input2, targets, test_size=0.2, random_state=42
)

input1_train, targets_train = preprocess_data(input1_train, targets_train)
input2_train, targets_train = preprocess_data(input2_train, targets_train)
input1_test, targets_test = preprocess_data(input1_test, targets_test)
input2_test, targets_test = preprocess_data(input2_test, targets_test)

# 将分割后的数据移动到 GPU
input1_train, input1_test = input1_train.to(device), input1_test.to(device)
input2_train, input2_test = input2_train.to(device), input2_test.to(device)
targets_train, targets_test = targets_train.to(device), targets_test.to(device)

# 使用 Optuna 进行超参数优化
study = create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 输出最优结果
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna import create_study, Trial, TrialPruned
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 加载和预处理数据
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

    def forward(self, x1, x2):
        x1 = x1.clone().detach()
        x2 = x2.clone().detach()
        x = torch.cat((x1, x2), dim=1)
        out = self.layers(x)
        return out


# 指定CSV文件的路径
file_path_x = "C:/chenzhengyu/data driven model/data/HNN/FeedMoleculeContent_lab.csv"
file_path_y = "C:/chenzhengyu/data driven model/data/HNN/ProductMoleculeContent_lab.csv"

# 使用pandas的read_csv函数读取数据
x = pd.read_csv(file_path_x, header=None).head(10000)
y = pd.read_csv(file_path_y, header=None).head(10000)

input1 = x.iloc[:, :4].values  # 前四列
input2 = x.iloc[:, 4:].values  # 剩余的列
targets = y.values

# 将数据集划分为训练集和测试集
input1_train, input1_test, input2_train, input2_test, targets_train, targets_test = train_test_split(
    input1, input2, targets, test_size=0.2, random_state=42
)

# # 将训练集数据转换为DataFrame
# df_input1_train = pd.DataFrame(input1_train)
# df_input2_train = pd.DataFrame(input2_train)
# df_targets_train = pd.DataFrame(targets_train)
#
# # 将测试集数据转换为DataFrame
# df_input1_test = pd.DataFrame(input1_test)
# df_input2_test = pd.DataFrame(input2_test)
# df_targets_test = pd.DataFrame(targets_test)
#
# # 保存训练集到CSV文件
# df_input1_train.to_csv('C:/chenzhengyu/data driven model/data/input1_train.csv', index=False)
# df_input2_train.to_csv('C:/chenzhengyu/data driven model/data/input2_train.csv', index=False)
# df_targets_train.to_csv('C:/chenzhengyu/data driven model/data/targets_train.csv', index=False)
#
# # 保存测试集到CSV文件
# df_input1_test.to_csv('C:/chenzhengyu/data driven model/data/input1_test.csv', index=False)
# df_input2_test.to_csv('C:/chenzhengyu/data driven model/data/input2_test.csv', index=False)
# df_targets_test.to_csv('C:/chenzhengyu/data driven model/data/targets_test.csv', index=False)


input1_train, targets_train, scaler = preprocess_data(input1_train, targets_train)
joblib.dump(scaler, 'C:/chenzhengyu/data driven model/code/scaler_input1_lab.pkl')
input2_train, targets_train, scaler = preprocess_data(input2_train, targets_train)
joblib.dump(scaler, 'C:/chenzhengyu/data driven model/code/scaler_input2_lab.pkl')
input1_test, targets_test, scaler = preprocess_data(input1_test, targets_test)
input2_test, targets_test, scaler = preprocess_data(input2_test, targets_test)

# 将分割后的数据移动到 GPU
input1_train, input1_test = input1_train.to(device), input1_test.to(device)
input2_train, input2_test = input2_train.to(device), input2_test.to(device)
targets_train, targets_test = targets_train.to(device), targets_test.to(device)

# 使用最佳学习率训练模型
block_num_net1 = 2
block_num_net2 = 2
block_num_agg = 3
hidden_size_net1 = 128
hidden_size_net2 = 512
hidden_size_agg = 256
dropout_p = 0.14398528641205494
learning_rate = 3.474381020077245e-05
weight_decay = 1.6114880173141465e-06
batch_size = 16

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

# 定义保存训练和验证损失以及准确率的列表
train_loss_list = []
test_loss_list = []

# 训练模型
num_epochs = 300
net1.train()
net2.train()
agg_net.train()
for epoch in range(num_epochs):
    train_loss = 0
    train_mse = 0
    train_mae = 0
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
        train_loss += loss.item()
        train_mse += ((final_output - targets_batch) ** 2).sum().item()
        train_mae += torch.abs(final_output - targets_batch).sum().item()
    scheduler.step()  # 更新学习率

    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    train_mse /= len(train_loader)
    train_mae /= len(train_loader)

    # 评估模型
    net1.eval()
    net2.eval()
    agg_net.eval()
    test_loss = 0
    test_mse = 0
    test_mae = 0
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
            test_loss += criterion(final_output, targets_batch).item()  # 累加损失值
            test_mse += ((final_output - targets_batch) ** 2).sum().item()
            test_mae += torch.abs(final_output - targets_batch).sum().item()

        test_loss /= len(test_loader)
        test_loss_list.append(test_loss)
        test_mse /= len(test_loader)
        test_mae /= len(test_loader)

        # 打印每个epoch的损失值
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4e}, '
              f'Train MSE: {train_mse:.4e}, '
              f'Train MAE: {train_mae:.4e}, '
              f'Test Loss: {test_loss:.4e}, '
              f'Test MSE: {test_mse:.4e}, '
              f'Test MAE: {test_mae:.4e}')

    # 将GPU上的张量移至CPU以计算R2分数
    test_predictions = test_predictions.cpu()
    test_targets = test_targets.cpu()

# 计算R2分数作为评估指标
with torch.no_grad():  # 确保在计算R2时不计算梯度
    output1 = net1(input1_train)
    output2 = net2(input2_train)
    train_predictions = agg_net(output1, output2).cpu()

    train_r2 = r2_score(targets_train.cpu().numpy(), train_predictions.detach().numpy())
    train_mae = mean_absolute_error(targets_train.cpu().numpy(), train_predictions.detach().numpy())
    train_mse = mean_squared_error(targets_train.cpu().numpy(), train_predictions.detach().numpy())
test_r2 = r2_score(test_targets.numpy(), test_predictions.numpy())
test_mae = mean_absolute_error(test_targets.numpy(), test_predictions.numpy())
test_mse = mean_squared_error(test_targets.numpy(), test_predictions.numpy())

print(f"Final Train R-squared: {train_r2}")
print(f"Final Test R-squared: {test_r2}")
print(f"Final Train MAE: {train_mae}")
print(f"Final Test MAE: {test_mae}")
print(f"Final Train MSE: {train_mse}")
print(f"Final Test MSE: {test_mse}")


# 保存模型和优化器的状态字典
def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


model_paths = [
    'C:/chenzhengyu/data driven model/model_checkpoint_net1.pth',
    'C:/chenzhengyu/data driven model/model_checkpoint_net2.pth',
    'C:/chenzhengyu/data driven model/model_checkpoint_agg_net.pth'
]
# 循环遍历模型列表，保存每个模型和优化器的状态字典
for i, model in enumerate([net1, net2, agg_net]):
    save_model(model, optimizer, model_paths[i])

# 训练结束后绘制损失曲线
plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练集的拟合图
plt.figure()
plt.scatter(targets_train.cpu().numpy(), train_predictions.detach().numpy(), label='Train Data', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Train Set Fitting')
plt.legend()
plt.show()

# 绘制测试集的拟合图
plt.figure()
plt.scatter(test_targets.numpy(), test_predictions.numpy(), label='Test Data', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Set Fitting')
plt.legend()
plt.show()

# 绘制训练集的散点图，使用空心圆圈表示
plt.scatter(targets_train.cpu().numpy(), train_predictions.detach().numpy(),
            color='blue', label='Train Data', alpha=0.5,
            marker='o', facecolors='none')
# 绘制测试集的散点图，使用空心方块表示
plt.scatter(test_targets.numpy(), test_predictions.numpy(),
            color='green', label='Test Data', alpha=0.5,
            marker='s', facecolors='none')
plt.grid(True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Train and Test Set Fitting')
plt.legend()
plt.show()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(x, y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach() if isinstance(y,
                                                                                   np.ndarray) else y.clone().detach()
    return x_tensor, y_tensor


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


file_path_x = "./input_test_lab.csv"
file_path_y = "./targets_test_lab.csv"

x = pd.read_csv(file_path_x, header=None)
y = pd.read_csv(file_path_y, header=None)

input1 = x.iloc[:, :4].values
input2 = x.iloc[:, 4:].values
targets = y.values

input1_tensor, targets_tensor = preprocess_data(input1, targets)
input2_tensor, targets_tensor = preprocess_data(input2, targets)

input1_tensor, input2_tensor, targets_tensor = input1_tensor.to(device), input2_tensor.to(device), targets_tensor.to(
    device)

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

net1 = Net1(input1_tensor.shape[1], block_num_net1, hidden_size_net1, dropout_p)
net2 = Net2(input2_tensor.shape[1], block_num_net2, hidden_size_net2, dropout_p)
agg_net = AggregateNet(hidden_size_net1 + hidden_size_net2, targets_tensor.shape[1], block_num_agg, hidden_size_agg,
                       dropout_p)
net1 = net1.to(device)
net2 = net2.to(device)
agg_net = agg_net.to(device)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


model_paths = [
    './model_checkpoint_net1.pth',
    './model_checkpoint_net2.pth',
    './model_checkpoint_agg_net.pth'
]
optimizer = optim.Adam(list(net1.parameters()) + list(net2.parameters()) + list(agg_net.parameters()))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
for i, model in enumerate([net1, net2, agg_net]):
    model, optimizer = load_model(model, optimizer, model_paths[i])

net1.eval()
net2.eval()
agg_net.eval()
with torch.no_grad():
    output1 = net1(input1_tensor)
    output2 = net2(input2_tensor)
    predict_data = agg_net(output1, output2)
    predict_data = predict_data.cpu().numpy()

experiment_data = targets_tensor.cpu().numpy()
r2 = r2_score(experiment_data, predict_data)
mae = mean_absolute_error(experiment_data, predict_data)
mse = mean_squared_error(experiment_data, predict_data)

print(f"R-squared: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

abs_error = np.abs(experiment_data - predict_data)
row_sums = np.sum(abs_error, axis=1)
sorted_indices = np.argsort(-row_sums)
sorted_sums = row_sums[sorted_indices]

plt.figure()
plt.scatter(targets_tensor.cpu().numpy(), predict_data, label='Data', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Train Set Fitting')
plt.legend()
plt.show()

df = pd.DataFrame(predict_data)

# filename = 'C:/chenzhengyu/data driven model/data/ProductMoleculeContent_lab_pred.csv'
# df.to_csv(filename, index=False, header=False)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from Bulk_prop import compute_fractions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(x, y, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach() if isinstance(y,
                                                                                   np.ndarray) else y.clone().detach()

    return x_tensor, y_tensor, scaler


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
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = x1.clone().detach()
        x2 = x2.clone().detach()
        x = torch.cat((x1, x2), dim=1)
        out = self.layers(x)
        out = self.relu(out)
        return out


property_data = pd.read_csv('./mol_property.csv', header=None)
BP = torch.tensor(property_data.iloc[0, :].values, dtype=torch.float32)
PIONA = torch.tensor(property_data.iloc[1, :].values, dtype=torch.long)
mw = torch.tensor(property_data.iloc[2, :].values, dtype=torch.float32)

file_path_x = "./input_train_pilot.csv"
file_path_y = "./targets_train_pilot.csv"

x = pd.read_csv(file_path_x, header=None)
y = pd.read_csv(file_path_y, header=None)


def sample_data(x, y, num_samples, random_seed):
    assert len(x) == len(y), "x and y must have the same length"

    np.random.seed(random_seed)
    indices = np.random.choice(len(x), size=num_samples, replace=False)
    x_sampled = x[indices]
    y_sampled = y[indices]

    return x_sampled, y_sampled

input1 = x.iloc[:, :4].values
input2 = x.iloc[:, 4:].values
targets = y.values

scaler = joblib.load('./scaler_input1_pilot_60.pkl')
input1_tensor, targets_tensor, _ = preprocess_data(input1, targets, scaler)
scaler = joblib.load('./scaler_input2_pilot_60.pkl')
input2_tensor, targets_tensor, _ = preprocess_data(input2, targets, scaler)

input1_tensor, input2_tensor, targets_tensor = input1_tensor.to(device), input2_tensor.to(device), targets_tensor.to(
    device)

block_num_net1 = 2
block_num_net2 = 2
block_num_agg = 3
hidden_size_net1 = 128
hidden_size_net2 = 512
hidden_size_agg = 256
dropout_p = 0.04589796813227005
unfrozen_layers = 12

net1 = Net1(input1_tensor.shape[1], block_num_net1, hidden_size_net1, dropout_p)
net2 = Net2(input2_tensor.shape[1], block_num_net2, hidden_size_net2, dropout_p)
agg_net = AggregateNet(hidden_size_net1 + hidden_size_net2, 129, block_num_agg, hidden_size_agg,
                       dropout_p)

checkpoint_net1 = torch.load('./model_checkpoint_net1.pth')
checkpoint_net2 = torch.load('./model_checkpoint_net2.pth')
checkpoint_agg_net = torch.load('./model_checkpoint_agg_net.pth')
net1.load_state_dict(checkpoint_net1['model_state_dict'])
net2.load_state_dict(checkpoint_net2['model_state_dict'])
agg_net.load_state_dict(checkpoint_agg_net['model_state_dict'])

for param in net1.parameters():
    param.requires_grad = False
for param in net2.parameters():
    param.requires_grad = False
for param in agg_net.parameters():
    param.requires_grad = False

for param in net1.parameters():
    param.requires_grad = False
for param in net2.parameters():
    param.requires_grad = False
for param in agg_net.parameters():
    param.requires_grad = False


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


model_paths = [
    './model_checkpoint_pilot_net1_prop_60.pth',
    './model_checkpoint_pilot_net2_prop_60.pth',
    './model_checkpoint_pilot_agg_net_prop_60.pth'
]

optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(net1.parameters()) + list(net2.parameters()) +
                              list(agg_net.parameters())))
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

    mass_fraction, mass_PIONA = compute_fractions(predict_data, BP, PIONA, mw, device=device)
    combined_output = torch.cat((mass_fraction, mass_PIONA), dim=1)

    predict_data = combined_output.cpu().numpy()

experiment_data = targets_tensor.cpu().numpy()
r2 = r2_score(experiment_data, predict_data)
mae = mean_absolute_error(experiment_data, predict_data)
mse = mean_squared_error(experiment_data, predict_data)

print(f"R-squared: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

plt.figure()
plt.scatter(targets_tensor.cpu().numpy(), predict_data, label='Data', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Train Set Fitting')
plt.legend()
plt.show()
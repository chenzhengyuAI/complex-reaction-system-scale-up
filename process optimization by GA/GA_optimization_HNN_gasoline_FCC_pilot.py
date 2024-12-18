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

file_path_x = "./FeedMoleculeContent_pilot_input.csv"
file_path_y = "./ProductBulkProperty_pilot_output.csv"

x = pd.read_csv(file_path_x, header=None, nrows=15)
y = pd.read_csv(file_path_y, header=None, nrows=15)

input1 = x.iloc[:, :4].values
input2 = x.iloc[:, 4:].values
targets = y.values

pop_size = 50
input2_first_row = input2[0, :]
input2_repeated = np.tile(input2_first_row, (pop_size, 1))

scaler = joblib.load('./scaler_input1_pilot_60.pkl')
input1_tensor, targets_tensor, input1_scaler = preprocess_data(input1, targets, scaler)
scaler = joblib.load('./scaler_input2_pilot_60.pkl')
input2_tensor, targets_tensor, input2_scaler = preprocess_data(input2_repeated, targets, scaler)

input1_tensor, input2_tensor = input1_tensor.to(device), input2_tensor.to(device)

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
agg_net = AggregateNet(hidden_size_net1 + hidden_size_net2, 129, block_num_agg, hidden_size_agg, dropout_p)

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


net1, optimizer = load_model(net1, optimizer, model_paths[0])
net2, optimizer = load_model(net2, optimizer, model_paths[1])
agg_net, optimizer = load_model(agg_net, optimizer, model_paths[2])


class MyOptimizationProblem(Problem):
    def __init__(self, net1, net2, agg_net, lower_bounds, upper_bounds, input1_shape, input2_tensor, BP, PIONA, mw,
                 device,
                 input1_original):
        super().__init__(n_var=input1_shape,
                         n_obj=2,
                         n_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds)

        self.net1 = net1
        self.net2 = net2
        self.agg_net = agg_net
        self.fixed_input2 = input2_tensor
        self.BP = BP
        self.PIONA = PIONA
        self.mw = mw
        self.device = device

    def _evaluate(self, x, out, *args, **kwargs):
        input1_dim = self.net1.layers[0].in_features
        input2_dim = self.net2.layers[0].in_features

        input1_tensor = torch.zeros((x.shape[0], input1_dim), dtype=torch.float32).to(self.device)

        input1_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        input2_tensor = self.fixed_input2.clone().detach().to(self.device)

        net1.eval()
        net2.eval()
        agg_net.eval()

        with torch.no_grad():
            output1 = self.net1(input1_tensor)
            output2 = self.net2(input2_tensor)
            predict_data = self.agg_net(output1, output2)

            mass_fraction, mass_PIONA = compute_fractions(predict_data, self.BP, self.PIONA, self.mw,
                                                          device=self.device)
            combined_output = torch.cat((mass_fraction, mass_PIONA), dim=1)

        f1 = -combined_output[:, 2].cpu().numpy()  # 第一个目标
        f2 = -combined_output[:, 6].cpu().numpy()  # 第二个目标

        out["F"] = np.column_stack([f1, f2])

lower_bounds = torch.min(input1_tensor, dim=0).values.cpu().numpy()
upper_bounds = torch.max(input1_tensor, dim=0).values.cpu().numpy()

problem = MyOptimizationProblem(
    net1, net2, agg_net,
    lower_bounds, upper_bounds,
    input1_tensor.shape[1], input2_tensor,
    BP, PIONA, mw, device,
    input1_original=input1_tensor[0].cpu().numpy()
)

algorithm = NSGA2(pop_size=pop_size)

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=True)

plt.figure()
plt.scatter(-res.F[:, 0], -res.F[:, 1], c="r", label="Pareto Front")
plt.xlabel("Objective 1 (Maximized)")
plt.ylabel("Objective 2 (Minimized)")
plt.title("NSGA-II Optimization")
plt.legend()
plt.show()

optimal_inputs = res.X
optimal_obj_values = res.F

input1_optimized = optimal_inputs[:, :input1_tensor.shape[1]]
input1_restored = input1_scaler.inverse_transform(input1_optimized)
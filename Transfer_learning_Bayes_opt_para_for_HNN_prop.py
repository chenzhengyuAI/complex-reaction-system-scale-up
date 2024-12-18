import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna import create_study, Trial, TrialPruned
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from Bulk_prop import compute_fractions

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
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = x1.clone().detach()
        x2 = x2.clone().detach()
        x = torch.cat((x1, x2), dim=1)
        out = self.layers(x)
        out = self.relu(out)
        return out


def optimize(trial):
    block_num_net1 = 2
    block_num_net2 = 2
    block_num_agg = 3
    hidden_size_net1 = 128
    hidden_size_net2 = 512
    hidden_size_agg = 256
    dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128, 256])
    unfrozen_layers = 12  # trial.suggest_int('unfrozen_layers', 1, 18)

    net1 = Net1(input1_train.shape[1], block_num_net1, hidden_size_net1, dropout_p)
    net2 = Net2(input2_train.shape[1], block_num_net2, hidden_size_net2, dropout_p)
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

    train_dataset = TensorDataset(input1_train, input2_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(net1.parameters()) + list(net2.parameters()) +
                                  list(agg_net.parameters())), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.MSELoss()

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

            mass_fraction, mass_PIONA = compute_fractions(final_output, BP, PIONA, mw, device=device)

            combined_output = torch.cat((mass_fraction, mass_PIONA), dim=1)

            loss = criterion(combined_output, targets_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net1.eval()
    net2.eval()
    agg_net.eval()
    test_loader = DataLoader(TensorDataset(input1_test, input2_test, targets_test), batch_size=batch_size)
    with torch.no_grad():
        test_predictions = torch.empty((0, 10), device=device)
        test_targets = torch.empty((0, 10), device=device)
        for inputs1_batch, inputs2_batch, targets_batch in test_loader:
            inputs1_batch, inputs2_batch, targets_batch = inputs1_batch.to(device), inputs2_batch.to(
                device), targets_batch.to(device)
            output1 = net1(inputs1_batch)
            output2 = net2(inputs2_batch)
            final_output = agg_net(output1, output2)

            mass_fraction, mass_PIONA = compute_fractions(final_output, BP, PIONA, mw, device=device)

            combined_output = torch.cat((mass_fraction, mass_PIONA), dim=1)

            test_predictions = torch.cat((test_predictions, combined_output), dim=0)
            test_targets = torch.cat((test_targets, targets_batch), dim=0)

        test_predictions = test_predictions.cpu()
        test_targets = test_targets.cpu()

    r2 = r2_score(test_targets.numpy(), test_predictions.numpy())
    mae = mean_absolute_error(test_targets.numpy(), test_predictions.numpy())

    return r2


file_path_x = "./FeedMoleculeContent_pilot_prop.csv"
file_path_y = "./ProductBulkProperty_pilot_prop.csv"

x = pd.read_csv(file_path_x, header=None)
y = pd.read_csv(file_path_y, header=None)

input1 = x.iloc[:, :4].values
input2 = x.iloc[:, 4:].values
targets = y.values


def sample_data(x, y, targets, num_samples, random_seed):
    assert len(x) == len(y), "x and y must have the same length"

    np.random.seed(random_seed)
    indices = np.random.choice(len(x), size=num_samples, replace=False)
    x_sampled = x[indices]
    y_sampled = y[indices]
    targets_sampled = targets[indices]

    return x_sampled, y_sampled, targets_sampled


input1, input2, targets = sample_data(input1, input2, targets, num_samples=60, random_seed=42)

input1_train, input1_test, input2_train, input2_test, targets_train, targets_test = train_test_split(
    input1, input2, targets, test_size=0.2, random_state=42
)

input1_train, targets_train = preprocess_data(input1_train, targets_train)
input2_train, targets_train = preprocess_data(input2_train, targets_train)
input1_test, targets_test = preprocess_data(input1_test, targets_test)
input2_test, targets_test = preprocess_data(input2_test, targets_test)

input1_train, input1_test = input1_train.to(device), input1_test.to(device)
input2_train, input2_test = input2_train.to(device), input2_test.to(device)
targets_train, targets_test = targets_train.to(device), targets_test.to(device)


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])


property_data = pd.read_csv('./mol_property.csv', header=None)
BP = torch.tensor(property_data.iloc[0, :].values, dtype=torch.float32)
PIONA = torch.tensor(property_data.iloc[1, :].values, dtype=torch.long)
mw = torch.tensor(property_data.iloc[2, :].values, dtype=torch.float32)

study = create_study(direction='maximize')
study.optimize(optimize, n_trials=100)

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

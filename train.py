import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
from torch_geometric.loader import DataLoader


class Ksigmoid(nn.Module):
    def __init__(self, num_parameters=1, init_k=1.0):
        super(Ksigmoid, self).__init__()
        self.k = nn.Parameter(torch.full((num_parameters,), init_k))

    def forward(self, x):
        # 计算k * sigmoid(x)
        return self.k * torch.sigmoid(x)
Ksigmoid = Ksigmoid()

class Ktanh(nn.Module):
    def __init__(self, num_parameters=1, init_k=1.0):
        super(Ktanh, self).__init__()
        self.k = nn.Parameter(torch.full((num_parameters,), init_k))

    def forward(self, x):
        return self.k * torch.tanh(x)

Ktanh = Ktanh()

class Kprelu(nn.Module):
    def __init__(self, num_parameters=1, init_k=1.0, init_alpha=0.25):
        super(Kprelu, self).__init__()
        self.k = nn.Parameter(torch.full((num_parameters,), init_k))
        self.alpha = nn.Parameter(torch.full((num_parameters,), init_alpha))
    def forward(self, x):
        return torch.where(x > 0, self.k * x, self.alpha * x)

Kprelu = Kprelu()

class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_gcn_layers, num_ff_layers, num_heads, norm_type, dropout,
                 activation, pool_type):
        super(GNNModel, self).__init__()
        self.num_heads = num_heads
        self.num_gcn_layers = num_gcn_layers
        self.num_ff_layers = num_ff_layers
        self.pool_type = pool_type
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation_func = activation

        # 多头自注意力层：GAT层
        head_dim = max(10, in_channels // (2 * self.num_heads))
        self.gat = GATConv(in_channels, head_dim, heads=self.num_heads, concat=True)

        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = head_dim * self.num_heads  # GAT的输出维度
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(GCNConv(current_dim, current_dim))  # 保持维度不变
            if norm_type == 'BN':
                self.norms.append(nn.BatchNorm1d(current_dim))
            elif norm_type == 'LN':
                self.norms.append(nn.LayerNorm(current_dim))

        if self.activation_func == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_func == 'Kprelu':
            self.activation = nn.PReLU()
        elif self.activation_func == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif self.activation_func == 'elu':
            self.activation = nn.ELU()
        elif self.activation_func == 'selu':
            self.activation = nn.SELU()
        elif self.activation_func == 'gelu':
            self.activation = nn.GELU()
        elif self.activation_func == 'softplus':
            self.activation = nn.Softplus()
        elif self.activation_func == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_func == 'softsign':
            self.activation = nn.Softsign()
        elif self.activation_func == 'prelu':
            self.activation = nn.SiLU()
        elif self.activation_func == 'swish':
            self.activation = nn.SiLU()
        elif self.activation_func == 'Kprelu':
            self.activation = Kprelu()
        elif self.activation_func == 'Ksigmoid':
            self.activation = Ksigmoid()
        elif self.activation_func == 'Kprelu':
            self.activation = Ktanh()

        # 前馈神经网络层
        self.ff_layers = nn.ModuleList()
        for _ in range(self.num_ff_layers):
            self.ff_layers.append(nn.Linear(current_dim, current_dim // 2))
            current_dim //= 2

        # 输出层
        self.fc = nn.Linear(current_dim, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT层
        x = self.gat(x, edge_index)
        x_res = x

        # GCN层
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        # 选择池化方法
        if self.pool_type == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, data.batch)


        for ff in self.ff_layers:
            x = ff(x)
            x = self.activation(x)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_and_evaluate_single_combination(params, data_list, kf):
    # 解压参数
    num_layers, num_ff_layers, num_heads, activation, norm_type, dropout, pool_type = params
    results = []

    # 确保使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_mcc = -1
    best_epoch = -1
    best_model_state = None

    for train_idx, test_idx in kf.split(data_list):
        train_data = [data_list[int(idx)].to(device) for idx in train_idx]
        test_data = [data_list[int(idx)].to(device) for idx in test_idx]

        # 创建 DataLoader
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        in_channels = train_data[0].x.size(1)
        # 初始化模型
        model = GNNModel(in_channels=in_channels,
                         out_channels=2,
                         num_gcn_layers=num_layers,
                         num_ff_layers=num_ff_layers,
                         num_heads=num_heads,
                         norm_type=norm_type,
                         dropout=dropout,
                         activation=activation,
                         pool_type=pool_type).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(100):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()

            model.eval()
            y_true, y_pred = [], []
            for data in test_loader:
                data = data.to(device)
                with torch.no_grad():
                    out = model(data)
                    pred = out.argmax(dim=1)
                    y_true.extend(data.y.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            SN = tp / (tp + fn) if (tp + fn) > 0 else 0
            SP = tn / (tn + fp) if (tn + fp) > 0 else 0
            ACC = (tp + tn) / (tp + tn + fp + fn)
            MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (
                        tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
            ROC = roc_auc_score(y_true, y_pred)

            if MCC > best_mcc:
                best_mcc = MCC
                best_epoch = epoch
                best_model_state = model.state_dict()

        # 记录每个组合的结果
        avg_results = {
            'num_layers': num_layers,
            'num_ff_layers': num_ff_layers,
            'num_heads': num_heads,
            'activation': activation,
            'norm_type': norm_type,
            'dropout': dropout,
            'pool_type': pool_type,
            'SN': SN,
            'SP': SP,
            'ACC': ACC,
            'MCC': MCC,
            'ROC': ROC,
            'best_epoch': best_epoch,
            'best_mcc': best_mcc
        }
        results.append(avg_results)

    return results, best_model_state

# 执行网格搜索
def grid_search_for_each_file():
    kf = KFold(n_splits=5, shuffle=True)
    activations = ['relu', 'Kprelu',
                   'leaky_relu', 'elu',
                   'selu', 'gelu', 'softplus',
                   'tanh', 'sigmoid',
                   'softsign', 'prelu', 'swish',
                   'Kprelu',"Ksigmoid",'Ktanh']
    norm_types = ['BN', 'LN']
    pool_types = ['mean','max']
    layer_ranges = range(1, 6)
    ff_layers_range = range(1, 6)
    heads_range = [1, 2, 4, 6, 8, 10, 12]
    dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    param_combinations = list(itertools.product(layer_ranges, ff_layers_range, heads_range, activations, norm_types, dropout_rates, pool_types))

    pt_files = [f for f in os.listdir('Select') if f.endswith('.pt')]

    for pt_file in pt_files:

        data_list1 = torch.load(os.path.join('Select', pt_file))
        data_list = []
        for raw_data in data_list1:
            data = Data(x=raw_data['x'], edge_index=raw_data['edge_index'], y=raw_data['y'])
            if isinstance(data.y, int):
                data.y = torch.tensor([data.y], dtype=torch.long)
            data_list.append(data)

        result_folder = os.path.join('Results', os.path.splitext(pt_file)[0])
        os.makedirs(result_folder, exist_ok=True)

        for params in tqdm(param_combinations):
            results, best_model_state = train_and_evaluate_single_combination(params, data_list, kf)
            if max(r['MCC'] for r in results) > 0.70:
                print(max(r['MCC'] for r in results))
                model_name = f"model_layers{params[0]}_fflayers{params[1]}_heads{params[2]}_" \
                             f"act{params[3]}_norm{params[4]}_dropout{params[5]}_pool{params[6]}_" \
                             f"MCC_{max(r['MCC'] for r in results):.4f}.pt"
                model_path = os.path.join(result_folder, model_name)
                torch.save(best_model_state, model_path)
        print(f"Finished processing file: {pt_file}")


if __name__ == "__main__":
    grid_search_for_each_file()

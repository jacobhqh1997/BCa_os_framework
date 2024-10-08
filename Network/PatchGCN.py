import torch
import torch.nn as nn

from torch.nn import ReLU, LayerNorm
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from network.model_utils import weight_init

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def reset_parameters(self):

        self.attention_a.apply(weight_init)
        self.attention_b.apply(weight_init)
        self.attention_c.apply(weight_init)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class PatchGCN_module(torch.nn.Module):

    def __init__(self, hidden_dim, i, dropout_rate):

        super(PatchGCN_module, self).__init__()
        self.conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                       t=1.0, learn_t=True, num_layers=2, norm='layer')
        self.norm = LayerNorm(hidden_dim, elementwise_affine=True)
        self.act = ReLU(inplace=True)
        self.layer = DeepGCNLayer(self.conv, self.norm, self.act, block='res', dropout=0.1, ckpt_grad=i % 3)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):

        self.layer.reset_parameters()

    def forward(self, x, edge_index):

        drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
        drop_node_mask = torch.bernoulli(drop_node_mask)
        drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
        drop_node_feature = x * drop_node_mask

        x_after = self.layer(drop_node_feature, edge_index)

        return x_after

class PatchGCN(torch.nn.Module):

    def __init__(self, Argument):
        super(PatchGCN, self).__init__()

        hidden_dim = Argument['initial_dim'] * Argument['attention_head_num']
        self.num_layers = Argument['number_of_layers']
        dropout = Argument['dropout_rate']

        # self.fc = nn.Sequential(*[nn.Linear(3584, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(3072, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(4608, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(2816, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])


        self.fc = nn.Sequential(*[nn.Linear(1536, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(1024, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(2560, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        # self.fc = nn.Sequential(*[nn.Linear(768, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])


        self.total_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.total_layers.append(PatchGCN_module(hidden_dim, i, dropout))

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 4, D=hidden_dim * 4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])

        self.risk_prediction_layer = nn.Linear(hidden_dim * 4, 1)

    def reset_parameters(self):

        self.fc.apply(weight_init)
        for i in range(len(self.total_layers)):
            self.total_layers[i].reset_parameters()
        self.path_phi.apply(weight_init)
        self.path_rho.apply(weight_init)
        self.risk_prediction_layer.apply(weight_init)
        self.path_attention_head.reset_parameters()

    def forward(self, data):

        x = self.fc(data.x)
        x_ = x

        edge_index = data.edge_index
        edge_attr = None
        batch = data.batch

        x = self.total_layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.total_layers[1:]:
            x = layer(x, edge_index)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = scatter_add(torch.mul(h_path.permute(1,0), softmax(A_path.flatten(), batch)).permute(1,0), batch, dim=0)
        h = self.path_rho(h_path).squeeze()
        h = self.risk_prediction_layer(h).flatten()

        return h, A_path
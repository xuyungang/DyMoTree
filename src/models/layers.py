import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import Linear
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    to_undirected,
)
from typing import Optional

class GraphAttention_layer(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 flow: str = 'source_to_target',
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = False,
                 to_undirected: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', flow)
        super(GraphAttention_layer, self).__init__(node_dim=0, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.to_undirected = to_undirected

        self.lin_l = Linear(input_dim, heads * output_dim, bias=False, weight_initializer='glorot')
        self.lin_r = Linear(input_dim, heads * output_dim, bias=False, weight_initializer='glorot')

        if concat:
            self.bias = nn.Parameter(Tensor(heads * output_dim))
            self.weight_concat = nn.Parameter(Tensor(heads * output_dim, output_dim))
        else:
            self.bias = nn.Parameter(Tensor(output_dim))
            self.register_parameter('weight_concat', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        zeros(self.bias)
        if self.concat:
            glorot(self.weight_concat)

    def forward(self, x: Tensor, edge_index: Adj, return_attention_weights: Optional[bool] = None):
        N, H, C = x.size(0), self.heads, self.output_dim

        if self.to_undirected:
            edge_index = to_undirected(edge_index)
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        else:
            edge_index, _ = remove_self_loops(edge_index)

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        x_norm_l = F.normalize(x_l, p=2., dim=-1)
        x_norm_r = F.normalize(x_r, p=2., dim=-1)
        out = self.propagate(edge_index, x=(x_l, x_r), x_norm=(x_norm_l, x_norm_r), size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.output_dim)
            out += self.bias
            out = torch.matmul(out, self.weight_concat)
        else:
            out = out.mean(dim=1)
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index_j: Tensor, x_i: Tensor, x_j: Tensor,
                x_norm_i: Optional[Tensor], x_norm_j: Optional[Tensor],
                size_i: Optional[int]):
        alpha = torch.exp((x_norm_i * x_norm_j).sum(dim=-1))
        Tau = 0.25 # temperature hyperparameter
        self._alpha = alpha
        alpha = softmax(alpha / Tau, edge_index_j, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
class GravityDecoder(torch.nn.Module):
    def __init__(self, lamda: float = 1.0):
        super(GravityDecoder, self).__init__()
        self.lamda = lamda
        
    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True):
        _z_dim = z.shape[1]
        dist = torch.norm(z[edge_index[0], 0:(_z_dim - 1)] - z[edge_index[1], 0:(_z_dim - 1)], dim=1, keepdim=True) + 1e-10
        outputs = z[edge_index[1], (_z_dim - 1):_z_dim] - self.lamda * torch.log(dist**2)
        return torch.sigmoid(outputs) if sigmoid else outputs

class Cross_GravityDecoder(torch.nn.Module):
    def __init__(self, lamda: float = 1.0):
        super(Cross_GravityDecoder, self).__init__()
        self.lamda = lamda
        
    def forward(self, z_source: Tensor, z_child: Tensor, edge_index: Tensor, sigmoid: bool = True):
        outputs = (z_source[edge_index[0]] * z_child[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(outputs) if sigmoid else outputs
    
class GraphEncoder(torch.nn.Module):  
    def __init__(self, in_channels: int, out_channels: int, hidden: int, hidden1: int, hidden2: int,
                 flow: str ='source_to_target', head: int =1, concat: bool = True,
                 add_self_loops: bool = False, to_undirected: bool = False, dropout: float = 0.0):
        super(GraphEncoder, self).__init__()
        self.x_input = nn.Linear(in_channels, hidden)

        self.gat1 = GraphAttention_layer(input_dim=hidden, output_dim=hidden1, flow=flow, heads=head,
                        concat=concat, dropout=dropout, add_self_loops=add_self_loops, to_undirected=to_undirected)

        self.gat2 = GraphAttention_layer(input_dim=hidden1, output_dim=hidden2, flow=flow, heads=head,
                        concat=concat, dropout=dropout, add_self_loops=add_self_loops, to_undirected=to_undirected)

        self.linear = nn.Sequential(
                    nn.Linear(hidden2, hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2, out_channels),
                )  
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, edge_index):
        x = self.x_input(x)
        x = self.act(x)
        
        x, att_1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.act(x)

        x, att_2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = self.act(x)
        
        z = self.linear(x)
        att_r = (att_1[1] + att_2[1]) / 2
        self.att = (att_1[0], att_r)

        return z

# [Fix] P0: 将原 LineageModule 重命名为 CrossAttentionModule 以对齐其他代码调用
class LineageModule(nn.Module):
    def __init__(self, terminal: str, progenitor: str, mlp_dict: dict, feature_dim: int):
        super(LineageModule, self).__init__()
        self.feature_dim = feature_dim
        self.terminal = terminal
        self.progenitor = progenitor

        self.linear = nn.Linear(feature_dim, feature_dim)
        self.query_linear = mlp_dict[self.terminal]['Q']
        self.key_linear = mlp_dict[self.progenitor]['K']
        self.output_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, q: Tensor, k: Tensor):
        Q = self.query_linear(q)  
        K = self.key_linear(k)    

        _dot_res = torch.matmul(Q, K.T)
        _mean = _dot_res.mean()
        _std = _dot_res.std()
        transition_matrix = torch.sigmoid(((_dot_res - _mean) / (_std / 1.0)))
        self.att = transition_matrix

        return transition_matrix


class Fusion_block(nn.Module):
    def __init__(self, terminal: str, progenitor: str):
        super(Fusion_block,self).__init__()
        self.terminal = terminal
        self.progenitor = progenitor
        
    def forward(self, z: Tensor, att: Tensor):
        attention_weights = att / (att.sum(dim=1, keepdim=True))
        return torch.matmul(attention_weights, z)
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import torch.optim as optim
from tqdm import trange

""" 
CellModule
"""

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

# GAE
class GAE(torch.nn.Module):
    """The Graph Auto-Encoder model."""
    def __init__(self, node_name, progenitor, encoder, decoder=None):
        super().__init__()
        self.name = node_name
        self.progenitor = progenitor
        self.z = None

        self.encoder = encoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        # [Fix] P0: 直接调用 decoder，去除了原先错误的 ['graph'] 字典索引
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decode(z, pos_edge_index, sigmoid=True) + 1e-10).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decode(z, neg_edge_index, sigmoid=True) +
                              1e-10).mean()
        g_loss = pos_loss + neg_loss

        return g_loss

    def pretrain(self, data, edge_index, iter, lr):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        with trange(len(range(iter)), ncols=100) as t:
            for _i in t:
                optimizer.zero_grad()
                z = self.encode(data, edge_index)
                graph_loss = self.recon_loss(z, edge_index)
                loss = graph_loss
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())
        
# VGAE
class VGAE(GAE):
    def __init__(self, node_name, parent_name, encoder, decoder=None):
        super().__init__(node_name, parent_name, encoder, decoder)

    def reparametrize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        self.z = z
        return z
    
    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
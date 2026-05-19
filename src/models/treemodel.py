import torch
import torch.nn as nn
import math
import copy


from .layers import GraphEncoder, Fusion_block, GravityDecoder, InnerProductDecoder, Cross_GravityDecoder
from .layers import LineageModule as CAM
from .cellmodule import GAE as nodemodel

from utils import bipartite_degree_aware_sampler as degree_sampler


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TreeModel(nn.Module):
    def __init__(self,
                treedata,
                embedding_dim,
                intra=0.5,
                inter=0.5,
                W=0.0,
                flow='source_to_target',
                heads=1,
                concat=True,
                add_self_loops=False,
                to_undirected=False,
                dropout=0.0,
                fusion_type='mix',
                device='cpu'):
        super(TreeModel, self).__init__()
        
        self.root_node = treedata.progenitor
        
        self.tree = self.make_tree(progenitor=treedata.progenitor, terminal=treedata.terminal)
        
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.input_dim = treedata.get_node(self.root_node, adata_object=False).data.shape[1]
        self.add_self_loops = add_self_loops
        self.to_undirected = to_undirected
        self.flow = flow
        self.intra = intra
        self.inter = inter
        self.W = W
        self.fusion_type = fusion_type
        self.device = device

        self.node_model = self.make_model(embedding_dim)
        self.node_model.to(self.device)
        
        self.att_mlp = {}
        self.make_att_mlp(node=self.root_node, embedding_dim=embedding_dim)
        self.att_mlp = nn.ModuleDict(self.att_mlp).to(self.device)        
        
        self.cross_att = {}
        self.fusion_block = {}
        self.make_fusion_block(self.root_node, embedding_dim)
        self.fusion_block = nn.ModuleDict(self.fusion_block).to(self.device)
        self.cross_att = nn.ModuleDict(self.cross_att).to(self.device)

        self.cross_decoder = Cross_GravityDecoder()

    def forward(self, treedata, train_mode='pre'):
        if train_mode != 'pre':
            cell_list = self.order_r(self.root_node, self.tree)
            att_dict = {}  
            z_dict = {}    
            for cell in cell_list.keys():
                node = treedata.get_node(cell, adata_object=False)
                z_dict_ = {}
                if self.tree[cell]['ancestor'] is None:
                    z = self.node_model[cell].encode(node.data, node.edge)
                    z_dict_['orig_z'] = z
                    z_dict[cell] = z_dict_
                else:
                    parent_node = self.tree[cell]['ancestor']
                    z = self.node_model[cell].encode(node.data, node.edge)
                    z_dict_['orig_z'] = z
                    if self.tree[parent_node]['ancestor'] is None:
                        parent_z = z_dict[parent_node]['orig_z']
                    else:
                        parent_z = z_dict[parent_node]['fused_z']
                        
                    att = self.cross_att[f'{parent_node}->{cell}'](z, parent_z)
                    fusion_feature = self.fusion_block[f'{parent_node}->{cell}'](parent_z, att)
                    z_dict_['fused_z'] = fusion_feature
                    z_dict[cell] = z_dict_
                    att_dict[f'{parent_node}->{cell}'] = att
            return att_dict, z_dict
        else:
            cell_list = self.order_r(self.root_node, self.tree)
            z_dict = {}    
            for cell in cell_list.keys():
                node = treedata.get_node(cell, adata_object=False)
                z = self.node_model[cell].encode(node.data, node.edge)
                z_dict[cell] = z
            return z_dict
    
    def make_tree(self, progenitor: str, terminal: list):
        if not isinstance(progenitor, str):
            raise TypeError("progenitor must be a str")
        if not isinstance(terminal, list):
            raise TypeError("terminal must be a list")

        terminals = list(dict.fromkeys(terminal))
        if progenitor in terminals:
            raise ValueError("progenitor should not appear in terminal")

        tree = {
            progenitor: {
                'ancestor': None,
                'descendant': terminals
            },
            **{
                t: {
                    'ancestor': progenitor,
                    'descendant': []
                }
                for t in terminals
            }
        }
        return tree
    
    def make_model(self,embedding_dim):
        node_models = {}
        for node_name in self.tree.keys():
            encoder = GraphEncoder(in_channels=self.input_dim,
                               out_channels=embedding_dim,
                               hidden=self.input_dim,
                               hidden1=512, hidden2=256, flow=self.flow, head=self.heads,
                               concat=self.concat, add_self_loops=self.add_self_loops,
                               to_undirected=self.to_undirected, dropout=0.0)
            if self.to_undirected:
                decoder = InnerProductDecoder()
            else:
                decoder = GravityDecoder()
                
            parent_name = 'None' if self.tree[node_name]['ancestor'] is None else self.tree[node_name]['ancestor']
            node_models[node_name] = nodemodel(node_name=self.tree[node_name], progenitor=parent_name, encoder=encoder, decoder=decoder)
            
        return nn.ModuleDict(node_models)
    
    def make_att_mlp(self, node, embedding_dim):
        if self.tree[node]['ancestor'] is None:
            mlp_dict = {'K': nn.Linear(embedding_dim, embedding_dim)}
            self.att_mlp[node] = nn.ModuleDict(mlp_dict)
            for child in self.tree[node]['descendant']:
                self.make_att_mlp(child, embedding_dim)
        elif not self.tree[node]['descendant']:
            mlp_dict = {'Q': nn.Linear(embedding_dim, embedding_dim)}
            self.att_mlp[node] = nn.ModuleDict(mlp_dict)
        else:
            mlp_dict = {
                'Q': nn.Linear(embedding_dim, embedding_dim),
                'K': nn.Linear(embedding_dim, embedding_dim)
            }
            self.att_mlp[node] = nn.ModuleDict(mlp_dict)
            for child in self.tree[node]['descendant']:
                self.make_att_mlp(child, embedding_dim)
    
    def make_fusion_block(self, node, embedding_dim):
        if self.tree[node]['ancestor'] is None:
            for child in self.tree[node]['descendant']:
                self.make_fusion_block(child, embedding_dim)
        else:
            parent = self.tree[node]['ancestor']
            self.cross_att[f'{parent}->{node}'] = CAM(node, parent, self.att_mlp, embedding_dim).to(self.device)
            self.fusion_block[f'{parent}->{node}'] = Fusion_block(node, parent).to(self.device)
            for child in self.tree[node]['descendant']:
                self.make_fusion_block(child, embedding_dim)     

    def pre_loss(self, treedata, z_dict,sample_ratio=1024):
        total_loss = []
        node_loss = []
        for cell in z_dict.keys():
            node_loss_ = self.node_model[cell].recon_loss(z_dict[cell], treedata.get_node(cell, adata_object=False).edge)
            node_loss.append(node_loss_)
        node_loss = torch.mean(torch.stack(node_loss))
        total_loss.append(node_loss)   

        cross_loss = []
        for cell in z_dict.keys():
            if self.tree[cell]['ancestor'] is not None:
                parent = self.tree[cell]['ancestor']
                pos_edge = treedata.get_lineage_pairs(cell,'pos_edge')
                neg_edge = treedata.get_lineage_pairs(cell,'neg_edge')

                num_pos_edges = pos_edge.shape[1]
                num_neg_edges = neg_edge.shape[1]
                num_edges = min(num_pos_edges, num_neg_edges)
                sample_size = math.ceil(num_edges / sample_ratio)
                
                # [Fix] P2: 使用 torch.randint 替代 torch.randperm 以大幅降低前向传播时 CPU/GPU 同步开销
                sampled_pos_indices = torch.randint(0, num_pos_edges, (sample_size,), device=self.device)
                sampled_neg_indices = torch.randint(0, num_neg_edges, (sample_size,), device=self.device)
                
                #pos_edge = pos_edge[:, sampled_pos_indices]  
                #neg_edge = neg_edge[:, sampled_neg_indices]  
                pos_batch, neg_batch = degree_sampler(
                    pos_edge,
                    neg_edge,
                    batch_size=sample_ratio,
                    alpha=0
                )                     
                pos_loss = -torch.log(self.cross_decoder(z_dict[parent], z_dict[cell], pos_batch, sigmoid=True) + 1e-10).mean()
                neg_loss = -torch.log(1 - self.cross_decoder(z_dict[parent], z_dict[cell], neg_batch, sigmoid=True) + 1e-10).mean()
                cross_loss.append(pos_loss + neg_loss)
                
        if cross_loss:
            cross_loss = torch.mean(torch.stack(cross_loss))
            total_loss.append(cross_loss)
            
        return torch.mean(torch.stack(total_loss))

    def tree_loss(self, treedata, z_dict,sample_ratio=1024):
        node_loss = []
        for cell in z_dict.keys():
            z_feat = z_dict[cell]['orig_z'] if self.tree[cell]['ancestor'] is None else z_dict[cell]['fused_z']
            node_loss.append(self.node_model[cell].recon_loss(z_feat, treedata.get_node(cell, adata_object=False).edge))
        node_loss = torch.mean(torch.stack(node_loss))
        
        cross_loss = []
        for cell in z_dict.keys():
            if self.tree[cell]['ancestor'] is not None:
                parent = self.tree[cell]['ancestor']
                pos_edge = treedata.get_lineage_pairs(cell,'pos_edge')
                neg_edge = treedata.get_lineage_pairs(cell,'neg_edge')
                
                num_pos_edges = pos_edge.shape[1]
                num_neg_edges = neg_edge.shape[1]
                num_edges = min(num_pos_edges, num_neg_edges)
                sample_size = math.ceil(num_edges / sample_ratio)

                # [Fix] P2: 使用 randint 优化下采样效率
                sampled_pos_indices = torch.randint(0, num_pos_edges, (sample_size,), device=self.device)
                sampled_neg_indices = torch.randint(0, num_neg_edges, (sample_size,), device=self.device)
                
                #pos_edge = pos_edge[:, sampled_pos_indices]  
                #neg_edge = neg_edge[:, sampled_neg_indices]         
                pos_batch, neg_batch = degree_sampler(
                    pos_edge,
                    neg_edge,
                    batch_size=sample_ratio,
                    alpha=0
                )                 
                parent_z = z_dict[parent]['orig_z'] if parent == self.root_node else z_dict[parent]['fused_z']
                cell_z = z_dict[cell]['fused_z']
                
                pos_loss = -torch.log(self.cross_decoder(parent_z, cell_z, pos_batch, sigmoid=True) + 1e-10).mean()
                neg_loss = -torch.log(1 - self.cross_decoder(parent_z, cell_z, neg_batch, sigmoid=True) + 1e-10).mean()
                cross_loss.append(pos_loss + neg_loss)
                
        cross_loss = torch.mean(torch.stack(cross_loss)) if cross_loss else 0.0
        
        return self.intra * node_loss + self.inter * cross_loss
    
    def pretrain(self, treedata, iter=100, lr=1e-3):
        for node_name in self.tree.keys():
            print(f':: Do pre-train Graph encoder for {node_name} ::')
            self.node_model[node_name].to(self.device)
            self.node_model[node_name].pretrain(
                data=treedata.get_node(node_name, adata_object=False).data,
                edge_index=treedata.get_node(node_name, adata_object=False).edge,
                iter=iter, lr=lr
            )

    def freeze_decoder_params(self):
        for name, param in self.named_parameters():
            if 'decoder' in name or 'encoder' in name:
                param.requires_grad = False

    # [Fix] P0: 修复可变默认参数导致的无限累加和递归引用 BUG
    def order_r(self, cell, tree, cell_dict=None):
        if cell_dict is None:
            cell_dict = {}
        current_cell = {cell: cell}
        cell_dict.update(current_cell)
        for desc in tree[cell]['descendant']:
            cell_dict = self.order_r(desc, tree, cell_dict)
        return cell_dict
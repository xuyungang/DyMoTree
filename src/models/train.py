"""
Training functions for DyMoTree models.
"""
from .treemodel import TreeModel
from data.treeloader import TreeDatasetLoader as TreeDataSet
from utils.set_seed import seed_all

import torch
import torch.optim as optim

import time
from datetime import datetime
from tqdm import trange

def train_model(treedata: TreeDataSet,
                      embedding_dim: int = 32,
                      intra: float = 0.5,
                      inter: float = 0.5,
                      flow: str = 'source_to_target',
                      heads: int = 1,
                      concat: bool = True,
                      add_self_loops: bool = False,
                      to_undirected: bool = False,
                      dropout: float = 0.0,
                      fusion_type: str = 'mix',
                      device: str = 'cuda',
                      pre_train: str = 'combined',
                      lr = {'formal':1e-4,
                            'intra':1e-3,
                            'lineage':1e-4},
                      iter = {'formal':300,
                            'intra':100,
                            'lineage':200},
                      sample_ratio: int = 5,
                      alpha: float = 0.5,
                      seed: int = 42):
    
    seed_all(seed)
    
    model = TreeModel(treedata,
                embedding_dim=embedding_dim,
                intra=intra,
                inter=inter,
                W=0.0,
                flow=flow,
                heads=heads,
                concat=concat,
                add_self_loops=add_self_loops,
                to_undirected=to_undirected,
                dropout=dropout,
                fusion_type=fusion_type,
                device=device)
    
    if pre_train == 'intra':
        
        print(':: Stage1 Pre-training -- intra-state transition graph::')
        
        model.pretrain(treedata=treedata,
                       lr=lr['intra'],
                       iter=iter['intra'])
        
    elif pre_train == 'lineage':
        
        print(':: Stage2 Pre-training -- lineage-graph::')
        
        optimizer = optim.Adam(model.parameters(), lr=lr['lineage']) 
        model.train()
        with trange(len(range(iter['lineage'])),ncols=100) as t:
            for it in t: 
                t.set_description('Iter: {}/{} '.format(it+1,iter['lineage'])) 
                optimizer.zero_grad()
                z_dict = model(treedata)
                loss = model.pre_loss(treedata,z_dict,sample_ratio=sample_ratio)
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())
                
    elif pre_train == 'combined':
        
        print(':: Stage1 Pre-training -- intra-state transition graph::')
        
        model.pretrain(treedata=treedata,
                       lr=lr['intra'],
                       iter=iter['intra'])  
        
        print(':: Stage2 Pre-training -- lineage-graph::')
        
        optimizer = optim.Adam(model.parameters(), lr=lr['lineage']) 
        model.train()
        with trange(len(range(iter['lineage'])),ncols=100) as t:
            for it in t: 
                t.set_description('Iter: {}/{} '.format(it+1,iter['lineage'])) 
                optimizer.zero_grad()
                z_dict = model(treedata)
                loss = model.pre_loss(treedata,z_dict,sample_ratio=sample_ratio)
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item())    
    
    print(':: DyMoTree Training::')

    optimizer = optim.Adam([
        {'params': (p for n in model.node_model for p in model.node_model[n].encoder.parameters()), 'lr': (lr['formal']/10.0)},
        {'params': (p for n in model.node_model for p in model.node_model[n].decoder.parameters()), 'lr': (lr['formal']/10.0)},
        {'params': model.fusion_block.parameters(), 'lr': lr['formal']},
        {'params': model.cross_att.parameters(), 'lr': lr['formal']}])   
    model.train()
    with trange(len(range(iter['formal'])),ncols=100) as t:
        for it in t: 
            t.set_description('Iter: {}/{} '.format(it+1,iter['formal'])) 
            optimizer.zero_grad()
            att_dict,z_dict = model(treedata,train_mode='core')
            loss = model.tree_loss(treedata,z_dict,sample_ratio=sample_ratio)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())
            
    return model
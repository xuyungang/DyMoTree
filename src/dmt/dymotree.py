"""
DyMoTree: Dynamic Tree Representation Learning with Temporal Graph Neural Networks
"""
from datetime import datetime, time
import pandas as pd
import numpy as np

from data.treeloader import TreeDatasetLoader as tree_loader, TreeNode
from models.train import train_model
from graph_builder.lineage_graph import make_lineage_graph
from utils.set_seed import seed_all

from downstream.diffaa import Find_State
from downstream.call_driver import Call_Driver as CD


class DyMoTree:
    def __init__(self, adata, k, progenitor, terminal, lineage_col, emb_key, seed,device):
        self.device = device
        self.adata = adata
        self.k = k
        self.seed = seed

        self.progenitor = progenitor
        self.terminal = terminal
        
        self.treedata = tree_loader(adata=adata,
                                    progenitor=progenitor,
                                    terminal=terminal, 
                                    lineage_col=lineage_col, 
                                    emb_key=emb_key,
                                    device=self.device)
        self.model = None

    def lineage_graph(self,mask_threshold=0.8, epsilon=1,mode = 'composite'):
        seed_all(self.seed)
        make_lineage_graph(self.treedata, k=self.k, mask_threshold=mask_threshold, epsilon=epsilon ,device=self.device, mode=mode)
        
    def get_node(self, celltype: str,adata_object: str = False) -> TreeNode:
        """
        retrieve the TreeNode object for a given cell state
        """
        node = self.treedata.nodes.get(celltype)
        if node is None:
            available = list(self.treedata.nodes.keys())
            raise KeyError(f"'{celltype}' not found. Available nodes: {available}")
        if adata_object:
            return node['adata']
        return node['data']
    
    def get_lineage_pairs(self,lineage: str,data:str) -> dict:
        """
        retrieve the data for a given lineaege pair (progenitor->terminal)
        """
        l_p = self.treedata.lineage_pairs.get(f"{self.progenitor}->{lineage}")
        if l_p is None:
            available = list(self.treedata.lineage_pairs.keys())
            raise KeyError(f"'{lineage}' not found. Available lineage pairs: {available}")
        d = l_p.get(data)
        if d is None:
            available_data = list(l_p.keys())
            raise KeyError(f"'{data}' not found in lineage pair '{self.progenitor}->{lineage}'. Available data: {available_data}")
        return d
    
    def get_model(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")
        return self.model
    
    def train(self,embedding_dim: int = 32,
              intra: float = 0.5,
              inter: float = 0.5,
              flow: str = 'source_to_target',
              heads: int = 1,
              concat: bool = True,
              add_self_loops: bool = False,
              to_undirected: bool = False,
              dropout: float = 0.0,
              fusion_type: str = 'mix',
              pre_train: str = 'combined',
              lr = {'formal':1e-4,
                    'intra':1e-3,
                    'lineage':1e-4},
              iter = {'formal':300,
                    'intra':100,
                    'lineage':200},
              sample_ratio: int = 5,
              alpha: float = 0.5):
        
        self.model = train_model(self.treedata,
                      embedding_dim=embedding_dim,
                      intra=intra,
                      inter=inter,
                      flow=flow,
                      heads=heads,
                      concat=concat,
                      add_self_loops=add_self_loops,
                      to_undirected=to_undirected,
                      dropout=dropout,
                      fusion_type=fusion_type,
                      device=self.device,
                      pre_train=pre_train,
                      lr = lr,
                      iter = iter,
                      sample_ratio=sample_ratio,
                      alpha=alpha,
                      seed=self.seed)
        
        p = self.treedata.get_node(self.progenitor,adata_object=True)
        for cell in self.model.tree.keys():
            if self.model.tree[cell]['descendant']:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] get fate space of {cell}") 
                self.model.eval()
        
                att_dict,_ = self.model(self.treedata,train_mode='core')
        
                fate_space = pd.DataFrame([att_dict[f'{cell}->{child}'].detach().T.to('cpu').mean(dim=1).numpy() for child in self.model.tree[cell]['descendant']]).T
                fate_space = fate_space.set_axis([child for child in self.model.tree[cell]['descendant']], axis='columns')
        
        
                for child in self.model.tree[cell]['descendant']:
                    p.obs[f'{child}_fate'] = fate_space[child].values
                    self.treedata.lineage_pairs[f'{cell}->{child}'][f'{child}_transition_matrix'] = att_dict[f'{cell}->{child}'].detach().T.to('cpu').numpy()
        
        self.treedata.nodes[self.progenitor]['adata'] = p
        
        
    def cal_fate_bias(self,terminal_A_fate,terminal_B_fate):
        
        return terminal_A_fate/(terminal_A_fate+terminal_B_fate)
        
        
    def find_state(self,
                   #ancestor,
                   n_state,
                   n_pca=10,
                   n_diff=50,
                   n_gene=30,
                   n_init=1,
                   max_iter=500,
                   tol=1e-4,
                   method='spearman'):
        
        seed_all(self.seed)
        
        for cell in self.model.tree.keys():
            if self.model.tree[cell]['descendant']:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] find cell state of {cell}") 
                D = pd.DataFrame(self.get_node(cell,adata_object=True).X)
                F = pd.DataFrame(self.get_node(cell,adata_object=True).obs.loc[:,[f'{child}_fate' for child in self.model.tree[cell]['descendant']]].values)
        
                fd = Find_State(D,F,n_pca=n_pca,n_diff=n_diff,n_gene=n_gene,n_neighbor=self.k,n_state=n_state,n_init=n_init,max_iter=max_iter,tol=tol,method=method)
                fd.fit()
        
                diffusion_embedding = fd.get_diff([0,1])
        
                state = fd.get_state()
                self.treedata.nodes[cell]['adata'].obs['State'] = state
                self.treedata.nodes[cell]['adata'].obs['State'] = self.treedata.nodes[cell]['adata'].obs['State'].astype('category')
                self.treedata.nodes[cell]['adata'].obs['Fate_State'] = np.char.add(f'{cell}_', state.astype(str))
                self.treedata.nodes[cell]['adata'].obs['Dif_1'] = diffusion_embedding[:,0]
                self.treedata.nodes[cell]['adata'].obs['Dif_2'] = diffusion_embedding[:,1]
            else:
                self.treedata.nodes[cell]['adata'].obs['Fate_State'] = cell

    
    def find_driver(self,progenitor,
                    soft_treshold=2,
                    graph_threshold=0.8,
                    method='pearson',
                    model='linear',
                    lasso_alpha=0.1,
                    top_n = None):
        seed_all(self.seed)
        D = pd.DataFrame(self.get_node(progenitor,adata_object=True).X)
        D.columns = self.treedata.nodes[progenitor]['adata'].var_names
        D.index = self.treedata.nodes[self.progenitor]['adata'].obs_names
        F = pd.DataFrame(self.get_node(progenitor,adata_object=True).obs.loc[:,[f'{child}_fate' for child in self.model.tree[progenitor]['descendant']]])
        
        cd = CD(D=D,
                 F=F,
                 soft_treshold = soft_treshold,
                 method = method,
                 graph_threshold = graph_threshold,
                 top_n = top_n,
                 model=model,
                 lasso_alpha=lasso_alpha)
        cd.fit()
        return cd.coef
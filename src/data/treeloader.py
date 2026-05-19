import torch
import anndata
import numpy as np
from typing import List, Union, Optional
from scipy.sparse import issparse

class TreeNode:
    """
    for each cell state (progenitor or terminal), creating a TreeNode object to store its data and features
    """
    def __init__(self, adata_subset: anndata.AnnData, celltype: str, emb_key: str = 'X_pca', device: str = 'cpu'):
        """
        adata_subset: the subset of AnnData corresponding to the current cell state
        celltype: the name of the current cell state
        emb_key: the key for the dimensionality reduction features (e.g., 'X_pca', 'X_umap')
        device: device to store tensors, default is 'cpu' for preprocessing
        """
        self.celltype = celltype
        self.device = device
        
        # extract raw data matrix (X) and convert to torch tensor
        if issparse(adata_subset.X):
            raw_data = adata_subset.X.toarray()
        else:
            raw_data = adata_subset.X
        
        # use float32 for deep learning compatibility and add small epsilon to avoid log(0)
        raw_data = raw_data.astype(np.float32) + 1e-6
        self.data = torch.from_numpy(raw_data).to(self.device)
        
        # extract embedding features and convert to torch tensor
        if emb_key not in adata_subset.obsm.keys():
            alt_key = f"X_{emb_key}" if not emb_key.startswith('X_') else emb_key
            if alt_key in adata_subset.obsm.keys():
                emb_key = alt_key
            else:
                raise ValueError(f"Feature not found in AnnData.obsm: {emb_key}")
        
        # convert embeddings to tensor and move to initial device
        self.emb = torch.tensor(adata_subset.obsm[emb_key], dtype=torch.float32).to(self.device)
        self.obs = adata_subset.obs.copy()

    def to(self, device: str):
        """
        move tensors to the specified device (e.g., 'cuda') during training
        """
        self.device = device
        self.data = self.data.to(device)
        self.emb = self.emb.to(device)
        return self

class TreeDatasetLoader:
    """
    TreeDataset object that only loads the specified lineage data from AnnData
    """
    def __init__(
        self, 
        adata: anndata.AnnData, 
        progenitor: str, 
        terminal: List[str], 
        lineage_col: str = 'lineage', 
        emb_key: str = 'pca', 
        device: str = 'cpu'
    ):
        """
        adata: the input single-cell AnnData object
        progenitor: the name of the progenitor cell state (string)
        terminal: the list of terminal cell states (list of strings)
        lineage_col: the column name in obs containing cell state names, default is 'lineage'
        emb_key: the key for dimensionality reduction features, default is 'pca'
        device: initial device for data (keep as 'cpu' for large datasets), default is 'cpu'
        """
        self.adata = adata
        self.lineage_col = lineage_col
        self.device = device
        self.emb_key = emb_key
        
        self._validate_params(progenitor, terminal)
        
        self.progenitor = progenitor
        self.terminal = terminal
        self.nodes = {} 
        self.lineage_pairs = {}
        
        self._load_lineage()

    def _validate_params(self, progenitor: str, terminal: List[str]):
        """
        validate if the provided cell states exist in the lineage column
        """
        if self.lineage_col not in self.adata.obs.columns:
            raise KeyError(f"Column not found in AnnData.obs: '{self.lineage_col}'")
        
        available_states = self.adata.obs[self.lineage_col].unique()
        
        if progenitor not in available_states:
            raise ValueError(f"Progenitor '{progenitor}' not found in specified lineage column '{self.lineage_col}'")
            
        for t_state in terminal:
            if t_state not in available_states:
                raise ValueError(f"Terminal state '{t_state}' not found in specified lineage column '{self.lineage_col}'")

    def _load_lineage(self):
        """
        load the data for the specified progenitor and terminal states, and create TreeNode objects
        """
        selected_states = [self.progenitor] + list(self.terminal)
        
        for state in selected_states:
            # extract subset for the current state
            mask = self.adata.obs[self.lineage_col] == state
            adata_sub = self.adata[mask].copy()
            
            # create TreeNode for the current state (defaults to CPU)
            self.nodes[state] = {}
            
            self.nodes[state]['data'] = TreeNode(
                adata_subset=adata_sub,
                celltype=state,
                emb_key=self.emb_key,
                device=self.device
            )
            self.nodes[state]['adata'] = adata_sub
            print(f"Successfully loaded state: {state}, cells: {len(adata_sub)}")
            
            if state != self.progenitor:
                self.lineage_pairs[f"{self.progenitor}->{state}"] = {}

    def cal_fate_bias(self,lineage: list):
        """
            calculate the fate bias for two given lineage pairs
        param:
            lineage: list of two terminal lineages (e.g., ['Neutrophil', 'Monocyte'])
        return:
            the fate bias for progenitor cells towards the two terminal lineages, the value is between 0 and 1, where a value close to 1 indicates a strong bias towards the first lineage in the list, while a value close to 0 indicates a strong bias towards the second lineage.
        """
        if len(lineage) != 2:
            raise ValueError("Lineage list must contain exactly two terminal states.")
        
        pair0 = f"{self.progenitor}->{lineage[0]}"
        pair1 = f"{self.progenitor}->{lineage[1]}"
        
        if pair0 not in self.lineage_pairs or pair1 not in self.lineage_pairs:
            available_pairs = list(self.lineage_pairs.keys())
            raise KeyError(f"One or both lineage pairs not found. Available pairs: {available_pairs}")
        
        bias0 = self.get_lineage_pairs(lineage[0], 'e_propensity')
        bias1 = self.get_lineage_pairs(lineage[1], 'e_propensity')
        
        fb = bias0 / (bias0 + bias1 + 1e-10)  # add small epsilon to avoid division by zero
        return fb

    def get_node(self, celltype: str,adata_object: str = False) -> TreeNode:
        """
        retrieve the TreeNode object for a given cell state
        """
        node = self.nodes.get(celltype)
        if node is None:
            available = list(self.nodes.keys())
            raise KeyError(f"'{celltype}' not found. Available nodes: {available}")
        if adata_object:
            return node['adata']
        return node['data']
    
    def get_lineage_pairs(self,lineage: str,data:str) -> dict:
        """
        retrieve the data for a given lineaege pair (progenitor->terminal)
        """
        l_p = self.lineage_pairs.get(f"{self.progenitor}->{lineage}")
        if l_p is None:
            available = list(self.lineage_pairs.keys())
            raise KeyError(f"'{lineage}' not found. Available lineage pairs: {available}")
        d = l_p.get(data)
        if d is None:
            available_data = list(l_p.keys())
            raise KeyError(f"'{data}' not found in lineage pair '{self.progenitor}->{lineage}'. Available data: {available_data}")
        return d

    def to(self, device: str):
        """
        move all loaded TreeNode data to the specified device (e.g., 'cuda')
        """
        self.device = device
        for node in self.nodes.values():
            node.to(device)
        return self
    

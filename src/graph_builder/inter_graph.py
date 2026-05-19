import numpy as np
import torch
from typing import Dict
from utils.similarity import cal_similarity
from data.treeloader import TreeDatasetLoader

def _inter_graph(treedata: TreeDatasetLoader, target_cell: str, mode: str = 'pos', mask_threshold: float = 0.8, device: str = 'cpu') -> torch.Tensor:
    """
    Generate a masked adjacency matrix based on the fate prior.
    
    Args:
        tree: The lineage tree dictionary structure.
        target_cell: The target descendant cell type name.
        mode: 'pos' for positive mask, 'neg' for negative mask.
        device: Target device for the resulting edge tensors.
        
    Returns:
        edge_index: A numpy array of shape (2, E) representing the masked edges.
    """
    if target_cell not in treedata.terminal:
        raise ValueError(f"Target cell '{target_cell}' is not a terminal cell in the lineage tree.")
    
    parent = treedata.progenitor
    distance_matrix = treedata.get_lineage_pairs(target_cell,'propensity')
    
    # Gather priors for other sibling fates
    if len(treedata.terminal) < 2:
        threshold = 1/treedata.get_lineage_pairs(target_cell,'e_propensity').shape[0]
        other_fate =  np.full(treedata.lineage_pairs[f'{parent}->{target_cell}']['e_propensity'].shape, threshold)
        other_fate = other_fate.reshape(-1, 1)
    else:
        other_fates_list = [
            treedata.get_lineage_pairs(child,'e_propensity')
            for child in treedata.terminal if child != target_cell
        ]
        other_fate = np.array(other_fates_list).T
    
    # Initialize masked adjacency matrix
    if mode == 'neg':
        masked_adj = np.ones_like(distance_matrix)
    elif mode == 'pos':
        masked_adj = np.zeros_like(distance_matrix)
    else:
        raise ValueError("Mode must be either 'pos' or 'neg'")

    _, n = distance_matrix.shape
    
    for cell in range(n):
        target_fate = distance_matrix[:, cell:cell+1]
        fate_space = np.concatenate((target_fate, other_fate), axis=1)
        
        # Find indices where the target fate has the highest probability (index 0)
        dominant_idx = np.where(np.argmax(fate_space, axis=1) == 0)[0]
        
        if mode == 'neg':
            masked_adj[dominant_idx, cell] = 0
        elif mode == 'pos':
            masked_adj[dominant_idx, cell] = 1

    # Filter out rows with too few connections in negative mode
    if mode == 'neg':
        row_sums = masked_adj.sum(axis=1)
    #    # Vectorized thresholding
        masked_adj[row_sums < mask_threshold * n, :] = 0
    # Filter out rows with too few connections in negative mode
    #if mode == 'pos':
    #    row_sums = masked_adj.sum(axis=1)
    #    # Vectorized thresholding
    #    masked_adj[row_sums < (1-mask_threshold) * n, :] = 0

    # Extract edge coordinates
    i, j = np.where(masked_adj == 1)
    edge_index = np.vstack((i, j))     
    edge_index = torch.from_numpy(edge_index).to(torch.int64).to(device)

    return edge_index



def make_inter_state_graph(treedata: TreeDatasetLoader, k: int = 50, mask_threshold: float = 0.8, epsilon: float = 0.1, mode:str = 'composite',device: str = 'cpu'):
    """
    Construct inter-state transition graphs based on the fate prior for each terminal cell state.
    
    Args:
        treedata: An instance of TreeDatasetLoader.
        k: Number of nearest neighbors for similarity calculation.
        mode: 'pos' for positive mask, 'neg' for negative mask.
        mask_threshold: The threshold for filtering edges in negative mode when constructing the inter-state graphs (default: 0.8).
        device: Target device for the resulting edge tensors.
    """
    cal_similarity(treedata, progenitor=treedata.progenitor, terminals=treedata.terminal, k=k, epsilon=epsilon, mode=mode)
    
    for target_cell in treedata.terminal:
        edge_index = _inter_graph(treedata, target_cell=target_cell, mode='pos', mask_threshold=mask_threshold, device=device)
        treedata.lineage_pairs[f'{treedata.progenitor}->{target_cell}'][f'pos_edge'] = edge_index
        
        edge_index = _inter_graph(treedata, target_cell=target_cell, mode='neg', mask_threshold=mask_threshold,  device=device)
        treedata.lineage_pairs[f'{treedata.progenitor}->{target_cell}'][f'neg_edge'] = edge_index
        
        print(f"[*] Inter-state graph built for '{target_cell}' (Edges: {edge_index.shape[1]})")
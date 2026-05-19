import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Union

def knn(
    data: Union[torch.Tensor, np.ndarray], 
    k: int = 10, 
    directed: bool = True, 
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a K-Nearest Neighbors graph from the input data.
    
    Args:
        data: Node features (Tensor or ndarray).
        k: Number of neighbors.
        directed: If True, returns a directed graph (source -> target). 
                  If False, adds reverse edges to make it undirected.
        device: Target device for the output tensors ('cpu' or 'cuda').
        
    Returns:
        weights: Edge weights (distances), shape (E,).
        edge_index: Edge indices [source; target], shape (2, E).
    """
    # sklearn only works with numpy arrays on CPU
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data

    # Fit KNN
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean')
    nn_model.fit(data_np)
    
    # Since we query the training set itself, the 0-th neighbor is always the node itself.
    dists, indices = nn_model.kneighbors(data_np)
    
    # Remove self-loops (exclude the first column) to save memory and time
    dists = dists[:, 1:]
    indices = indices[:, 1:]
    
    num_nodes = data_np.shape[0]
    
    # Vectorized edge construction
    sources = np.repeat(np.arange(num_nodes), k - 1)
    targets = indices.flatten()
    weights = dists.flatten()

    # Make graph undirected if specified
    if not directed:
        sym_sources = np.concatenate([sources, targets])
        sym_targets = np.concatenate([targets, sources])
        sym_weights = np.concatenate([weights, weights])
        
        # Remove duplicate edges to avoid redundancy
        edges = np.vstack([sym_sources, sym_targets])
        unique_edges, unique_indices = np.unique(edges, axis=1, return_index=True)
        
        sources = unique_edges[0]
        targets = unique_edges[1]
        weights = sym_weights[unique_indices]

    # Convert to PyTorch tensors and move to the target device
    edge_index = torch.tensor(np.vstack((sources, targets)), dtype=torch.long, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    return weights_tensor, edge_index
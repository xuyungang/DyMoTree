import torch
from utils import knn
from data.treeloader import TreeDatasetLoader

def make_intra_state_graph(treedata: TreeDatasetLoader, k: int = 50, device: str = 'cpu'):
    """
    Construct directed KNN transition graphs for each individual cell state.
    
    Args:
        treedata: An instance of TreeDatasetLoader.
        nodes: Dictionary of TreeNode objects (e.g., from TreeDatasetLoader.nodes).
        k: Number of nearest neighbors.
        device: Target device for the resulting edge tensors.
    """
    cell_states = [treedata.progenitor] + treedata.terminal
    for state in cell_states:
        node_obj = treedata.nodes.get(state)
        if not node_obj:
            continue
        # Force computations on CPU to save VRAM
        emb_cpu = node_obj['data'].emb.to('cpu')
        
        # Build directed KNN graph for intra-state transitions
        _, edge_index = knn(data=emb_cpu, k=k, directed=True, device='cpu')
        
        # Assign to TreeNode and move to target device
        node_obj['data'].edge = edge_index.to(torch.int64).to(device)
        
        print(f"[*] Intra-state graph built for '{state}' (Edges: {node_obj['data'].edge.shape[1]})")
        
        

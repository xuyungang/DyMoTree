from graph_builder.intra_graph import make_intra_state_graph
from graph_builder.inter_graph import make_inter_state_graph
from data.treeloader import TreeDatasetLoader


def make_lineage_graph(treedata: TreeDatasetLoader, k: int = 50, mask_threshold: float = 0.8, epsilon: float = 0.1, device: str = 'cpu', mode: str = 'composite'):
    """
    Build the complete lineage graph using the provided TreeDatasetLoader.
    This function orchestrates the construction of intra-state and inter-state graphs.

    Args:
        treedata: An instance of TreeDatasetLoader containing cell nodes and lineage pairs.
        k: The number of nearest neighbors for KNN graph construction (default: 50).
        mask_threshold: The threshold for filtering edges in negative mode when constructing the inter-state graphs (default: 0.8).
        device: The target device for tensor operations ('cpu' or 'cuda', default: 'cpu').

    """
    print("--- Start Building Lineage Graph ---")
    
    # 1. Build intra-state graphs (KNN within same cell type)
    make_intra_state_graph(treedata, k, device)
    
    # 2. Build inter-state priors (composite similarity based on shortest paths and linear kernel)
    make_inter_state_graph(treedata, k, mask_threshold, epsilon=epsilon, mode=mode, device=device)



import torch
import numpy as np
import multiprocessing
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from joblib import Parallel, delayed
from typing import Dict, List
from utils.knn_builder import knn
from data.treeloader import TreeNode, TreeDatasetLoader

def cal_similarity(treedata: TreeDatasetLoader, progenitor: str, terminals: List[str], k: int = 50, epsilon: float = 0.1, mode: str = 'composite'):
    """
    Calculate global shortest distance, linear kernel, and fate priors across states.
    Modifies the 'tree' dictionary in-place.
    """
    if not terminals:
        return
        
    print(f"[*] Calculating composite similarity between progenitor '{progenitor}' and terminals {terminals}...")
    
    # 1. Merge embeddings for current ancestor and its descendants (on CPU)
    data_list_np = [treedata.get_node(progenitor,adata_object=False).emb.cpu().numpy()]
    n_p = data_list_np[0].shape[0]
    
    terminal_counts = {}
    for desc in terminals:
        desc_emb = treedata.get_node(desc,adata_object=False).emb.cpu().numpy()
        data_list_np.append(desc_emb)
        terminal_counts[desc] = desc_emb.shape[0]
    
    joint_emb_np = np.vstack(data_list_np)
    joint_emb_tensor = torch.from_numpy(joint_emb_np)
    
    # 2. Construct undirected global KNN graph for shortest path
    w, edge = knn(data=joint_emb_tensor, k=k, directed=False, device='cpu')
    edge = edge.numpy().astype(int)
    w = w.numpy()

    sources = edge[0, :]
    targets = edge[1, :]
    num_nodes = max(sources.max(), targets.max()) + 1
    
    A = coo_matrix((w, (sources, targets)), shape=(num_nodes, num_nodes))
    adj = A.maximum(A.T) # Ensure symmetric adjacency
    
    # 3. Parallel calculation of shortest distance matrix
    def shortest_path_worker(graph, indices):
        return shortest_path(csgraph=graph.copy(), method='D', directed=False, indices=indices)
        
    num_cores = multiprocessing.cpu_count()
    node_chunks = np.array_split(np.arange(num_nodes), num_cores)
    
    results_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(shortest_path_worker)(adj, chunk) for chunk in node_chunks
    )
    distance_matrix = np.vstack(results_list)
    distance_matrix = distance_matrix[0:n_p, n_p:]
    #tree[progenitor]['dist'] = distance_matrix
    
    # 4. Calculate Linear Kernel
    joint_emb_np_f32 = joint_emb_np.astype(np.float32)
    linear_kernel_matrix = np.dot(joint_emb_np_f32, joint_emb_np_f32.T)
    linear_kernel_matrix = linear_kernel_matrix[0:n_p, n_p:]
    
    col_offset = 0
    for desc in terminals:
        # Get the number of columns for the current terminal state
        n_c = terminal_counts[desc]

        # Slice columns from col_offset to col_offset + n_c
        path = distance_matrix[:, col_offset : col_offset + n_c]
        linear = linear_kernel_matrix[:, col_offset : col_offset + n_c]
        
        norm_path = 1 / (path + 1e-10)
        norm_linear = (linear - linear.min(axis=0, keepdims=True) + 1e-10) / (linear.max(axis=0, keepdims=True) - linear.min(axis=0, keepdims=True) + 1e-10)
        norm_path = (norm_path - norm_path.min(axis=0, keepdims=True) + 1e-10) / (norm_path.max(axis=0, keepdims=True) - norm_path.min(axis=0, keepdims=True) + 1e-10)
        
        if mode == 'composite':
            m_propensity = norm_path + epsilon*norm_linear
        elif mode == 'shortest_path':
            m_propensity = norm_path
        elif mode == 'linear_kernel':
            m_propensity = norm_linear
        else:
            raise ValueError("Invalid mode. Please choose from 'composite', 'shortest_path', or 'linear_kernel'.")

        m_propensity = m_propensity / m_propensity.sum(axis=0, keepdims=True)
        
        e_propensity = m_propensity.mean(axis=1)
        
        
        treedata.lineage_pairs[f"{progenitor}->{desc}"]['shortest_path'] = path
        treedata.lineage_pairs[f"{progenitor}->{desc}"]['linear_kernel'] = linear
        treedata.lineage_pairs[f"{progenitor}->{desc}"]['propensity'] = m_propensity
        treedata.lineage_pairs[f"{progenitor}->{desc}"]['e_propensity'] = e_propensity
        
        treedata.nodes.get(progenitor).get('adata').obs[f'{desc}_propensity'] = e_propensity
        
        col_offset += n_c
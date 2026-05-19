import torch

def bipartite_degree_aware_sampler(
    pos_edge,
    neg_edge,
    batch_size,
    alpha=0.,
    eps=1e-8,
    replacement=False,
):
    """
    二部图场景下，对正负边分别进行基于 A 类节点随机采样的边采样。

    采样逻辑：
    1) 在每个边集（正边或负边）中，取出出现过的 A 类节点
    2) 随机采样固定数量的 A 类节点
    3) 返回这些 A 类节点关联的所有边

    参数：
        pos_edge: Tensor [2, n1]
            第一行为 A 类节点，第二行为 B 类节点
        neg_edge: Tensor [2, n2]
            第一行为 A 类节点，第二行为 B 类节点
        batch_size: int
            每个边集中固定采样的 A 节点数
        alpha: float
            保留该参数以保持函数签名不变，本实现中未使用
        eps: float
            保留该参数以保持函数签名不变，本实现中未使用
        replacement: bool
            采样 A 节点时是否允许重复

    返回：
        pos_batch: Tensor [2, m1]
        neg_batch: Tensor [2, m2]
        其中 m1, m2 为采样到的 A 节点对应的全部边数
    """

    device = pos_edge.device

    def sample_one_edge_set(edge_index):
        """
        对单个边集（正边或负边）进行：
        1) 找出实际出现过的 A 类节点
        2) 随机采样固定数量的 A 类节点
        3) 返回这些 A 类节点关联的所有边
        """
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return edge_index[:, :0]

        a_nodes = edge_index[0]

        # 只对实际出现过的 A 节点进行采样
        unique_a = torch.unique(a_nodes)

        num_a_sample = batch_size
        if not replacement:
            num_a_sample = min(num_a_sample, unique_a.numel())

        if num_a_sample <= 0:
            return edge_index[:, :0]

        # 均匀随机采样 A 节点
        if replacement:
            sampled_idx = torch.randint(
                low=0,
                high=unique_a.numel(),
                size=(num_a_sample,),
                device=device
            )
        else:
            sampled_idx = torch.randperm(unique_a.numel(), device=device)[:num_a_sample]

        sampled_a = unique_a[sampled_idx]

        # 取这些 A 节点关联的所有边
        mask = torch.isin(a_nodes, sampled_a)
        candidate_edges = edge_index[:, mask]

        return candidate_edges

    pos_batch = sample_one_edge_set(pos_edge)
    neg_batch = sample_one_edge_set(neg_edge)

    return pos_batch, neg_batch
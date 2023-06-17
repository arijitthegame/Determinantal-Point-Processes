import numpy as np


class TreeNode(object):

    def __init__(self, items, level=-1):

        self.items = items
        self.is_leaf = len(items) == 1
        self.level = level
        self.Sigma = None

    def is_leaf(self):
        return self.left is None

    def compute_probability(self, query_matrix, BTB_all=None):
        if self.Sigma is not None:
            return (self.Sigma * query_matrix).sum()
        elif self.Sigma is None and BTB_all is not None:
            Sigma = BTB_all[self.items].sum(axis=0)
            return (Sigma * query_matrix).sum()
        else:
            raise NotImplementedError

    def sampling(self, query_matrix, BTB_all=None):
        if self.is_leaf:
            return self.items[0]
        p_l = self.left.compute_probability(query_matrix, BTB_all).clip(min=0.0) 
        p_r = self.right.compute_probability(query_matrix, BTB_all).clip(min=0.0)
        prob = np.random.rand()

        if prob < p_l / (p_l + p_r):
            return self.left.sampling(query_matrix, BTB_all)
        else:
            return self.right.sampling(query_matrix, BTB_all)


def split_items(items):
    n = len(items)
    return items[:n // 2], items[n // 2:]


def construct_tree(items, B):

    def branch(level, items, B):
        num_data = len(items)

        if num_data == 1:
            new_node = TreeNode(items=items, level=level)
            v_j = B[:, items[0]]
            new_node.Sigma = np.outer(v_j, v_j)
            assert new_node.is_leaf
            return new_node

        node = TreeNode(items=items, level=level)
        left_items, right_items = split_items(items)
        node.left = branch(level=level + 1, items=left_items, B=B)
        node.right = branch(level=level + 1, items=right_items, B=B)
        node.Sigma = node.left.Sigma + node.right.Sigma
        return node

    node = branch(0, np.arange(len(items)), B)
    return node


def update_query_matrix(V, samples, mask=None):
    d = V.shape[1]
    masked_eye = np.eye(d) if mask is None else np.diag(mask) 
    if len(samples) == 0:
        return masked_eye
    else:
        C = torch.zeros(d, len(samples)).astype(V.dtype)
        for i, j in enumerate(samples):
            v_j = V[j, :] if mask is None else V[j, :] * mask
            if i == 0:
                C[:, i] = v_j / np.sqrt(np.dot(v_j, v_j))
            else:
                tmp = v_j[:, None] - C[:, :i] @ (C[:, :i].T @ v_j[:, None])
                C[:, i] = tmp.flatten() / np.linalg.norm(tmp)
        return masked_eye - C @ C.T


def tree_sampling_dpp(tree, V, eigen_vals):
    d = V.shape[1]
    assert len(eigen_vals) == d
    assert len(tree.items) == V.shape[0]

    # Phase 1.
    rand_nums = np.random.rand(d)
    probs = eigen_vals / (eigen_vals + 1)
    selected_indices = np.where(rand_nums < probs)[0]

    mask = np.zeros(d)
    mask[selected_indices] = 1.0

    # Phase 2.
    sample = []
    for _ in range(len(selected_indices)):
        query_matrix = update_query_matrix(V, sample, mask)
        sample.append(tree.sampling(query_matrix))
    return sorted(sample)
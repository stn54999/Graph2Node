# 获取每个图的全局表示
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def get_global_representations(model, loader):
    model.eval()
    global_representations = []
    global_labels = []

    for data in loader:
        output = model(data.x, data.edge_index, data.batch)
        global_representations.extend(output.tolist())
        global_labels.extend(data.y.tolist())

    return global_representations, global_labels


# 生成超级节点数据集
def create_masked_super_node_dataset_transductive(train_global_representations, train_global_labels, test_global_representations=None, test_global_labels=None, k=5):
    num_train = len(train_global_representations)

    if test_global_representations is not None and test_global_labels is not None:
        num_test = len(test_global_representations)
        all_representations = train_global_representations + test_global_representations
        all_labels = train_global_labels + test_global_labels
    else:
        num_test = 0
        all_representations = train_global_representations
        all_labels = train_global_labels

    x = torch.tensor(all_representations, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    # 创建训练集和测试集掩码
    train_mask = torch.tensor([True] * num_train + [False] * num_test)
    test_mask = torch.tensor([False] * num_train + [True] * num_test)

    # 使用 k-近邻算法计算相似度
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
    adjacency_matrix = nbrs.kneighbors_graph(x, mode='connectivity').toarray()

    # 将邻接矩阵转换为边缘索引
    edge_index = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)

    return data


def create_masked_super_node_dataset_inductive(train_global_representations, train_global_labels, test_global_representations, test_global_labels, test_index, k=5):
    num_train = len(train_global_representations)
    num_test = len(test_global_representations)

    # 仅使用 test_index 指定的测试样本
    all_representations = train_global_representations + [test_global_representations[test_index]]
    all_labels = train_global_labels + [test_global_labels[test_index]]

    x = torch.tensor(all_representations, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    # 创建训练集和测试集掩码
    train_mask = torch.tensor([True] * num_train + [False])
    test_mask = torch.tensor([False] * num_train + [True])

    # 使用 k-近邻算法计算相似度
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
    adjacency_matrix = nbrs.kneighbors_graph(x, mode='connectivity').toarray()

    # 将邻接矩阵转换为边缘索引
    edge_index = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)

    return data
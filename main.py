import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.transforms import knn_graph
import numpy as np
from sklearn.neighbors import NearestNeighbors
# 加载数据集
dataset = TUDataset(root='/tmp/TUDataset', name='ENZYMES')
dataset = dataset.shuffle()
train_dataset = dataset[:540]
test_dataset = dataset[540:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_layer, pool_layer):
        super(GNNModel, self).__init__()
        self.conv1 = conv_layer(input_dim, hidden_dim)
        self.conv2 = conv_layer(hidden_dim, hidden_dim)
        self.pool = pool_layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index ,batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 训练和测试函数
def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(train_dataset)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        output = model(data.x, data.edge_index,data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)

# 参数设置
input_dim = dataset.num_node_features
hidden_dim = 64
output_dim = dataset.num_classes
conv_layer = GCNConv  # 可以选择GCNConv、SAGEConv或GATConv
pool_layer = global_mean_pool  # 可以选择global_mean_pool、global_max_pool或global_add_pool

# 定义和训练原始GNN模型
model = GNNModel(input_dim, hidden_dim, output_dim, conv_layer, pool_layer)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 1000):
    train_loss = train(model, train_loader, optimizer)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# 获取每个图的全局表示
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
def create_masked_super_node_dataset(train_global_representations, train_global_labels, test_global_representations, test_global_labels,k=5):
    num_train = len(train_global_representations)
    num_test = len(test_global_representations)

    all_representations = train_global_representations + test_global_representations
    all_labels = train_global_labels + test_global_labels

    x = torch.tensor(all_representations, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)
    print(x.shape,y.shape)

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
    data = Data(x=x, y=y,edge_index=edge_index,train_mask=train_mask, test_mask=test_mask)

    return data


# 获取训练集和测试集的全局表示和标签
train_global_representations, train_global_labels = get_global_representations(model, train_loader)
test_global_representations, test_global_labels = get_global_representations(model, test_loader)

# 创建掩码的超级节点数据集
super_node_data = create_masked_super_node_dataset(train_global_representations, train_global_labels, test_global_representations, test_global_labels,5)

# 定义掩码 GNN 模型
class MaskedGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_layer):
        super(MaskedGNNModel, self).__init__()
        self.conv1 = conv_layer(input_dim, hidden_dim)
        self.conv2 = conv_layer(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 训练和测试掩码 GNN 模型的函数
def train_masked(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_masked(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.max(dim=1)[1]
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    return correct / (data.test_mask).sum().item()

# 训练新的掩码 GNN 模型进行节点分类
super_node_model = MaskedGNNModel(output_dim, hidden_dim, output_dim, conv_layer)
optimizer = torch.optim.Adam(super_node_model.parameters(), lr=0.01)

for epoch in range(1, 1001):
    train_loss = train_masked(super_node_model, super_node_data, optimizer)
    test_acc = test_masked(super_node_model, super_node_data)
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')

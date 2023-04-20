from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data

# 加载数据集
dataset = TUDataset(root='/tmp/TUDataset', name='ENZYMES')
dataset = dataset.shuffle()
train_dataset = dataset[:540]
test_dataset = dataset[540:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
import torch
import torch.nn.functional as F

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
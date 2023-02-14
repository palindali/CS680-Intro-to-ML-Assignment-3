# Q2.1

# Following tutorial used as reference
# https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
# https://arshren.medium.com/different-graph-neural-network-implementation-using-pytorch-geometric-23f5bf2f3e9f

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import json


# Implement Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels=64):
        super().__init__()

        self.conv1 = GCNConv(num_feats, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, x, edge_index):
        # x: Node feature matrix 
        # edge_index: Graph connectivity matrix

        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Visualization function
# emb: (nNodes, hidden_dim)
# node_type: (nNodes,). Entries are torch.int64 ranged from 0 to num_class - 1
def visualize(emb: torch.tensor, node_type: torch.tensor):
    z = TSNE(n_components=2).fit_transform(emb.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.scatter(z[:, 0], z[:, 1], s=70, c=node_type, cmap="Set2")
    plt.show()

if __name__ == '__main__':
    # Load dataset
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]
    print(data)

    # Params
    num_epochs = 200
    learning_rate = 0.01
    # Initialize model, optimizer and loss
    model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    # Testing function
    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        # Train set accuracy
        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = 100 * int(train_correct.sum()) / int(data.train_mask.sum())
        # Test set accuracy
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = 100 * int(test_correct.sum()) / int(data.test_mask.sum())
        return train_acc, test_acc

    # Visualize initial node embeddings
    model.eval()

    out = model(data.x, data.edge_index)
    visualize(out, data.y)

    # Train model
    trn_lss_list = []
    trn_acc_list = []
    tst_acc_list = []
    for epoch in range(1, num_epochs + 1):
        loss = train()
        trn_lss_list.append(loss.item())
        train_acc, test_acc = test()
        trn_acc_list.append(train_acc)
        tst_acc_list.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}')

    # Save output
    torch.save(model.state_dict(), f"../model_checkpoints/Q2/Q2.1_GCN.pt")

    res_dict = {
        'Train Loss': trn_lss_list,
        'Train Accuracy': trn_acc_list,
        'Test Accuracy': tst_acc_list,
    }
    with open("../plots/Q2_1_res.json", "w") as write_file:
        json.dump(res_dict, write_file, indent=4)
    
    # Plots
    iters = list(range(1, num_epochs+1))
    # A
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    lns1 = ax1.plot(iters, trn_lss_list, color='springgreen', label = "Training Loss")
    lns2 = ax2.plot(iters, trn_acc_list, color='rebeccapurple', label = "Train Accuracy")
    lns3 = ax2.plot(iters, tst_acc_list, color='darkturquoise', label = "Test Accuracy")

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.title("Q2.1 Training Loss, Train and Test Accuracy per iteration")
    plt.show()
    
    # Visualize final node embeddings
    model.eval()

    out = model(data.x, data.edge_index)
    visualize(out, data.y)

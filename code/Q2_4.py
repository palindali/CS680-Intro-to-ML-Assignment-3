# Q2.4

# Following tutorial used as reference
# https://docs.dgl.ai/en/0.9.x/tutorials/multi/1_graph_classification.html
# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=HvhgQoO8Svw4
# https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/examples/mutag_gin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import json


# Implement GIN
class GIN(nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels=64):
        super(GIN, self).__init__()

        self.conv1 = GINConv(nn.Linear(num_feats, hidden_channels), aggregator_type='sum')
        self.conv2 = GINConv(nn.Linear(hidden_channels, num_classes), aggregator_type='sum')
    
    def forward(self, x, edge_index, batch):
        # x: Node feature matrix 
        # edge_index: Graph connectivity matrix
        # batch: Batch

        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
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
    dataset = TUDataset(root='data/TUDataset', name='MUTAG', transform=NormalizeFeatures())
    
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    
    # dataset = dataset.shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8): ]
    
    print('==== train_dataset =====')
    print(train_dataset)
    print('==== test_dataset =====')
    print(test_dataset)
    
    # Params
    num_epochs = 30
    learning_rate = 0.01
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer and loss
    model = GIN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training function
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)
    
    # Testing function
    @torch.no_grad()
    def test(loader):
        model.eval()
        
        correct = 0
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        accuracy = 100 * correct / len(loader.dataset)
        
        return accuracy

    # Train model
    trn_lss_list = []
    trn_acc_list = []
    tst_acc_list = []
    for epoch in range(1, num_epochs + 1):
        loss = train()
        trn_lss_list.append(loss)
        train_acc = test(train_loader)
        trn_acc_list.append(train_acc)
        test_acc = test(test_loader)
        tst_acc_list.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}')

    # Save output
    torch.save(model.state_dict(), f"../model_checkpoints/Q2/Q2.4_GIN_{batch_size}.pt")

    res_dict = {
        'Train Loss': trn_lss_list,
        'Train Accuracy': trn_acc_list,
        'Test Accuracy': tst_acc_list,
    }
    with open(f"../plots/Q2_4_res_{batch_size}.json", "w") as write_file:
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

    plt.title(f"Q2.4 batch_size={batch_size} Training Loss, Train and Test Accuracy per iteration")
    plt.show()

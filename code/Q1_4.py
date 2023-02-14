# Q1.4 code part 1 (model training)

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from vgg11 import VGG11

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == '__main__':
    # Params
    batch_size = 256
    
    num_classes = 10
    in_channels = 1

    num_epochs = 5
    learning_rate = 0.005

    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loading
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ])
    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0, 0.316)
    ])
    transform3 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        torchvision.transforms.ToTensor(),
    ])
    transform4 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.RandomVerticalFlip(p=1),
        torchvision.transforms.ToTensor(),
    ])
        
    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )
    train_dataset2 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform2, download=True
    )
    train_dataset3 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform3, download=True
    )
    train_dataset4 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform4, download=True
    )
    final_train_dataset = ConcatDataset([
        train_dataset, train_dataset2, train_dataset3, train_dataset4
    ])

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Training
    model = VGG11(in_channels, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    test_critereon = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay = 0.005, 
        momentum = 0.9
    )  

    # Train the model
    total_step = len(train_loader)

    trn_accu = []
    trn_loss = []
    tst_accu = []
    tst_loss = []
    model_state_dicts = []
    for epoch in range(num_epochs):
        model.train()
        # Training
        print(f"Epoch {epoch+1}:")
        for i, (images, labels) in enumerate(tqdm(train_loader, "Training", leave=False)):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        model_state_dicts.append(model.state_dict())
        # torch.save(model.state_dict(), f"../model_checkpoints/Q1/vgg11_aug_epoch{epoch+1}.pt")
        
        model.eval()
        # Train error
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0
            for images, labels in tqdm(train_loader, "Testing on Training Set", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = test_critereon(outputs, labels)
                running_loss += loss.item()
                del images, labels, outputs

            accuracy = 100 * correct / total
            loss = running_loss / total
            print(f"Accuracy of the network on the {total} training images: {accuracy} %") 
            print(f"Training Loss: {loss}")
            trn_accu.append(accuracy)
            trn_loss.append(loss)
        
        # Test error
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0
            for images, labels in tqdm(test_loader, "Testing on Test Set", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = test_critereon(outputs, labels)
                running_loss += loss.item()
                del images, labels, outputs

            accuracy = 100 * correct / total
            loss = running_loss / total
            print(f"Accuracy of the network on the {total} testing images: {accuracy} %") 
            print(f"Testing Loss: {loss}")
            tst_accu.append(accuracy)
            tst_loss.append(loss)
    
    # Save output
    torch.save(model.state_dict(), f"../model_checkpoints/Q1/vgg11_aug.pt")

    res_dict = {
        'Train Accuracy': trn_accu,
        'Train Loss': trn_loss,
        'Test Accuracy': tst_accu,
        'Test Loss': tst_loss,
    }
    with open("../plots/Q1_4_res.json", "w") as write_file:
        json.dump(res_dict, write_file, indent=4)
    
    # Plots
    # A
    plt.plot(list(range(1, num_epochs+1)), tst_accu, color="rebeccapurple")
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.title('Q1.4: Test Accuracy per Epoch')
    plt.savefig("../plots/Q1_4A.png")
    plt.show()
    # B
    plt.plot(list(range(1, num_epochs+1)), trn_accu, color="rebeccapurple")
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epoch')
    plt.title('Q1.4: Train Accuracy per Epoch')
    plt.savefig("../plots/Q1_4B.png")
    plt.show()
    # C
    plt.plot(list(range(1, num_epochs+1)), tst_loss, color="rebeccapurple")
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.title('Q1.4: Test Loss per Epoch')
    plt.savefig("../plots/Q1_4C.png")
    plt.show()
    # D
    plt.plot(list(range(1, num_epochs+1)), trn_loss, color="rebeccapurple")
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.title('Q1.4: Train Loss per Epoch')
    plt.savefig("../plots/Q1_4D.png")
    plt.show()

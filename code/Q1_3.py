# Q1.3

import torch
import torch.nn as nn
import torchvision
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
    
    # Flip Horizontally
    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        torchvision.transforms.ToTensor()
    ])

    test_dataset1 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform1, download=True
    )

    # Flip Vertically
    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.RandomVerticalFlip(p=1),
        torchvision.transforms.ToTensor()
    ])
    
    test_dataset2 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform2, download=True
    )

    
    # Gaussian Noise: 0.01 var
    transform3 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0, 0.1)
    ])

    test_dataset3 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform3, download=True
    )

    # Gaussian Noise: 0.1 var
    transform4 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0, 0.316)
    ])

    test_dataset4 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform4, download=True
    )

    # Gaussian Noise: 1 var
    transform5 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0, 1)
    ])

    test_dataset5 = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform5, download=True
    )

    test_loader1 = torch.utils.data.DataLoader(
        test_dataset1, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader2 = torch.utils.data.DataLoader(
        test_dataset2, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader3 = torch.utils.data.DataLoader(
        test_dataset3, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader4 = torch.utils.data.DataLoader(
        test_dataset4, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader5 = torch.utils.data.DataLoader(
        test_dataset5, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Load model
    model = VGG11(in_channels, num_classes).to(device)
    model.load_state_dict(torch.load("../model_checkpoints/Q1/vgg11.pt"))
    model.eval()
    
    test_critereon = nn.CrossEntropyLoss(reduction='sum')

    # Test error
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader1, "Testing 1:", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        accuracy1 = 100 * correct / total
        print(f"Accuracy 1: {accuracy1} %") 

        correct = 0
        total = 0
        for images, labels in tqdm(test_loader2, "Testing 2:", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        accuracy2 = 100 * correct / total
        print(f"Accuracy 2: {accuracy2} %") 

        correct = 0
        total = 0
        for images, labels in tqdm(test_loader3, "Testing 3:", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        accuracy3 = 100 * correct / total
        print(f"Accuracy 3: {accuracy3} %") 

        correct = 0
        total = 0
        for images, labels in tqdm(test_loader4, "Testing 4:", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        accuracy4 = 100 * correct / total
        print(f"Accuracy 4: {accuracy4} %") 

        correct = 0
        total = 0
        for images, labels in tqdm(test_loader5, "Testing 5:", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del images, labels, outputs

        accuracy5 = 100 * correct / total
        print(f"Accuracy 5: {accuracy5} %") 


    plt.bar(['Horizontal Flip', 'Vertical Flip'], [accuracy1, accuracy2], color ='rebeccapurple', width = 0.4)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Flip Type')
    plt.title('Q1.3: Test Accuracy vs. Flip Type')
    plt.savefig("../plots/Q1_3E.png")
    plt.show()
    
    plt.bar(['0.01', '0.1', '1'], [accuracy3, accuracy4, accuracy5], color ='rebeccapurple', width = 0.4)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Variance of Noise')
    plt.title('Q1.3: Test Accuracy vs. Variance of Noise')
    plt.savefig("../plots/Q1_3F.png")
    plt.show()

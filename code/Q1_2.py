# Q1.1 and Q1.2 code

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from vgg11 import VGG11

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

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
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
        # torch.save(model.state_dict(), f"../model_checkpoints/Q1/vgg11_epoch{epoch+1}.pt")
        
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
    torch.save(model.state_dict(), f"../model_checkpoints/Q1/vgg11.pt")

    res_dict = {
        'Train Accuracy': trn_accu,
        'Train Loss': trn_loss,
        'Test Accuracy': tst_accu,
        'Test Loss': tst_loss,
    }
    with open("../plots/Q1_2_res.json", "w") as write_file:
        json.dump(res_dict, write_file, indent=4)
    
    # Plots
    # A
    plt.plot(list(range(1, num_epochs+1)), tst_accu, color ='rebeccapurple')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.title('Q1.2: Test Accuracy per Epoch')
    plt.savefig("../plots/Q1_2A.png")
    plt.show()
    # B
    plt.plot(list(range(1, num_epochs+1)), trn_accu, color ='rebeccapurple')
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epoch')
    plt.title('Q1.2: Train Accuracy per Epoch')
    plt.savefig("../plots/Q1_2B.png")
    plt.show()
    # C
    plt.plot(list(range(1, num_epochs+1)), tst_loss, color ='rebeccapurple')
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.title('Q1.2: Test Loss per Epoch')
    plt.savefig("../plots/Q1_2C.png")
    plt.show()
    # D
    plt.plot(list(range(1, num_epochs+1)), trn_loss, color ='rebeccapurple')
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.title('Q1.2: Train Loss per Epoch')
    plt.savefig("../plots/Q1_2D.png")
    plt.show()

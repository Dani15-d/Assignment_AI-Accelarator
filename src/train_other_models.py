import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms

from models.lenet5 import get_lenet5_mnist
from models.mobilenetv2 import get_mobilenetv2_cifar100

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def train_lenet5():
    print("Training LeNet-5 on MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = dset.MNIST(root='./data', train=True, transform=transform, download=False)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    
    model = get_lenet5_mnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    epochs = 3
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
        acc = correct / total * 100.0
        print(f"LeNet-5 Epoch {ep+1}/{epochs}: Loss = {total_loss/total:.4f}, Accuracy = {acc:.2f}%")

    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), './checkpoints/lenet5_mnist.pth')
    print("LeNet-5 checkpoint saved to checkpoints/lenet5_mnist.pth\n")

def train_mobilenetv2():
    print("Training MobileNetV2 on CIFAR-100...")
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor],
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ) if False else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_data = dset.CIFAR100(root='./data', train=True, transform=transform, download=False)
    subset_indices = list(range(5000))
    train_subset = Subset(train_data, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    
    model = get_mobilenetv2_cifar100()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    max_steps = 35
    step = 0
    while step < max_steps:
        for x, y in train_loader:
            step += 1
            if step > max_steps:
                break
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if step % 10 == 0 or step == max_steps:
                print(f"MobileNetV2 Step {step}/{max_steps}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), './checkpoints/mobilenetv2_cifar100.pth')
    print("MobileNetV2 checkpoint saved to checkpoints/mobilenetv2_cifar100.pth\n")

if __name__ == "__main__":
    train_lenet5()
    train_mobilenetv2()

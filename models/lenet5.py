import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Computational layers: Conv, Activation, Pooling, Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

def get_lenet5_mnist():
    return LeNet5(num_classes=10)

if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    m = get_lenet5_mnist()
    y = m(x)
    print("LeNet-5 output shape:", y.shape)

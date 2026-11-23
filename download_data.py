import os
import torchvision

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
data_dir = os.path.abspath('./data')
os.makedirs(data_dir, exist_ok=True)

print("Downloading MNIST...")
torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
print("MNIST downloaded successfully.")

print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
print("CIFAR-10 downloaded successfully.")

print("Downloading CIFAR-100...")
torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
print("CIFAR-100 downloaded successfully.")

print("All datasets downloaded successfully to", data_dir)

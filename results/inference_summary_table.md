# Inference Numerical Range & Precision Analysis Across Models

## 1. Summary Across Computational Layers and Data Types

| Model | Dataset | Layer / Data Type | Min Value | Max Value | Smallest |x|>0 | Min Spacing | Zero Sparsity | INT8 SQNR (dB) | INT16 SQNR (dB) | FP16 Feasibility | INT8 Feasibility |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_cifar10 | CIFAR-10 | Inputs | -2.429 | 2.754 | 7.670e-04 | 3.219e-06 | 0.0% | 46.0 | 94.2 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Output Logits | -18.460 | 17.909 | 7.303e-06 | 1.490e-08 | 0.0% | 34.5 | 82.7 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Weight: conv1.weight | -0.303 | 0.307 | 2.337e-05 | 4.470e-08 | 0.0% | 44.8 | 92.9 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Weight: layer3.0.conv1.weight | -0.051 | 0.051 | 5.470e-08 | 2.910e-10 | 0.0% | 43.6 | 91.8 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Weight: fc.weight | -0.161 | 0.214 | 2.515e-05 | 7.451e-09 | 0.0% | 37.3 | 85.7 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: conv1 | -6.502 | 6.963 | 5.975e-06 | 7.451e-09 | 0.0% | 36.9 | 85.1 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: bn1 | -4.544 | 4.669 | 2.146e-06 | 1.490e-08 | 0.0% | 38.8 | 87.1 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: relu | 0.000 | 4.294 | 1.907e-06 | 1.118e-08 | 49.3% | 39.7 | 88.0 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: layer1.0.conv1 | -16.464 | 11.226 | 6.579e-06 | 1.490e-08 | 0.0% | 36.8 | 85.0 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: layer1.0.bn1 | -4.050 | 6.584 | 1.037e-05 | 1.490e-08 | 0.0% | 36.5 | 84.7 | Adequate | Adequate |
| resnet18_cifar10 | CIFAR-10 | Act: fc | -17.970 | 17.909 | 7.303e-06 | 1.490e-08 | 0.0% | 34.7 | 82.9 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Inputs | -1.897 | 2.025 | 1.659e-03 | 6.795e-06 | 0.0% | 46.7 | 95.0 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Output Logits | -3.216 | 1.081 | 6.638e-03 | 4.657e-10 | 0.0% | 43.3 | 91.4 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Weight: features.0.0.weight | -0.199 | 0.200 | 2.791e-04 | 5.960e-07 | 0.0% | 48.0 | 96.1 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Weight: features.9.conv.2.weight | -0.661 | 0.716 | 1.004e-05 | 3.260e-09 | 0.0% | 40.7 | 89.0 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Weight: classifier.1.weight | -0.048 | 0.044 | 7.238e-08 | 2.328e-10 | 0.0% | 43.8 | 92.0 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: features.1.conv.0.0 | -1.483 | 1.294 | 1.675e-08 | 9.313e-10 | 30.8% | 35.7 | 83.9 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: features.1.conv.0.1 | -6.470 | 4.649 | 1.690e-05 | 5.588e-09 | 0.0% | 32.2 | 80.4 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: features.1.conv.1 | -4.624 | 4.671 | 1.937e-06 | 1.490e-08 | 0.0% | 37.6 | 85.8 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: features.1.conv.2 | -4.649 | 3.626 | 5.603e-06 | 7.451e-09 | 0.0% | 35.5 | 83.7 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: features.2.conv.0.0 | -3.243 | 3.677 | 6.847e-06 | 3.725e-09 | 0.0% | 32.5 | 80.7 | Adequate | Adequate |
| mobilenetv2_cifar100 | CIFAR-100 | Act: classifier.1 | -3.216 | 1.081 | 6.646e-03 | 4.657e-10 | 0.0% | 43.3 | 91.4 | Adequate | Adequate |
| lenet5_mnist | MNIST | Inputs | -0.424 | 2.821 | 4.181e-03 | 1.273e-02 | 0.0% | 49.5 | 88.6 | Adequate | Adequate |
| lenet5_mnist | MNIST | Output Logits | -20.746 | 22.209 | 2.024e-05 | 2.980e-08 | 0.0% | 41.7 | 89.9 | Adequate | Adequate |
| lenet5_mnist | MNIST | Weight: conv1.weight | -0.406 | 0.362 | 3.707e-03 | 4.924e-05 | 0.0% | 46.3 | 94.2 | Adequate | Adequate |
| lenet5_mnist | MNIST | Weight: fc1.weight | -0.469 | 0.484 | 1.145e-06 | 2.328e-10 | 0.0% | 35.8 | 84.0 | Adequate | Adequate |
| lenet5_mnist | MNIST | Weight: fc3.weight | -0.290 | 0.242 | 1.213e-04 | 8.941e-08 | 0.0% | 43.8 | 92.0 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: conv1 | -6.731 | 7.117 | 3.094e-05 | 9.313e-10 | 0.0% | 38.2 | 86.4 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: relu1 | 0.000 | 7.071 | 1.854e-04 | 9.313e-10 | 51.2% | 39.2 | 88.8 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: pool1 | 0.000 | 6.548 | 1.093e-05 | 2.328e-10 | 40.4% | 39.1 | 85.6 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: conv2 | -20.326 | 10.810 | 3.392e-05 | 3.725e-09 | 0.0% | 36.3 | 84.5 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: relu2 | 0.000 | 11.068 | 1.734e-05 | 3.725e-09 | 62.0% | 38.5 | 86.7 | Adequate | Adequate |
| lenet5_mnist | MNIST | Act: fc3 | -20.746 | 22.209 | 2.024e-05 | 1.490e-08 | 0.0% | 41.7 | 89.9 | Adequate | Adequate |
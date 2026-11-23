import torch
import torch.nn as nn
import torchvision.models as models

def get_mobilenetv2_cifar100():
    """
    MobileNetV2 adapted for CIFAR-100 (32x32 images, 100 classes).
    Initial conv stride set to 1 to preserve spatial dimensions,
    and final classifier mapped to 100 classes.
    """
    model = models.mobilenet_v2(weights=None)
    # Adapt first conv layer: stride 1 instead of 2 for 32x32 inputs
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    # Adapt classifier
    model.classifier[1] = nn.Linear(model.last_channel, 100, bias=True)
    # Set inplace=False on all activations
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            m.inplace = False
    return model

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    m = get_mobilenetv2_cifar100()
    y = m(x)
    print("MobileNetV2 output shape:", y.shape)

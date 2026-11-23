import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from models.resnet import get_resnet18_cifar10
from src.collector import TrainingDataCollector
from src.metrics import compute_detailed_metrics
from src.plotter import plot_dual_distribution

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def train_and_analyze_resnet():
    device = torch.device('cpu')
    print("Setting up CIFAR-10 data loaders...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = dset.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
    # Using a 10,000-image subset for swift convergence and controlled training iterations
    indices = list(range(10000))
    train_subset = torch.utils.data.Subset(train_set, indices)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)

    model = get_resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    target_layers = [
        'conv1', 'bn1',
        'layer1.0.conv1', 'layer1.0.bn1',
        'layer2.0.conv1', 'layer2.0.bn1',
        'layer4.1.conv2', 'layer4.1.bn2',
        'avgpool', 'fc'
    ]

    total_steps = 60
    stage_steps = {
        'beginning': 1,
        'middle': 30,
        'end': 60
    }

    all_stage_data = {}
    all_stage_metrics = {}

    print(f"Beginning training analysis for ResNet-18 on CIFAR-10. Total steps: {total_steps}...")
    global_step = 0
    epoch = 0

    while global_step < total_steps:
        epoch += 1
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step += 1
            if global_step > total_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # Check if this step is one of our target stages
            current_stage = None
            for stage_name, s_step in stage_steps.items():
                if global_step == s_step:
                    current_stage = stage_name
                    break

            if current_stage:
                print(f"\n>>> Capturing training data for stage: '{current_stage}' at step {global_step} (Epoch {epoch}) <<<")
                collector = TrainingDataCollector(model, target_layer_names=target_layers)
                collector.capture_weights_before()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Collect gradients before optimizer step
                weight_grads = collector.collect_weight_gradients()
                activation_grads = dict(collector.activation_grads)
                activations = dict(collector.activations)
                input_data = collector.inputs

                # Optimizer step and capture weight updates
                optimizer.step()
                collector.capture_updates_after()
                weight_updates = dict(collector.weight_updates)
                weights, biases = collector.collect_weights_and_biases()
                collector.remove_hooks()

                stage_record = {
                    'step': global_step,
                    'loss': float(loss.item()),
                    'inputs': input_data,
                    'weights': weights,
                    'activations': activations,
                    'weight_grads': weight_grads,
                    'activation_grads': activation_grads,
                    'weight_updates': weight_updates,
                    'biases': biases
                }
                all_stage_data[current_stage] = stage_record

                # Process metrics and generate plots for this stage
                stage_metrics = process_stage_data(stage_record, current_stage)
                all_stage_metrics[current_stage] = stage_metrics
                print(f"Captured and processed stage '{current_stage}' successfully. Loss: {loss.item():.4f}")
            else:
                # Normal training step
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if global_step % 25 == 0:
                print(f"Step {global_step}/{total_steps} - Epoch {epoch} - Loss: {loss.item():.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), './checkpoints/resnet18_cifar10.pth')
    print("ResNet-18 model saved to checkpoints/resnet18_cifar10.pth")

    # Save all metrics
    with open('./results/training_metrics.json', 'w') as f:
        json.dump(all_stage_metrics, f, indent=2)
    print("All training metrics saved to results/training_metrics.json")
    return all_stage_metrics

def process_stage_data(stage_record, stage_name):
    stage_metrics = {
        'step': stage_record['step'],
        'loss': stage_record['loss'],
        'data_types': {}
    }
    
    # 1. Inputs
    if stage_record['inputs'] is not None:
        m = compute_detailed_metrics(stage_record['inputs'], name=f"{stage_name}_inputs")
        stage_metrics['data_types']['inputs'] = m
        plot_dual_distribution(
            stage_record['inputs'],
            f"ResNet-18 Inputs ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_inputs.png",
            metrics=m, x_label="Input Pixel Value (Normalized)"
        )

    # 2. Weights (per target layer)
    stage_metrics['data_types']['weights'] = {}
    for name, w_arr in stage_record['weights'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(w_arr, name=f"{stage_name}_weight_{clean_name}")
        stage_metrics['data_types']['weights'][name] = m
        plot_dual_distribution(
            w_arr,
            f"ResNet-18 Weights: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_weights_{clean_name}.png",
            metrics=m, x_label="Weight Value"
        )

    # 3. Activations (per target layer)
    stage_metrics['data_types']['activations'] = {}
    for name, a_arr in stage_record['activations'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(a_arr, name=f"{stage_name}_activation_{clean_name}")
        stage_metrics['data_types']['activations'][name] = m
        plot_dual_distribution(
            a_arr,
            f"ResNet-18 Activations: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_activations_{clean_name}.png",
            metrics=m, x_label="Activation Value"
        )

    # 4. Weight Gradients
    stage_metrics['data_types']['weight_grads'] = {}
    for name, g_arr in stage_record['weight_grads'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(g_arr, name=f"{stage_name}_wgrad_{clean_name}")
        stage_metrics['data_types']['weight_grads'][name] = m
        plot_dual_distribution(
            g_arr,
            f"ResNet-18 Weight Gradients: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_weight_grads_{clean_name}.png",
            metrics=m, x_label="dL/dW Gradient Value"
        )

    # 5. Activation Gradients
    stage_metrics['data_types']['activation_grads'] = {}
    for name, ag_arr in stage_record['activation_grads'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(ag_arr, name=f"{stage_name}_agrad_{clean_name}")
        stage_metrics['data_types']['activation_grads'][name] = m
        plot_dual_distribution(
            ag_arr,
            f"ResNet-18 Activation Gradients: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_activation_grads_{clean_name}.png",
            metrics=m, x_label="dL/dA Gradient Value"
        )

    # 6. Weight Updates
    stage_metrics['data_types']['weight_updates'] = {}
    for name, u_arr in stage_record['weight_updates'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(u_arr, name=f"{stage_name}_wupdate_{clean_name}")
        stage_metrics['data_types']['weight_updates'][name] = m
        plot_dual_distribution(
            u_arr,
            f"ResNet-18 Weight Updates: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_weight_updates_{clean_name}.png",
            metrics=m, x_label="Weight Update Delta W"
        )

    # 7. Biases
    stage_metrics['data_types']['biases'] = {}
    for name, b_arr in stage_record['biases'].items():
        clean_name = name.replace('.', '_')
        m = compute_detailed_metrics(b_arr, name=f"{stage_name}_bias_{clean_name}")
        stage_metrics['data_types']['biases'][name] = m
        plot_dual_distribution(
            b_arr,
            f"ResNet-18 Biases / Stats: {name} ({stage_name.capitalize()} Stage)",
            f"plots/training/{stage_name}_biases_{clean_name}.png",
            metrics=m, x_label="Bias / Statistic Value"
        )

    return stage_metrics

if __name__ == "__main__":
    t0 = time.time()
    train_and_analyze_resnet()
    print(f"ResNet-18 training analysis completed in {time.time()-t0:.2f}s")

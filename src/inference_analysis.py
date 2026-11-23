import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from models.resnet import get_resnet18_cifar10
from models.mobilenetv2 import get_mobilenetv2_cifar100
from models.lenet5 import get_lenet5_mnist
from src.collector import InferenceDataCollector
from src.metrics import compute_detailed_metrics
from src.plotter import plot_dual_distribution

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def run_inference_analysis():
    results = {}
    
    # 1. ResNet-18 on CIFAR-10
    print("=" * 60)
    print("Starting Inference Analysis: ResNet-18 on CIFAR-10 (10,000 test samples)")
    print("=" * 60)
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_test = dset.CIFAR10(root='./data', train=False, transform=transform_cifar10, download=False)
    cifar10_loader = DataLoader(cifar10_test, batch_size=256, shuffle=False, num_workers=0)
    
    resnet = get_resnet18_cifar10()
    if os.path.exists('./checkpoints/resnet18_cifar10.pth'):
        resnet.load_state_dict(torch.load('./checkpoints/resnet18_cifar10.pth', map_location='cpu'))
        print("Loaded checkpoint: checkpoints/resnet18_cifar10.pth")
    resnet.eval()
    
    results['resnet18_cifar10'] = evaluate_model_inference(
        model=resnet,
        loader=cifar10_loader,
        model_name='resnet18_cifar10',
        dataset_name='CIFAR-10'
    )
    
    # 2. MobileNetV2 on CIFAR-100
    print("\n" + "=" * 60)
    print("Starting Inference Analysis: MobileNetV2 on CIFAR-100 (10,000 test samples)")
    print("=" * 60)
    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    cifar100_test = dset.CIFAR100(root='./data', train=False, transform=transform_cifar100, download=False)
    cifar100_loader = DataLoader(cifar100_test, batch_size=256, shuffle=False, num_workers=0)
    
    mobilenet = get_mobilenetv2_cifar100()
    if os.path.exists('./checkpoints/mobilenetv2_cifar100.pth'):
        mobilenet.load_state_dict(torch.load('./checkpoints/mobilenetv2_cifar100.pth', map_location='cpu'))
        print("Loaded checkpoint: checkpoints/mobilenetv2_cifar100.pth")
    mobilenet.eval()
    
    results['mobilenetv2_cifar100'] = evaluate_model_inference(
        model=mobilenet,
        loader=cifar100_loader,
        model_name='mobilenetv2_cifar100',
        dataset_name='CIFAR-100'
    )
    
    # 3. LeNet-5 on MNIST
    print("\n" + "=" * 60)
    print("Starting Inference Analysis: LeNet-5 on MNIST (10,000 test samples)")
    print("=" * 60)
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_test = dset.MNIST(root='./data', train=False, transform=transform_mnist, download=False)
    mnist_loader = DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=0)
    
    lenet = get_lenet5_mnist()
    if os.path.exists('./checkpoints/lenet5_mnist.pth'):
        lenet.load_state_dict(torch.load('./checkpoints/lenet5_mnist.pth', map_location='cpu'))
        print("Loaded checkpoint: checkpoints/lenet5_mnist.pth")
    lenet.eval()
    
    results['lenet5_mnist'] = evaluate_model_inference(
        model=lenet,
        loader=mnist_loader,
        model_name='lenet5_mnist',
        dataset_name='MNIST'
    )
    
    os.makedirs('./results', exist_ok=True)
    with open('./results/inference_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAll inference analysis completed and saved to results/inference_metrics.json")
    return results

def evaluate_model_inference(model, loader, model_name, dataset_name):
    collector = InferenceDataCollector(model, max_samples_per_layer=100000)
    all_inputs = []
    all_logits = []
    
    t0 = time.time()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # Sample inputs across dataset
            if batch_idx % 4 == 0:
                all_inputs.append(inputs.cpu().numpy().flatten())
            outputs = model(inputs)
            all_logits.append(outputs.cpu().numpy())
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += len(targets)

    inference_time = time.time() - t0
    accuracy = correct / total * 100.0
    print(f"[{model_name}] Inference pass finished in {inference_time:.2f}s. Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # 1. Weights and Biases (collected once)
    weights, biases = collector.collect_weights_and_biases()
    
    # 2. Aggregated Activations across full test set
    activations = collector.get_aggregated_activations()
    collector.remove_hooks()
    
    # 3. Concatenate Inputs and Logits
    concat_inputs = np.concatenate(all_inputs)
    concat_logits = np.concatenate(all_logits, axis=0).flatten()
    
    model_metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'test_accuracy': accuracy,
        'inference_time_seconds': inference_time,
        'layers_analyzed': list(activations.keys()),
        'metrics': {}
    }
    
    base_plot_dir = f"plots/inference/{model_name}"
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # Process Inputs
    m_in = compute_detailed_metrics(concat_inputs, name=f"{model_name}_inputs")
    model_metrics['metrics']['inputs'] = m_in
    plot_dual_distribution(
        concat_inputs,
        f"{model_name.upper()} Inputs ({dataset_name} Test Set)",
        f"{base_plot_dir}/inputs.png",
        metrics=m_in, x_label="Input Pixel Value (Normalized)"
    )
    
    # Process Logits
    m_logit = compute_detailed_metrics(concat_logits, name=f"{model_name}_logits")
    model_metrics['metrics']['output_logits'] = m_logit
    plot_dual_distribution(
        concat_logits,
        f"{model_name.upper()} Output Logits ({dataset_name} Test Set)",
        f"{base_plot_dir}/output_logits.png",
        metrics=m_logit, x_label="Logit Value"
    )
    
    # Process Weights
    model_metrics['metrics']['weights'] = {}
    for w_name, w_arr in weights.items():
        clean_name = w_name.replace('.', '_')
        m_w = compute_detailed_metrics(w_arr, name=f"{model_name}_weight_{clean_name}")
        model_metrics['metrics']['weights'][w_name] = m_w
        plot_dual_distribution(
            w_arr,
            f"{model_name.upper()} Weights: {w_name}",
            f"{base_plot_dir}/weight_{clean_name}.png",
            metrics=m_w, x_label="Weight Value"
        )
        
    # Process Biases
    model_metrics['metrics']['biases'] = {}
    for b_name, b_arr in biases.items():
        clean_name = b_name.replace('.', '_')
        m_b = compute_detailed_metrics(b_arr, name=f"{model_name}_bias_{clean_name}")
        model_metrics['metrics']['biases'][b_name] = m_b
        plot_dual_distribution(
            b_arr,
            f"{model_name.upper()} Biases/Stats: {b_name}",
            f"{base_plot_dir}/bias_{clean_name}.png",
            metrics=m_b, x_label="Bias / Statistic Value"
        )
        
    # Process Activations across all computational layers
    model_metrics['metrics']['activations'] = {}
    for act_name, act_arr in activations.items():
        clean_name = act_name.replace('.', '_')
        m_act = compute_detailed_metrics(act_arr, name=f"{model_name}_act_{clean_name}")
        model_metrics['metrics']['activations'][act_name] = m_act
        plot_dual_distribution(
            act_arr,
            f"{model_name.upper()} Activations: {act_name}",
            f"{base_plot_dir}/activation_{clean_name}.png",
            metrics=m_act, x_label="Activation Value"
        )
        
    print(f"[{model_name}] Computed metrics and generated plots for {len(weights)} weights, {len(biases)} biases, {len(activations)} layer activations, inputs, and logits.")
    return model_metrics

if __name__ == "__main__":
    t_start = time.time()
    run_inference_analysis()
    print(f"Total Inference Analysis completed in {time.time()-t_start:.2f}s")

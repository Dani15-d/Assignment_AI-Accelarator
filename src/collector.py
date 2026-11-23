import torch
import torch.nn as nn
import numpy as np

class TrainingDataCollector:
    """
    Hooks into PyTorch models to collect:
    - Inputs
    - Weights
    - Activations (forward pass)
    - Gradients (weights and activations)
    - Weight updates (W_t+1 - W_t)
    - Bias values
    """
    def __init__(self, model, target_layer_names=None):
        self.model = model
        self.target_layer_names = target_layer_names
        self.activations = {}
        self.activation_grads = {}
        self.weight_grads = {}
        self.weights_before = {}
        self.weight_updates = {}
        self.inputs = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook on input
        def input_hook(module, args):
            if self.inputs is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                self.inputs = args[0].detach().cpu().numpy()

        self.hooks.append(self.model.register_forward_pre_hook(input_hook))

        for name, module in self.model.named_modules():
            if self.target_layer_names is not None and name not in self.target_layer_names:
                continue
            
            # Forward hook for activations
            def get_fwd_hook(l_name):
                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        self.activations[l_name] = out.detach().cpu().numpy()
                return hook

            # Backward hook for activation gradients
            def get_bwd_hook(l_name):
                def hook(mod, grad_in, grad_out):
                    if len(grad_out) > 0 and grad_out[0] is not None:
                        self.activation_grads[l_name] = grad_out[0].detach().cpu().numpy()
                return hook

            # Only hook meaningful computational layers
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                self.hooks.append(module.register_forward_hook(get_fwd_hook(name)))
                self.hooks.append(module.register_full_backward_hook(get_bwd_hook(name)))

    def capture_weights_before(self):
        """Saves current weights before optimizer.step()"""
        self.weights_before = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                self.weights_before[name] = param.detach().cpu().clone()

    def capture_updates_after(self):
        """Computes weight updates = W_new - W_old after optimizer.step()"""
        self.weight_updates = {}
        for name, param in self.model.named_parameters():
            if name in self.weights_before:
                old_w = self.weights_before[name]
                new_w = param.detach().cpu()
                self.weight_updates[name] = (new_w - old_w).numpy()

    def collect_weights_and_biases(self):
        weights = {}
        biases = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights[name] = param.detach().cpu().numpy()
            elif 'bias' in name:
                biases[name] = param.detach().cpu().numpy()
        
        # Also collect BatchNorm running mean and variance as bias/stat equivalents
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.running_mean is not None:
                    biases[f"{name}.running_mean"] = module.running_mean.detach().cpu().numpy()
                if module.running_var is not None:
                    biases[f"{name}.running_var"] = module.running_var.detach().cpu().numpy()
        return weights, biases

    def collect_weight_gradients(self):
        w_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'weight' in name:
                w_grads[name] = param.grad.detach().cpu().numpy()
        return w_grads

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class InferenceDataCollector:
    """
    Collects activations, inputs, logits, weights, and biases across full test dataset.
    """
    def __init__(self, model, max_samples_per_layer=100000):
        self.model = model
        self.max_samples_per_layer = max_samples_per_layer
        self.layer_samples = {}
        self.layer_modules = {}
        self.hooks = []
        self._find_computational_layers()
        self._register_hooks()

    def _find_computational_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                self.layer_modules[name] = module
                self.layer_samples[name] = []

    def _register_hooks(self):
        for name, module in self.layer_modules.items():
            def make_hook(layer_name):
                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        flat = out.detach().cpu().flatten().numpy()
                        # Reservoir / subsample per batch to accumulate a representative global sample
                        if len(flat) > 2000:
                            sub_idx = np.random.choice(len(flat), size=2000, replace=False)
                            self.layer_samples[layer_name].append(flat[sub_idx])
                        else:
                            self.layer_samples[layer_name].append(flat)
                return hook
            self.hooks.append(module.register_forward_hook(make_hook(name)))

    def get_aggregated_activations(self):
        aggregated = {}
        for name, list_of_arrs in self.layer_samples.items():
            if len(list_of_arrs) > 0:
                cat = np.concatenate(list_of_arrs)
                if len(cat) > self.max_samples_per_layer:
                    idx = np.random.choice(len(cat), size=self.max_samples_per_layer, replace=False)
                    aggregated[name] = cat[idx]
                else:
                    aggregated[name] = cat
        return aggregated

    def collect_weights_and_biases(self):
        weights = {}
        biases = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights[name] = param.detach().cpu().numpy()
            elif 'bias' in name:
                biases[name] = param.detach().cpu().numpy()

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.running_mean is not None:
                    biases[f"{name}.running_mean"] = module.running_mean.detach().cpu().numpy()
                if module.running_var is not None:
                    biases[f"{name}.running_var"] = module.running_var.detach().cpu().numpy()
        return weights, biases

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

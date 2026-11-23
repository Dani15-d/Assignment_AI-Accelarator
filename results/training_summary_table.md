# ResNet-18 Training Numerical Range & Precision Analysis

## 1. Evolution Across Training Stages (Beginning, Middle, End)

| Stage | Data Type | Min Value | Max Value | Smallest |x|>0 | Min Spacing | Zero Sparsity | INT8 SQNR (dB) | INT16 SQNR (dB) | FP8 E4M3 Underflow % | FP16 Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Beginning | Inputs | -2.4291e+00 | 2.7537e+00 | 7.6703e-04 | 3.2187e-06 | 0.0% | 46.7 | 95.4 | 0.2% | Adequate |
| Beginning | Weights (layer1.0.conv1.weight) | -4.3982e-02 | 4.4223e-02 | 6.0584e-07 | 4.6566e-10 | 0.0% | 47.6 | 95.8 | 4.8% | Adequate |
| Beginning | Activations (layer1.0.conv1) | -3.1444e+00 | 3.1487e+00 | 7.4506e-09 | 3.7253e-09 | 0.0% | 34.9 | 83.1 | 0.5% | Adequate |
| Beginning | Weight Grads (layer1.0.conv1.weight) | -8.3081e-02 | 7.8294e-02 | 6.1782e-07 | 1.1642e-10 | 0.0% | 37.9 | 86.1 | 12.7% | Adequate |
| Beginning | Act Grads (layer1.0.conv1) | -1.1820e-03 | 9.3826e-04 | 3.4665e-14 | 8.8818e-16 | 0.0% | 32.4 | 77.6 | 100.0% | Caution (Underflow/Overflow) |
| Beginning | Weight Updates (layer1.0.conv1.weight) | -3.9156e-03 | 4.1536e-03 | 1.4901e-08 | 1.1642e-10 | 0.0% | 37.9 | 86.0 | 98.5% | Adequate |
| Beginning | Biases (fc.bias) | -4.3413e-02 | 1.3696e-02 | 6.8841e-03 | 3.2886e-05 | 0.0% | 50.2 | 97.9 | 0.0% | Adequate |
| Middle | Inputs | -2.4291e+00 | 2.7537e+00 | 7.6703e-04 | 3.2187e-06 | 0.0% | 47.2 | 95.8 | 0.1% | Adequate |
| Middle | Weights (layer1.0.conv1.weight) | -1.0425e-01 | 1.2991e-01 | 6.2616e-07 | 4.6566e-10 | 0.0% | 39.5 | 87.8 | 4.7% | Adequate |
| Middle | Activations (layer1.0.conv1) | -1.4257e+01 | 1.1964e+01 | 3.2992e-07 | 2.7940e-09 | 0.0% | 36.3 | 84.5 | 0.1% | Adequate |
| Middle | Weight Grads (layer1.0.conv1.weight) | -1.7096e-02 | 2.0655e-02 | 2.1823e-08 | 2.1828e-11 | 0.0% | 37.9 | 86.1 | 55.7% | Adequate |
| Middle | Act Grads (layer1.0.conv1) | -4.2400e-04 | 3.1635e-04 | 1.6634e-15 | 4.4409e-16 | 0.0% | 20.1 | 62.0 | 100.0% | Caution (Underflow/Overflow) |
| Middle | Weight Updates (layer1.0.conv1.weight) | -3.2768e-03 | 4.0552e-03 | 1.8626e-08 | 2.1828e-11 | 0.0% | 35.6 | 83.7 | 99.2% | Adequate |
| Middle | Biases (fc.bias) | -7.3506e-02 | 3.3728e-02 | 1.0852e-02 | 8.0768e-04 | 0.0% | 46.8 | 95.6 | 0.0% | Adequate |
| End | Inputs | -2.4291e+00 | 2.7537e+00 | 7.6703e-04 | 3.2187e-06 | 0.0% | 46.8 | 95.6 | 0.2% | Adequate |
| End | Weights (layer1.0.conv1.weight) | -1.2084e-01 | 1.4020e-01 | 2.0334e-06 | 6.9849e-10 | 0.0% | 39.3 | 87.6 | 4.6% | Adequate |
| End | Activations (layer1.0.conv1) | -1.6477e+01 | 1.3951e+01 | 3.8743e-07 | 1.8626e-09 | 0.0% | 36.5 | 84.8 | 0.1% | Adequate |
| End | Weight Grads (layer1.0.conv1.weight) | -1.1933e-02 | 8.6704e-03 | 5.3264e-09 | 7.2760e-12 | 0.0% | 31.6 | 79.9 | 94.0% | Adequate |
| End | Act Grads (layer1.0.conv1) | -5.3310e-05 | 5.2674e-05 | 4.8465e-15 | 1.1102e-16 | 0.0% | 20.6 | 64.8 | 100.0% | Caution (Underflow/Overflow) |
| End | Weight Updates (layer1.0.conv1.weight) | -8.7648e-04 | 9.1780e-04 | 1.1176e-08 | 2.9104e-11 | 0.0% | 35.1 | 83.3 | 100.0% | Adequate |
| End | Biases (fc.bias) | -9.5904e-02 | 5.8694e-02 | 5.9336e-03 | 3.3596e-03 | 0.0% | 47.3 | 96.4 | 0.0% | Adequate |
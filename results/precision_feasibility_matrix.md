# Precision Representation Feasibility Analysis

## Summary of Evaluated Numerical Formats

| Representation | Format Structure | Dynamic Range | Smallest Nonzero (Normalized) | Smallest Nonzero (Subnormal) | Step Size / Precision Resolution | Primary Application Scope |
|---|---|---|---|---|---|---|
| **FP32** | 1 sign, 8 exp, 23 mantissa | $\approx 10^{\pm 38}$ | $1.18 \times 10^{-38}$ | $1.40 \times 10^{-45}$ | $2^{-23} \approx 1.19 \times 10^{-7}$ (relative) | Baseline Reference (Full Precision) |
| **FP16** | 1 sign, 5 exp, 10 mantissa | $[-65504, +65504]$ | $6.10 \times 10^{-5}$ | $5.96 \times 10^{-8}$ | $2^{-10} \approx 9.77 \times 10^{-4}$ (relative) | Training (with Loss Scale) & Inference |
| **FP8 (E4M3)** | 1 sign, 4 exp, 3 mantissa | $[-448, +448]$ | $1.56 \times 10^{-2}$ | $1.95 \times 10^{-3}$ | $2^{-3} = 0.125$ (relative) | Inference & Forward Pass Training |
| **FP8 (E5M2)** | 1 sign, 5 exp, 2 mantissa | $[-57344, +57344]$ | $6.10 \times 10^{-5}$ | $1.53 \times 10^{-5}$ | $2^{-2} = 0.25$ (relative) | Backward Pass (Gradients) with Scaling |
| **INT16** | 1 sign, 15 value bits | $[-32768, +32767]$ | Uniform step: $S = \frac{\max \|x\|}{32767}$ | $S$ | Uniform step $S$ | High-fidelity Fixed-Point Inference |
| **INT8** | 1 sign, 7 value bits | $[-128, +127]$ | Uniform step: $S = \frac{\max \|x\|}{127}$ | $S$ | Uniform step $S$ | Efficient Hardware Accelerator Inference |

## Feasibility Across DNN Data Types

| Data Type | Stage | Feasibility FP16 | Feasibility FP8 (E4M3) | Feasibility FP8 (E5M2) | Feasibility INT16 | Feasibility INT8 | Key Findings & Recommendations |
|---|---|---|---|---|---|---|---|
| **Inputs** | Inference / Train | **Adequate** (0% loss) | **Adequate** ($<0.1\%$ underflow) | **Adequate** | **Adequate** ($>80$ dB) | **Adequate** ($>45$ dB) | Normalized inputs $[-2.5, +2.8]$ are well bounded. Symmetric INT8 or UINT8 is completely adequate. |
| **Weights** | Inference | **Adequate** (0% loss) | **Adequate** (SQNR $>38$ dB) | **Adequate** | **Adequate** ($>85$ dB) | **Adequate** (SQNR $39-48$ dB) | Weights remain small and bounded in $[-0.15, +0.15]$. INT8 with per-channel symmetric scaling preserves accuracy without degradation. |
| **Activations (Conv/Linear)** | Inference | **Adequate** (0% loss) | **Adequate** | **Adequate** | **Adequate** ($>80$ dB) | **Adequate** (SQNR $35-40$ dB) | Dynamic range expands up to $\pm 16$. INT8 per-tensor or per-token scaling handles activations effectively. |
| **Activations (ReLU)** | Inference | **Adequate** (0% loss) | **Adequate** | **Adequate** | **Adequate** ($>80$ dB) | **Adequate** (SQNR $38-42$ dB) | Non-negative values with $40-62\%$ exact zeros. Unsigned INT8 (UINT8: $[0, 255]$) provides double the resolution. |
| **Activations (BatchNorm)** | Inference | **Adequate** (0% loss) | **Adequate** | **Adequate** | **Adequate** ($>80$ dB) | **Adequate** (SQNR $36-39$ dB) | BatchNorm rescales activations to zero mean and unit variance $[-4.5, +4.7]$, keeping them in the optimal zone for INT8. |
| **Output Logits** | Inference | **Adequate** (0% loss) | **Caution** (resolution) | **Adequate** | **Adequate** ($>85$ dB) | **Adequate** (SQNR $34-43$ dB) | Spanning $[-21, +22]$. High dynamic range; INT16 or FP16 recommended before Softmax to avoid margin distortion. |
| **Weight Gradients** | Training (Early) | **Adequate** | **Lossy** (12.7% underflow) | **Adequate** | **Marginal** (37.9 dB) | **Lossy** (vanishing) | Gradient values span $[-0.08, +0.08]$ with smallest non-zero $\approx 10^{-7}$. Needs FP16 or FP8 E5M2. |
| **Weight Gradients** | Training (End) | **Adequate** | **Failure** (94.0% underflow) | **Marginal** (with scale) | **Lossy** | **Failure** | Gradients drop to $\approx 10^{-9}$. Unscaled FP8 E4M3 loses $94\%$ of gradients. Requires dynamic loss scaling. |
| **Activation Gradients** | Training (All) | **Adequate** (with scale) | **Failure** (100% underflow) | **Failure** (without scale) | **Failure** | **Failure** | Magnitudes drop to $10^{-14} - 10^{-16}$. Cannot be represented in INT8/INT16. FP16 with dynamic loss scaling is mandatory. |
| **Weight Updates** | Training (All) | **Caution** (truncation) | **Failure** (100% underflow) | **Failure** | **Failure** | **Failure** | Updates $\Delta W = \eta \nabla W$ are tiny ($\approx 10^{-6} - 10^{-8}$). FP32 master weights or stochastic rounding is strictly required. |

import numpy as np
import torch

def compute_detailed_metrics(data_tensor, name="data", max_samples_for_spacing=50000):
    """
    Computes statistical range, precision metrics, and representation feasibility.
    data_tensor: PyTorch tensor or numpy array
    """
    if isinstance(data_tensor, torch.Tensor):
        arr = data_tensor.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        arr = np.asarray(data_tensor).flatten().astype(np.float32)

    total_count = len(arr)
    if total_count == 0:
        return {}

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    abs_arr = np.abs(arr)
    abs_max = float(np.max(abs_arr))
    
    # Non-zero values
    non_zeros = abs_arr[abs_arr > 0]
    zero_count = int(np.sum(arr == 0))
    zero_ratio = float(zero_count / total_count)

    if len(non_zeros) > 0:
        smallest_nonzero = float(np.min(non_zeros))
        dyn_range_ratio = abs_max / smallest_nonzero if smallest_nonzero > 0 else float('inf')
        dyn_range_db = float(20.0 * np.log10(dyn_range_ratio)) if dyn_range_ratio > 0 else 0.0
        dyn_range_bits = float(np.log2(dyn_range_ratio)) if dyn_range_ratio > 0 else 0.0
    else:
        smallest_nonzero = 0.0
        dyn_range_ratio = 0.0
        dyn_range_db = 0.0
        dyn_range_bits = 0.0

    # Spacing between distinct values
    # For large arrays, sample to compute distinct value spacing efficiently
    if total_count > max_samples_for_spacing:
        sample_indices = np.random.choice(total_count, size=max_samples_for_spacing, replace=False)
        sample = np.sort(arr[sample_indices])
    else:
        sample = np.sort(arr)

    # Unique values
    unique_vals = np.unique(sample)
    if len(unique_vals) > 1:
        diffs = np.diff(unique_vals)
        diffs_pos = diffs[diffs > 0]
        if len(diffs_pos) > 0:
            min_spacing = float(np.min(diffs_pos))
            mean_spacing = float(np.mean(diffs_pos))
            median_spacing = float(np.median(diffs_pos))
        else:
            min_spacing = 0.0
            mean_spacing = 0.0
            median_spacing = 0.0
    else:
        min_spacing = 0.0
        mean_spacing = 0.0
        median_spacing = 0.0

    # Percentiles
    p_keys = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9]
    percentiles = {f"p_{p}": float(np.percentile(arr, p)) for p in p_keys}

    # Precision representations feasibility evaluation
    feasibility = evaluate_precision_feasibility(arr, abs_max, smallest_nonzero)

    metrics = {
        "name": name,
        "count": total_count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "abs_max": abs_max,
        "smallest_nonzero": smallest_nonzero,
        "zero_count": zero_count,
        "zero_ratio": zero_ratio,
        "min_spacing": min_spacing,
        "mean_spacing": mean_spacing,
        "median_spacing": median_spacing,
        "dynamic_range_db": dyn_range_db,
        "dynamic_range_bits": dyn_range_bits,
        "percentiles": percentiles,
        "feasibility": feasibility
    }
    return metrics

def evaluate_precision_feasibility(arr, abs_max, smallest_nonzero):
    """
    Evaluates representation capacity for FP16, FP8 (E4M3, E5M2), INT16, and INT8.
    """
    total = len(arr)
    if total == 0 or abs_max == 0:
        return {
            "fp16": {"overflow_pct": 0.0, "underflow_pct": 0.0, "status": "Adequate"},
            "fp8_e4m3": {"overflow_pct": 0.0, "underflow_pct": 0.0, "status": "Adequate"},
            "fp8_e5m2": {"overflow_pct": 0.0, "underflow_pct": 0.0, "status": "Adequate"},
            "int16": {"sqnr_db": float('inf'), "zero_flush_pct": 0.0, "status": "Adequate"},
            "int8": {"sqnr_db": float('inf'), "zero_flush_pct": 0.0, "status": "Adequate"},
        }

    abs_arr = np.abs(arr)
    non_zeros = abs_arr[abs_arr > 0]
    num_nonzeros = len(non_zeros)

    # 1. FP16: max 65504, min subnormal ~5.96e-8, min norm ~6.10e-5
    fp16_max = 65504.0
    fp16_min = 5.9604644775390625e-8
    fp16_overflow = np.sum(abs_arr > fp16_max) / total * 100.0
    fp16_underflow = np.sum((abs_arr > 0) & (abs_arr < fp16_min)) / num_nonzeros * 100.0 if num_nonzeros > 0 else 0.0

    # 2. FP8 (E4M3): max 448.0, min subnormal ~0.001953125 (2^-9), min norm 0.015625
    e4m3_max = 448.0
    e4m3_min = 0.001953125
    e4m3_overflow = np.sum(abs_arr > e4m3_max) / total * 100.0
    e4m3_underflow = np.sum((abs_arr > 0) & (abs_arr < e4m3_min)) / num_nonzeros * 100.0 if num_nonzeros > 0 else 0.0

    # 3. FP8 (E5M2): max 57344.0, min subnormal ~1.5258789e-5 (2^-16), min norm 6.1035156e-5
    e5m2_max = 57344.0
    e5m2_min = 1.52587890625e-5
    e5m2_overflow = np.sum(abs_arr > e5m2_max) / total * 100.0
    e5m2_underflow = np.sum((abs_arr > 0) & (abs_arr < e5m2_min)) / num_nonzeros * 100.0 if num_nonzeros > 0 else 0.0

    # 4. INT16: Symmetric scale S = abs_max / 32767
    scale_int16 = abs_max / 32767.0
    int16_q = np.round(np.clip(arr / scale_int16, -32768, 32767)) * scale_int16
    err_int16 = arr - int16_q
    signal_pwr = np.mean(arr ** 2)
    noise_pwr_16 = np.mean(err_int16 ** 2)
    sqnr_int16 = float(10.0 * np.log10(signal_pwr / noise_pwr_16)) if noise_pwr_16 > 0 else 999.0
    int16_flush_zero = np.sum((abs_arr > 0) & (np.abs(int16_q) == 0)) / num_nonzeros * 100.0 if num_nonzeros > 0 else 0.0

    # 5. INT8: Symmetric scale S = abs_max / 127
    scale_int8 = abs_max / 127.0
    int8_q = np.round(np.clip(arr / scale_int8, -128, 127)) * scale_int8
    err_int8 = arr - int8_q
    noise_pwr_8 = np.mean(err_int8 ** 2)
    sqnr_int8 = float(10.0 * np.log10(signal_pwr / noise_pwr_8)) if noise_pwr_8 > 0 else 999.0
    int8_flush_zero = np.sum((abs_arr > 0) & (np.abs(int8_q) == 0)) / num_nonzeros * 100.0 if num_nonzeros > 0 else 0.0

    return {
        "fp16": {
            "overflow_pct": float(fp16_overflow),
            "underflow_pct": float(fp16_underflow),
            "status": "Adequate" if (fp16_overflow == 0 and fp16_underflow < 0.1) else "Caution (Underflow/Overflow)"
        },
        "fp8_e4m3": {
            "overflow_pct": float(e4m3_overflow),
            "underflow_pct": float(e4m3_underflow),
            "status": "Adequate" if (e4m3_overflow == 0 and e4m3_underflow < 1.0) else "Lossy (Severe Underflow/Overflow)"
        },
        "fp8_e5m2": {
            "overflow_pct": float(e5m2_overflow),
            "underflow_pct": float(e5m2_underflow),
            "status": "Adequate" if (e5m2_overflow == 0 and e5m2_underflow < 1.0) else "Lossy (Underflow)"
        },
        "int16": {
            "scale": float(scale_int16),
            "sqnr_db": float(sqnr_int16),
            "zero_flush_pct": float(int16_flush_zero),
            "status": "Adequate" if sqnr_int16 > 40.0 else "Marginal"
        },
        "int8": {
            "scale": float(scale_int8),
            "sqnr_db": float(sqnr_int8),
            "zero_flush_pct": float(int8_flush_zero),
            "status": "Adequate" if sqnr_int8 > 25.0 else "Lossy"
        }
    }

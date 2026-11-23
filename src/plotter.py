import os
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_dual_distribution(data, title, save_path, metrics=None, x_label="Numerical Value", max_points=200000):
    """
    Creates a dual-panel figure:
    1. Complete plot showing overall numerical range
    2. Magnified plot around zero to examine small and closely spaced values
    Annotates min, max, smallest non-zero magnitude, and min spacing.
    """
    if isinstance(data, (list, tuple)):
        data = np.asarray(data)
    flat = data.flatten()
    if len(flat) > max_points:
        flat = np.random.choice(flat, size=max_points, replace=False)

    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # 1. Full Range Distribution
    # Use adaptive histogram bins
    min_v = np.min(flat)
    max_v = np.max(flat)
    
    if min_v == max_v:
        bins_full = np.linspace(min_v - 1, max_v + 1, 50)
    else:
        bins_full = np.linspace(min_v, max_v, 100)

    counts, bin_edges, _ = ax_full.hist(
        flat, bins=bins_full, density=True, color='#2b5c8f', alpha=0.75, edgecolor='#1b3b5c', linewidth=0.5
    )
    ax_full.set_title("Full Numerical Range", fontsize=11, fontweight='semibold')
    ax_full.set_xlabel(x_label, fontsize=10)
    ax_full.set_ylabel("Probability Density", fontsize=10)
    ax_full.grid(True, linestyle='--', alpha=0.5)

    # Annotation text on full range
    if metrics is not None:
        info_text = (
            f"Count: {metrics.get('count', len(flat)):,}\n"
            f"Min: {metrics.get('min', min_v):.4e}\n"
            f"Max: {metrics.get('max', max_v):.4e}\n"
            f"Mean: {metrics.get('mean', np.mean(flat)):.4e}\n"
            f"Std: {metrics.get('std', np.std(flat)):.4e}\n"
            f"Smallest |x|>0: {metrics.get('smallest_nonzero', 0):.4e}\n"
            f"Min Spacing: {metrics.get('min_spacing', 0):.4e}\n"
            f"Zero Sparsity: {metrics.get('zero_ratio', 0)*100:.1f}%"
        )
        ax_full.text(
            0.96, 0.94, info_text, transform=ax_full.transAxes,
            fontsize=8.5, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
        )

    # 2. Magnified Plot Around Zero
    # Determine zoom range around zero
    # Look at non-zero values or narrow quantile near zero
    std = np.std(flat)
    if std > 0:
        zoom_limit = min(max(abs(min_v), abs(max_v)), 0.5 * std, 0.05)
        # ensure zoom limit captures non-zero values
        if zoom_limit == 0 or np.isnan(zoom_limit):
            zoom_limit = 0.01
    else:
        zoom_limit = 0.01

    zoom_mask = (flat >= -zoom_limit) & (flat <= zoom_limit)
    zoom_data = flat[zoom_mask]

    if len(zoom_data) > 10:
        bins_zoom = np.linspace(-zoom_limit, zoom_limit, 80)
        ax_zoom.hist(
            zoom_data, bins=bins_zoom, density=True, color='#d95f02', alpha=0.75, edgecolor='#a04000', linewidth=0.5
        )
        ax_zoom.set_xlim(-zoom_limit, zoom_limit)
    else:
        # Fallback to 10th-90th percentiles if zoom_mask too sparse
        p10, p90 = np.percentile(flat, 10), np.percentile(flat, 90)
        bins_zoom = np.linspace(p10, p90, 80)
        ax_zoom.hist(
            flat, bins=bins_zoom, density=True, color='#d95f02', alpha=0.75, edgecolor='#a04000', linewidth=0.5
        )
        ax_zoom.set_xlim(p10, p90)

    ax_zoom.set_title("Magnified Around Zero", fontsize=11, fontweight='semibold')
    ax_zoom.set_xlabel(f"{x_label} (Magnified)", fontsize=10)
    ax_zoom.set_ylabel("Probability Density", fontsize=10)
    ax_zoom.grid(True, linestyle='--', alpha=0.5)
    ax_zoom.axvline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.8, label="Zero line")
    ax_zoom.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

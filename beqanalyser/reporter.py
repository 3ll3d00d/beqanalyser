import logging
from itertools import groupby

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from beqanalyser import rms, AssignmentRecord, BEQFilter, BEQCompositePipelineResult, RejectionReason

logger = logging.getLogger()


def summarize_assignments(result: BEQCompositePipelineResult) -> None:
    logger.info(f"Total catalogue entries: {len(result.assignment_table)}")

    logger.info("\nComposite assignment counts:")
    for i, comp in enumerate(result.composites):
        logger.info(f"  Composite {i + 1}: {len(comp.assigned_indices)} assigned")

    logger.info(f"\nTotal assigned: {sum(len(c.assigned_indices) for c in result.composites)}")
    logger.info(f"Total rejected: {len([i for i in result.assignment_table.values() if i.rejected])}")

    logger.info("\nRejection breakdown:")
    reason_counts: dict[RejectionReason, int] = {x: len(list(y)) for x, y in groupby(
        sorted([a for a in result.assignment_table.values() if a.rejection_reason is not None],
               key=lambda b: b.rejection_reason), lambda x: x.rejection_reason)}
    for reason, count in dict(sorted(reason_counts.items(), key=lambda item: item[1])).items():
        logger.info(f"  {reason.name}: {count}")


def plot_assigned_fan_curves(catalogue: np.ndarray,
                             result: BEQCompositePipelineResult,
                             freqs: np.ndarray) -> None:
    """Plot assigned fan curves and composite shapes."""
    n_comps = len(result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

    for i, comp in enumerate(result.composites):
        ax: Axes = axes[i]

        # Fan curves
        for lvl, fan_curves in enumerate(comp.fan_envelopes):
            if fan_curves.size == 0:
                continue
            alpha = 0.2 + 0.6 * lvl / max(1, len(comp.fan_envelopes) - 1)
            for curve in fan_curves:
                ax.plot(freqs, curve, color='lightblue', lw=1, alpha=alpha, zorder=1)

        # Composite
        ax.plot(freqs, comp.shape, color='black', lw=2, label='Composite', zorder=3)

        ax.set_title(f"Composite {i + 1} ({len(comp.assigned_indices)} assigned)")
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel('Magnitude (dB)')
        if i >= ncols * (nrows - 1):
            ax.set_xlabel('Frequency (Hz)')

        # Inset histogram for RMS of assigned curves
        inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
        assigned_rms = np.array([rms(catalogue[idx] - comp.shape) for idx in comp.assigned_indices])
        if len(assigned_rms) > 0:
            inset.hist(assigned_rms, bins=15, color='lightblue', alpha=0.7)
        inset.set_title('Assigned RMS', fontsize=8)
        inset.tick_params(axis='both', labelsize=6)

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=True)
    fig.suptitle("Assigned Fan Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_rejected_by_reason(catalogue: np.ndarray,
                            result: BEQCompositePipelineResult,
                            freqs: np.ndarray) -> None:
    """Plot rejected curves per rejection reason with metric-specific histograms."""
    n_comps = len(result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    reasons = list(RejectionReason)

    for reason in reasons:
        if not any(r for r in result.assignment_table.values() if r.rejected and r.rejection_reason == reason):
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

        for i, comp in enumerate(result.composites):
            ax: Axes = axes[i]

            # Rejected entries of this reason for this composite
            rejected_entries = [r for r in result.assignment_table.values()
                                if r.rejected and r.assigned_composite == i and r.rejection_reason == reason]
            if not rejected_entries:
                continue

            rejected_indices = [r.entry_index for r in rejected_entries]
            curves = catalogue[rejected_indices]

            # Fan-style plotting
            rms_vals = np.array([rms(c - comp.shape) for c in curves])
            sort_idx = np.argsort(rms_vals)
            for j, idx_r in enumerate(sort_idx):
                alpha = 0.2 + 0.3 * j / max(1, len(sort_idx) - 1)
                ax.plot(freqs, curves[idx_r], color='lightcoral', lw=1, alpha=alpha, zorder=1)

            # Composite overlay
            ax.plot(freqs, comp.shape, color='black', lw=2, zorder=2)

            # Inset histogram for metric that triggered rejection
            inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
            if reason == RejectionReason.RMS_EXCEEDED:
                vals = np.array([r.rms_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='lightblue', alpha=0.7)
                inset.set_title('RMS', fontsize=8)
            elif reason == RejectionReason.MAX_EXCEEDED:
                vals = np.array([r.max_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='salmon', alpha=0.7)
                inset.set_title('Max', fontsize=8)
            elif reason == RejectionReason.BOTH_EXCEEDED:
                rms_vals = np.array([r.rms_value for r in rejected_entries])
                max_vals = np.array([r.max_value for r in rejected_entries])
                inset.hist(rms_vals, bins=15, color='lightblue', alpha=0.7, label='RMS')
                inset.hist(max_vals, bins=15, color='salmon', alpha=0.7, label='Max')
                inset.set_title('RMS/Max', fontsize=8)
                inset.legend(fontsize=6)
            elif reason == RejectionReason.COSINE_TOO_LOW:
                vals = np.array([1 - r.cosine_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='violet', alpha=0.7)
                inset.set_title('1 - Cosine', fontsize=8)
            elif reason == RejectionReason.DERIVATIVE_TOO_HIGH:
                vals = np.array([r.derivative_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='orange', alpha=0.7)
                inset.set_title('Derivative', fontsize=8)

            inset.tick_params(axis='both', labelsize=6)
            ax.set_title(f"Composite {i + 1} ({len(rejected_entries)} rejected)")
            ax.grid(True, alpha=0.3)
            if i % ncols == 0:
                ax.set_ylabel('Magnitude (dB)')
            if i >= ncols * (nrows - 1):
                ax.set_xlabel('Frequency (Hz)')

        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        fig.suptitle(f"Rejected Curves by Composite — Reason: {reason.name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def plot_all_beq_curves(catalogue: np.ndarray,
                        result: BEQCompositePipelineResult,
                        freqs: np.ndarray) -> None:
    """Convenience function to plot assigned and rejected curves."""
    plot_assigned_fan_curves(catalogue, result, freqs)
    plot_rejected_by_reason(catalogue, result, freqs)


# ------------------------------
# RMS vs Max scatter with density
# ------------------------------
def plot_rms_max_scatter(result: BEQCompositePipelineResult) -> None:
    all_rms: np.ndarray = np.concatenate([np.array(c.deltas) for c in result.composites])
    all_max: np.ndarray = np.concatenate([np.array(c.max_deltas) for c in result.composites])

    xy: np.ndarray = np.vstack([all_rms, all_max])
    kde: np.ndarray = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 5))
    plt.scatter(all_rms, all_max, c=kde, s=20, cmap='viridis')
    plt.xlabel('RMS deviation (dB)')
    plt.ylabel('Max deviation (dB)')
    plt.title('RMS vs Max-deviation scatter with density')
    plt.grid(True, alpha=0.3)
    plt.show()


# ------------------------------
# Histograms from assignment table
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histograms_from_table(assignment_table: list[AssignmentRecord]) -> None:
    rms_vals: list[float] = [r.rms_value for r in assignment_table if r.rms_value is not None]
    max_vals: list[float] = [r.max_value for r in assignment_table if r.max_value is not None]
    cosine_vals: list[float] = [r.cosine_value for r in assignment_table if r.cosine_value is not None]

    fig: Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Helper to add percentile lines and annotate them slightly offset
    def add_percentile_lines(ax, data):
        ylim = ax.get_ylim()
        for p in [50, 90, 95]:
            val = np.percentile(data, p)
            ax.axvline(val, color='lightgrey', linestyle='dotted', linewidth=1)
            # Small horizontal offset to avoid overlapping the line
            ax.text(val + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ylim[1] * 0.95,
                    f'{p}%', rotation=90, verticalalignment='top',
                    color='grey', fontsize=8)

    # RMS histogram
    axes[0].hist(rms_vals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('RMS deviation (dB)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('RMS deviation')
    add_percentile_lines(axes[0], rms_vals)

    # Max deviation histogram
    axes[1].hist(max_vals, bins=30, color='salmon', edgecolor='black')
    axes[1].set_xlabel('Max deviation (dB)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Max Deviation')
    add_percentile_lines(axes[1], max_vals)

    # Cosine similarity histogram
    axes[2].hist(cosine_vals, bins=100, color='palegreen', edgecolor='black')
    axes[2].set_xlabel('Cosine')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Cosine Similarity')
    add_percentile_lines(axes[2], cosine_vals)

    plt.tight_layout()
    plt.show()


def print_assignments(result: BEQCompositePipelineResult, filters: list[BEQFilter]) -> None:
    with open('beq_composites.csv', 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(
            ['composite_id', 'reject reason', 'content type', 'author', 'title', 'year', 'cosine', 'rms',
             'max', 'derivative', 'digest', 'beqc url'])
        for idx, rec in result.assignment_table.items():
            underlying: BEQFilter = filters[idx]
            writer.writerow([
                rec.assigned_composite,
                rec.rejection_reason.name if rec.rejection_reason else '',
                underlying.entry.content_type,
                underlying.entry.author,
                underlying.entry.formatted_title,
                underlying.entry.year,
                rec.cosine_value,
                rec.rms_value,
                rec.max_value,
                rec.derivative_value,
                underlying.entry.digest,
                underlying.entry.beqc_url,
            ])

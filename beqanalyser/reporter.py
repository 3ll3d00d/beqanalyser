import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from scipy.stats import gaussian_kde

from beqanalyser import (
    BEQCompositeComputation,
    BEQFilter,
    ComputationCycle,
    RejectionReason,
)

logger = logging.getLogger()


def summarize_assignments(computation: BEQCompositeComputation) -> None:
    logger.info(f"Total catalogue entries: {computation.inputs.shape[0]}")

    logger.info("Composite assignment counts:")
    for comp in computation.result.composites:
        logger.info(f"  Composite {comp.id}: {len(comp.assigned_entry_ids)} assigned")

    logger.info(f"Total assigned: {computation.total_assigned_count}")
    logger.info(f"Total rejected: {computation.total_rejected_count}")

    logger.info("Rejection breakdown:")
    for reason, count in computation.result.reject_reason_counts.items():
        logger.info(f"  {reason.name}: {count}")

    logger.info(f"Iterations required: {len(computation.cycles)}")
    logger.info("Reject Rates:")
    for i, cycle in enumerate(computation.cycles):
        logger.info(
            f"  Iteration {i + 1}: {cycle.reject_rate(computation.input_count):.2%}"
        )


def plot_composite_iterations(
    computation: BEQCompositeComputation,
    freqs: np.ndarray,
    band: tuple[float, float] = (5, 50),
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot the evolution of each composite across iterations.

    Each composite gets its own subplot showing how its magnitude response
    evolved through the iterative refinement process.

    Args:
        computation: BEQCompositeComputation result from build_beq_composites
        freqs: Full frequency array
        band: Frequency band to display (min_freq, max_freq)
        figsize: Figure size as (width, height). If None, auto-calculated
        save_path: Optional path to save the figure. If None, displays interactively
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.error(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
        return

    # Extract band frequencies
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[band_mask]

    # Get number of composites
    n_composites = len(computation.cycles[0].composites)

    # Calculate grid dimensions (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_composites)))
    n_rows = int(np.ceil(n_composites / n_cols))

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    # Color map for iterations
    n_iterations = len(computation.cycles)
    colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))

    # Plot each composite
    for comp_id in range(n_composites):
        row = comp_id // n_cols
        col = comp_id % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Plot each iteration for this composite
        for cycle_idx, cycle in enumerate(computation.cycles):
            composite = cycle.composites[comp_id]

            # Plot the composite curve
            ax.plot(
                band_freqs,
                composite.mag_response,
                color=colors[cycle_idx],
                linewidth=2 if cycle_idx == n_iterations - 1 else 1,
                alpha=0.8 if cycle_idx == n_iterations - 1 else 0.5,
                label=f"Iter {cycle.iteration}",
            )

        # Get the final assignment count
        final_composite = computation.cycles[-1].composites[comp_id]
        n_assigned = len(final_composite.assigned_entry_ids)

        # Formatting
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Composite {comp_id + 1} ({n_assigned})")
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_tick_params(which='both', labelsize=6)

        # Add legend only to the first subplot
        if comp_id == 0:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # Overall title
    fig.suptitle("Composite Evolution", y=0.98)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_assigned_fan_curves(
    computation: BEQCompositeComputation, freqs: np.ndarray
) -> None:
    """Plot assigned fan curves and composite shapes."""
    n_comps = len(computation.result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True, sharey=True
    )
    axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

    for comp in computation.result.composites:
        ax: Axes = axes[comp.id]

        # Fan curves
        for lvl, fan_curves in enumerate(comp.fan_envelopes):
            if fan_curves.size == 0:
                continue
            alpha = 0.2 + 0.6 * lvl / max(1, len(comp.fan_envelopes) - 1)
            for curve in fan_curves:
                ax.plot(freqs, curve, color="lightblue", lw=1, alpha=alpha, zorder=1)

        # Composite
        ax.plot(
            freqs, comp.mag_response, color="black", lw=2, label="Composite", zorder=3
        )

        ax.set_title(
            f"Composite {comp.id + 1} ({len(comp.assigned_entry_ids)} assigned)"
        )
        ax.grid(True, alpha=0.3)
        if comp.id % ncols == 0:
            ax.set_ylabel("Magnitude (dB)")
        if comp.id >= ncols * (nrows - 1):
            ax.set_xlabel("Frequency (Hz)")

        # Inset histogram for RMS of assigned curves
        inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
        assigned_rms = np.array(
            [m.rms_delta for m in comp.mappings if m.is_best and not m.rejected]
        )
        if len(assigned_rms) > 0:
            inset.hist(assigned_rms, bins=15, color="lightblue", alpha=0.7)
        inset.set_title("Assigned RMS", fontsize=8)
        inset.tick_params(axis="both", labelsize=6)

    # delete axes from all but the 1st column
    for j in range(comp.id + 1, nrows * ncols):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle("Assigned Fan Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_rejected_by_reason(
    catalogue: np.ndarray, computation: BEQCompositeComputation, freqs: np.ndarray
) -> None:
    """Plot rejected curves per rejection reason with metric-specific histograms."""
    n_comps = len(computation.result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    reasons = list(RejectionReason)
    reject_counts = computation.result.reject_reason_counts

    for reason in reasons:
        if reject_counts[reason] == 0:
            continue

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True, sharey=True
        )
        axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

        for comp in computation.result.composites:
            ax: Axes = axes[comp.id]

            rejected_entries = comp.rejected_mappings_for_reason(reason)
            if not rejected_entries:
                continue

            curves = catalogue[[i.entry_id for i in rejected_entries]]

            # Fan-style plotting
            rms_vals = np.array([i.rms_delta for i in rejected_entries])
            sort_idx = np.argsort(rms_vals)
            for j, idx_r in enumerate(sort_idx):
                alpha = 0.2 + 0.3 * j / max(1, len(sort_idx) - 1)
                ax.plot(
                    freqs,
                    curves[idx_r],
                    color="lightcoral",
                    lw=1,
                    alpha=alpha,
                    zorder=1,
                )

            # Composite overlay
            ax.plot(freqs, comp.mag_response, color="black", lw=2, zorder=2)

            # Inset histogram for metric that triggered rejection
            inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
            if reason == RejectionReason.RMS_EXCEEDED:
                inset.hist(rms_vals, bins=15, color="lightblue", alpha=0.7)
                inset.set_title("RMS", fontsize=8)
            elif reason == RejectionReason.MAX_EXCEEDED:
                vals = np.array([i.max_delta for i in rejected_entries])
                inset.hist(vals, bins=15, color="salmon", alpha=0.7)
                inset.set_title("Max", fontsize=8)
            elif reason == RejectionReason.RMS_MAX_EXCEEDED:
                max_vals = np.array([i.max_delta for i in rejected_entries])
                inset.hist(rms_vals, bins=15, color="lightblue", alpha=0.7, label="RMS")
                inset.hist(max_vals, bins=15, color="salmon", alpha=0.7, label="Max")
                inset.set_title("RMS/Max", fontsize=8)
                inset.legend(fontsize=6)
            elif reason == RejectionReason.COSINE_TOO_LOW:
                vals = np.array([1 - i.cosine_similarity for i in rejected_entries])
                inset.hist(vals, bins=15, color="violet", alpha=0.7)
                inset.set_title("1 - Cosine", fontsize=8)
            elif reason == RejectionReason.DERIVATIVE_TOO_HIGH:
                vals = np.array([i.derivative_delta for i in rejected_entries])
                inset.hist(vals, bins=15, color="orange", alpha=0.7)
                inset.set_title("Derivative", fontsize=8)

            inset.tick_params(axis="both", labelsize=6)
            ax.set_title(f"Composite {comp.id} ({len(rejected_entries)} rejected)")
            ax.grid(True, alpha=0.3)
            if comp.id % ncols == 0:
                ax.set_ylabel("Magnitude (dB)")
            if comp.id >= ncols * (nrows - 1):
                ax.set_xlabel("Frequency (Hz)")

        for j in range(comp.id + 1, nrows * ncols):
            fig.delaxes(axes[j])

        fig.suptitle(
            f"Rejected Curves by Composite — Reason: {reason.name}", fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


# ------------------------------
# RMS vs Max scatter with density
# ------------------------------
def plot_rms_max_scatter(computation: BEQCompositeComputation) -> None:
    all_rms: np.ndarray = np.array(
        [
            m.rms_delta
            for c in computation.result.composites
            for m in c.mappings
            if m.is_best
        ]
    )
    all_max: np.ndarray = np.array(
        [
            m.max_delta
            for c in computation.result.composites
            for m in c.mappings
            if m.is_best
        ]
    )

    xy: np.ndarray = np.vstack([all_rms, all_max])
    kde: np.ndarray = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 5))
    plt.scatter(all_rms, all_max, c=kde, s=20, cmap="viridis")
    plt.xlabel("RMS deviation (dB)")
    plt.ylabel("Max deviation (dB)")
    plt.title("RMS vs Max-deviation scatter with density")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_histograms(result: ComputationCycle) -> None:
    rms_vals: list[float] = [
        m.rms_delta
        for c in result.composites
        for m in c.mappings
        if m.rms_delta is not None and m.is_best
    ]
    max_vals: list[float] = [
        m.max_delta
        for c in result.composites
        for m in c.mappings
        if m.max_delta is not None and m.is_best
    ]
    cosine_vals: list[float] = [
        m.cosine_similarity
        for c in result.composites
        for m in c.mappings
        if m.cosine_similarity is not None and m.is_best
    ]
    derivative_deltas: list[float] = [
        m.derivative_delta
        for c in result.composites
        for m in c.mappings
        if m.derivative_delta is not None and m.is_best
    ]

    fig: Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    # Helper to add percentile lines and annotate them slightly offset
    def add_percentile_lines(ax, data):
        ylim = ax.get_ylim()
        for p in [50, 90, 95]:
            val = np.percentile(data, p)
            ax.axvline(val, color="lightgrey", linestyle="dotted", linewidth=1)
            # Small horizontal offset to avoid overlapping the line
            ax.text(
                val + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ylim[1] * 0.95,
                f"{p}%",
                rotation=90,
                verticalalignment="top",
                color="grey",
                fontsize=8,
            )

    # RMS histogram
    i = 0
    axes[i].hist(rms_vals, bins=100, color="skyblue", edgecolor="black")
    axes[i].set_xlabel("RMS deviation (dB)")
    axes[i].set_ylabel("Count")
    axes[i].set_title("RMS deviation")
    add_percentile_lines(axes[i], rms_vals)

    # Max deviation histogram
    i = i + 1
    axes[i].hist(max_vals, bins=100, color="salmon", edgecolor="black")
    axes[i].set_xlabel("Max deviation (dB)")
    axes[i].set_ylabel("Count")
    axes[i].set_title("Max Deviation")
    add_percentile_lines(axes[i], max_vals)

    # Cosine similarity histogram
    i = i + 1
    axes[i].hist(cosine_vals, bins=100, color="palegreen", edgecolor="black")
    axes[i].set_xlabel("Cosine")
    axes[i].set_ylabel("Count")
    axes[i].set_title("Cosine Similarity")
    axes[i].set_xlim(0.5, 1)
    add_percentile_lines(axes[i], cosine_vals)

    # Cosine similarity histogram
    i = i + 1
    axes[i].hist(cosine_vals, bins=100, color="darkviolet", edgecolor="black")
    axes[i].set_xlabel("Derivative Delta")
    axes[i].set_ylabel("Count")
    axes[i].set_title("Derivative Delta")
    axes[i].set_xlim(0.5, 1)
    add_percentile_lines(axes[i], derivative_deltas)

    plt.tight_layout()
    plt.show()


def print_assignments(
    computation: BEQCompositeComputation, filters: list[BEQFilter]
) -> None:
    with open("beq_composites.csv", "w", newline="") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(
            [
                "composite_id",
                "reject reason",
                "content type",
                "author",
                "title",
                "year",
                "cosine",
                "rms",
                "max",
                "derivative",
                "digest",
                "beqc url",
            ]
        )
        for c in computation.result.composites:
            for m in c.mappings:
                if m.is_best:
                    underlying: BEQFilter = filters[m.entry_id]
                    writer.writerow(
                        [
                            m.composite_id,
                            m.rejection_reason.name if m.rejection_reason else "",
                            underlying.entry.content_type,
                            underlying.entry.author,
                            underlying.entry.formatted_title,
                            underlying.entry.year,
                            m.cosine_similarity,
                            m.rms_delta,
                            m.max_delta,
                            m.derivative_delta,
                            underlying.entry.digest,
                            underlying.entry.beqc_url,
                        ]
                    )

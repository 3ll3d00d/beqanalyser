import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

from beqanalyser import (
    BEQComposite,
    BEQCompositeComputation,
    BEQFilter,
    BEQResult,
    BiquadCoefficients,
)
from beqanalyser.filter import CompositeCurveFitter

logger = logging.getLogger()


def summarise_result(result: BEQResult) -> None:
    logger.info("---------------")
    logger.info("Final Result")
    logger.info("---------------")
    logger.info(f"Catalogue entries: {result.input_size}")
    logger.info(
        f"Assigned: {result.total_assigned_count} ({result.total_assigned_count / result.input_size:.1%})"
    )
    logger.info(
        f"Rejected: {result.total_rejected_count} ({result.total_rejected_count / result.input_size:.1%})"
    )

    logger.info("Composite assignment counts:")
    for comp in result.composites:
        assigned_count = len(comp.assigned_entry_ids)
        logger.info(
            f"  Composite {comp.id}: {assigned_count} assigned ({assigned_count / result.input_size:.1%})"
        )


def summarise_assignments(
    iteration: int, computation: BEQCompositeComputation, level: int = logging.INFO
) -> None:
    logger.log(level, "---------------")
    logger.log(level, f"Iteration {iteration}")
    logger.log(level, "---------------")
    logger.log(level, f"Total catalogue entries: {computation.inputs.shape[0]}")

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


def plot_composite_evolution(
    result: BEQResult,
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
        result: result from build_beq_composites
        freqs: Full frequency array
        band: Frequency band to display (min_freq, max_freq)
        figsize: Figure size as (width, height). If None, auto-calculated
        save_path: Optional path to save the figure. If None, displays interactively
    """
    # Extract band frequencies
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[band_mask]

    # Get total number of composites & max no of iterations
    n_composites = len(result.composites)
    n_iterations = max(
        len([y for y in c.cycles if not y.is_copy]) for c in result.calculations
    )

    # Calculate grid dimensions (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_composites)))
    n_rows = int(np.ceil(n_composites / n_cols))

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    # Create figure and subplots
    fig, all_ax = plt.subplots(
        n_rows, n_cols, sharex=True, sharey=True, figsize=figsize
    )
    axes = np.array(all_ax).flatten() if n_cols * n_rows > 1 else np.array([all_ax])

    # Colour map for iterations
    colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))

    # Plot each composite
    pos = -1
    calc = -1

    for computation in result.calculations:
        calc += 1
        for j in range(len(computation.result.composites)):
            pos += 1
            row = pos // n_cols
            col = pos % n_cols
            logger.info(
                f"Plotting composite {calc}/{j} in pos {pos} at coordinates {row, col}..."
            )
            ax = axes[pos]
            unique_iteration_count = len(
                [c for c in computation.cycles if c.is_copy is False]
            )
            for cycle_idx, cycle in enumerate(computation.cycles):
                if cycle.is_copy:
                    continue
                ax.plot(
                    band_freqs,
                    cycle.composites[j].mag_response,
                    color=colors[cycle_idx],
                    linewidth=2 if cycle_idx == unique_iteration_count - 1 else 1,
                    alpha=0.8 if cycle_idx == unique_iteration_count - 1 else 0.5,
                    label=f"Iter {cycle.iteration}",
                )

            # Formatting
            if row == n_rows - 1:
                ax.set_xlabel("Frequency (Hz)")
            if col == 0:
                ax.set_ylabel("Magnitude (dB)")
            ax.set_title(
                f"Composite {pos + 1} ({len(computation.cycles[-1].composites[j].assigned_entry_ids)})"
            )
            ax.grid(True, which="both", alpha=0.3)
            # ax.set_xscale("log")
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
            ax.xaxis.set_tick_params(which="both", labelsize=6)

            # Add legend only to the first subplot
            if pos == 0:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # Overall title
    fig.suptitle("Composite Evolution", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_distance_by_composite(
    composites: list[BEQComposite], freqs: np.ndarray
) -> None:
    """Plot distance  and composite shapes."""
    pass


def plot_assigned_fan_curves(composites: list[BEQComposite], freqs: np.ndarray) -> None:
    """Plot assigned fan curves and composite shapes."""
    n_comps = len(composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True, sharey=True
    )
    axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

    for comp in composites:
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

        ax.grid(True, which="both", alpha=0.3)
        # ax.set_xscale("log")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_tick_params(which="both", labelsize=6)
        # ax.set_ylim(bottom=0)

        # Inset histogram for distance of assigned curves
        distance_scores = np.array(
            [m.distance_score for m in comp.mappings if m.is_best and not m.rejected]
        )
        if len(distance_scores) > 0:
            inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
            inset.hist(distance_scores, bins=50, color="lightblue", alpha=0.7)
            inset.tick_params(axis="both", labelsize=6)

    # delete axes from all but the 1st column
    for j in range(comp.id + 1, nrows * ncols):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle("Assigned Fan Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_distance_histograms(composites: list[BEQComposite]) -> None:
    rms_vals: list[float] = [
        m.rms_delta
        for c in composites
        for m in c.mappings
        if m.rms_delta is not None and m.is_best
    ]
    max_vals: list[float] = [
        m.max_delta
        for c in composites
        for m in c.mappings
        if m.max_delta is not None and m.is_best
    ]
    cosine_vals: list[float] = [
        m.cosine_similarity
        for c in composites
        for m in c.mappings
        if m.cosine_similarity is not None and m.is_best
    ]
    derivative_deltas: list[float] = [
        m.derivative_delta
        for c in composites
        for m in c.mappings
        if m.derivative_delta is not None and m.is_best
    ]
    distance_vals: list[float] = [
        m.distance_score for c in composites for m in c.mappings if m.is_best
    ]

    fig: Figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5))

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
    ax1.hist(rms_vals, bins=100, color="skyblue", edgecolor="black")
    ax1.set_xlabel("RMS deviation (dB)")
    ax1.set_ylabel("Count")
    add_percentile_lines(ax1, rms_vals)

    # Max deviation histogram
    ax2.hist(max_vals, bins=100, color="salmon", edgecolor="black")
    ax2.set_xlabel("Max deviation (dB)")
    ax2.set_ylabel("Count")
    add_percentile_lines(ax2, max_vals)

    # Cosine similarity histogram
    ax3.hist(cosine_vals, bins=100, color="palegreen", edgecolor="black")
    ax3.set_xlabel("Cosine")
    ax3.set_ylabel("Count")
    ax3.set_xlim(0.5, 1)
    add_percentile_lines(ax3, cosine_vals)

    # Cosine similarity histogram
    ax4.hist(cosine_vals, bins=100, color="darkviolet", edgecolor="black")
    ax4.set_xlabel("Derivative Delta")
    ax4.set_ylabel("Count")
    ax4.set_xlim(0.5, 1)
    add_percentile_lines(ax4, derivative_deltas)

    # Distance
    ax5.hist(distance_vals, bins=100, color="yellow", edgecolor="black")
    ax5.set_xlabel("Distance Score")
    ax5.set_ylabel("Count")
    add_percentile_lines(ax5, distance_vals)

    fig.delaxes(ax6)

    plt.tight_layout()
    plt.show()


def print_assignments(composites: list[BEQComposite], filters: list[BEQFilter]) -> None:
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
        for c in composites:
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


def plot_filter_comparison(
    composite_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    fitted_filters: dict[str, list[BiquadCoefficients]],
    fs: float = 48000,
    freq_range: tuple[float, float] = (5, 50),
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """
    Plot composite curves and their fitted filters on a grid of subplots.

    Args:
        composite_curves: Dict mapping curve names to (freq, dB) tuples
        fitted_filters: Dict mapping curve names to lists of BiquadCoefficients
        fs: Sample rate in Hz
        freq_range: Frequency range for plotting (min, max) in Hz
        figsize: Figure size (width, height). Auto-calculated if None
        save_path: Path to save the figure. If None, displays instead
    """
    n_curves = len(composite_curves)

    if n_curves == 0:
        logging.warning("No curves to plot")
        return

    # Calculate grid dimensions
    n_cols = min(3, n_curves)  # Max 3 columns
    n_rows = (n_curves + n_cols - 1) // n_cols

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    fig = plt.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    # Generate frequency grid for fitted responses
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 500)

    # Create fitter for computing responses
    fitter = CompositeCurveFitter(fs=fs, freq_range=freq_range)

    for idx, (name, (target_freqs, target_db)) in enumerate(composite_curves.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Plot target curve
        ax.semilogx(
            target_freqs,
            target_db,
            "o-",
            linewidth=2,
            markersize=4,
            alpha=0.7,
            label="Target Composite",
            color="#2E86AB",
        )

        # Plot fitted filter response if available
        if name in fitted_filters and fitted_filters[name]:
            biquads = fitted_filters[name]
            fitted_response = fitter.compute_filter_response(biquads)
            ax.semilogx(
                fitter.freqs,
                fitted_response,
                "-",
                linewidth=2,
                label=f"Fitted ({len(biquads)} filters)",
                color="#A23B72",
            )

            # Calculate and display RMS error
            target_interp = np.interp(fitter.freqs, target_freqs, target_db)
            rms_error = np.sqrt(np.mean((fitted_response - target_interp) ** 2))

            # Plot individual filter contributions (optional, lighter lines)
            if len(biquads) <= 5:  # Only show for simple cases
                cumulative = np.zeros(len(fitter.freqs))
                for i, bq in enumerate(biquads):
                    response = fitter.compute_filter_response([bq])
                    cumulative += response
                    ax.semilogx(
                        fitter.freqs,
                        response,
                        "--",
                        linewidth=1,
                        alpha=0.3,
                        label=f"{bq.filter_type} @ {bq.fc:.0f}Hz",
                    )
        else:
            rms_error = None

        # Formatting
        ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Magnitude (dB)", fontsize=10)
        ax.set_xlim(freq_range)

        # Set y-axis limits with some padding
        all_db = list(target_db)
        if name in fitted_filters and fitted_filters[name]:
            all_db.extend(fitted_response)
        y_min, y_max = min(all_db), max(all_db)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Title with RMS error if available
        title = name.replace("_", " ").title()
        if rms_error is not None:
            title += f"\n(RMS Error: {rms_error:.2f} dB)"
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Legend
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # Remove empty subplots
    for idx in range(n_curves, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(fig.add_subplot(gs[row, col]))

    plt.suptitle(
        "BEQ Composite Curves vs Fitted Biquad Filters",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_single_filter_detail(
    name: str,
    target_freqs: np.ndarray,
    target_db: np.ndarray,
    biquads: list[BiquadCoefficients],
    fs: float = 48000,
    freq_range: tuple[float, float] = (10, 200),
    save_path: str | None = None,
):
    """
    Create a detailed plot for a single filter showing individual contributions.

    Args:
        name: Name of the curve
        target_freqs: Target frequency points
        target_db: Target magnitude in dB
        biquads: List of fitted biquad coefficients
        fs: Sample rate
        freq_range: Frequency range for plotting
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    fitter = CompositeCurveFitter(fs=fs, freq_range=freq_range)

    # Top plot: Overall comparison
    ax1.semilogx(
        target_freqs,
        target_db,
        "o-",
        linewidth=2,
        markersize=6,
        label="Target Composite",
        color="#2E86AB",
    )

    if biquads:
        fitted_response = fitter.compute_filter_response(biquads)
        ax1.semilogx(
            fitter.freqs,
            fitted_response,
            "-",
            linewidth=2.5,
            label="Fitted Response",
            color="#A23B72",
        )

        # Calculate error
        target_interp = np.interp(fitter.freqs, target_freqs, target_db)
        error = fitted_response - target_interp
        rms_error = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))

        ax1.text(
            0.02,
            0.98,
            f"RMS Error: {rms_error:.3f} dB\nMax Error: {max_error:.3f} dB",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(f"{name} - Overall Fit", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(freq_range)

    # Bottom plot: Individual filter contributions
    if biquads:
        colors = plt.cm.tab10(np.linspace(0, 1, len(biquads)))

        for i, (bq, color) in enumerate(zip(biquads, colors)):
            response = fitter.compute_filter_response([bq])
            label = f"{i + 1}. {bq.filter_type}: {bq.fc:.1f}Hz, {bq.gain:+.1f}dB, Q={bq.q:.2f}"
            ax2.semilogx(
                fitter.freqs, response, "-", linewidth=2, label=label, color=color
            )

        # Also plot cumulative sum
        cumulative = np.zeros(len(fitter.freqs))
        for bq in biquads:
            cumulative += fitter.compute_filter_response([bq])
        ax2.semilogx(
            fitter.freqs,
            cumulative,
            "k--",
            linewidth=2,
            alpha=0.5,
            label="Sum of Individual",
        )

    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Magnitude (dB)", fontsize=11)
    ax2.set_title("Individual Filter Contributions", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="best", ncol=1)
    ax2.set_xlim(freq_range)
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Detailed plot saved to {save_path}")
    else:
        plt.show()

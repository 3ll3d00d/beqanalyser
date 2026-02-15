import logging
import math

import matplotlib as mpl
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
)
from beqanalyser.filter import BiquadFilter

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
    fit_results: dict,
):
    """
    Plot composite curves and their fitted filters on a grid of subplots.
    """
    n_curves = len(fit_results)

    if n_curves == 0:
        logging.warning("No curves to plot")
        return

    # Calculate grid dimensions
    n_cols = min(3, n_curves)  # Max 3 columns
    n_rows = (n_curves + n_cols - 1) // n_cols

    # Auto-calculate figure size if not provided
    figsize = (6 * n_cols, 4 * n_rows)

    fig = plt.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    idx = -1
    for id, result in fit_results.items():
        freq_range = result["freqs"]

        # Generate frequency grid for fitted responses
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[-1]), 500)

        idx += 1
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Plot target curve
        ax.plot(
            freqs,
            np.interp(freqs, freq_range, result["target_response"]),
            "o-",
            linewidth=1,
            markersize=4,
            alpha=0.7,
            label="Target Composite",
            color="#2E86AB",
        )

        # Plot fitted filter response
        ax.plot(
            freqs,
            np.interp(freqs, freq_range, result["fitted_response"]),
            "-",
            linewidth=1,
            label=f"{len(result['filters'])} fitted",
            color="#A23B72",
        )

        # Formatting
        ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Magnitude (dB)", fontsize=10)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
        # ax.set_xlim(freq_range)
        ax.set_title(
            f"Composite {id + 1}\n(RMS Error: {result['rms_error']:.2f} dB)",
            fontsize=11,
            fontweight="bold",
        )
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

    plt.show()


def show_filters(filter_sets: dict[str, list[BiquadFilter]]):
    n = len(filter_sets)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(6 * cols, 3 * rows),
        constrained_layout=True
    )

    axes = axes.flatten() if n > 1 else [axes]

    # Infer schema
    first_filter = next(iter(filter_sets.values()))[0]
    labels = first_filter.column_labels()
    ncols = len(labels)

    # Uniform column widths
    col_widths = [1.0 / ncols] * ncols

    # 🔥 Compute max row count (including header)
    max_filters = max(len(v) for v in filter_sets.values())
    total_rows = max_filters + 1  # +1 header
    row_height = 0.9 / total_rows  # 0.9 = bbox height

    dark = True # fix
    cell_bg = "#222222" if dark else "white"
    header_bg = "#333333" if dark else "#eeeeee"
    text_color = "white" if dark else "black"
    edge_color = "#666666" if dark else "#444444"

    for ax, (title, filters) in zip(axes, filter_sets.items()):
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=6)

        # Pad with blank rows so all tables have identical row count
        padded_filters = list(filters)
        while len(padded_filters) < max_filters:
            padded_filters.append(None)

        rows_data = []
        for obj in padded_filters:
            if obj is None:
                rows_data.append([""] * ncols)
            else:
                rows_data.append(obj.as_row())

        table = ax.table(
            cellText=rows_data,
            colLabels=labels,
            colWidths=col_widths,
            bbox=[0, 0, 1, 0.9]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # 🔥 Force identical row height for every cell
        for (r, c), cell in table.get_celld().items():
            cell.set_height(row_height)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.6)
            cell.get_text().set_color(text_color)

            if r == 0:
                cell.set_facecolor(header_bg)
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(cell_bg)

    for ax in axes[len(filter_sets):]:
        ax.axis("off")

    plt.show()
import logging
import sys
from collections import defaultdict

import numpy as np

from beqanalyser import BEQCompositeComputation, BEQFilter
from beqanalyser.analyser import build_beq_composites, merge_calculations
from beqanalyser.loader import load
from beqanalyser.reporter import (
    plot_assigned_fan_curves,
    plot_composite_evolution,
    plot_histograms,
    plot_rejected_by_reason,
    plot_rms_max_scatter,
    print_assignments,
    summarize_assignments,
)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - [%(threadName)s] - %(message)s',
    )

    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)
    rms_limit = 10.0
    max_limit = 10.0
    cosine_limit = 0.9
    derivative_limit = 0.9
    use_constraints = True

    catalogue: list[BEQFilter] = load()
    by_author_by_year = defaultdict(lambda: defaultdict(int))

    freqs = catalogue[0].mag_freqs
    responses_db = np.array([f.mag_db - f.mag_db[-1] for f in catalogue])
    weights = np.ones_like(freqs)

    assigned_rate = 1.0
    params = [
        (100, 10, 3.0),
        (50, 5, 3.0),
        (30, 3, 3.0),
        (10, 2, 3.0),
        (5, 2, 3.0),
        (3, 2, 3.0),
    ]
    calcs: list[BEQCompositeComputation] = []
    input_size = responses_db.shape[0]
    param_idx = 0
    while assigned_rate >= 0.01 and param_idx < len(params):
        logging.info(
            f"Running iteration {param_idx + 1} with params {params[param_idx]}"
        )
        last_calc = build_beq_composites(
            responses_db=responses_db
            if not calcs
            else responses_db[calcs[-1].result.rejected_entry_ids],
            freqs=freqs,
            weights=weights,
            band=(min_freq, max_freq),
            rms_limit=rms_limit,
            max_limit=max_limit,
            cosine_limit=cosine_limit,
            derivative_limit=derivative_limit,
            fan_counts=fan_counts,
            hdbscan_min_cluster_size=params[param_idx][0],
            hdbscan_min_samples=params[param_idx][1],
            hdbscan_cluster_selection_epsilon=params[param_idx][2],
            hdbscan_use_constraints=use_constraints,
            hdbscan_distance_chunk_size=input_size,
        )
        summarize_assignments(last_calc)
        calcs.append(last_calc)
        assigned_count = sum([a.total_assigned_count for a in calcs])
        composite_count = sum([len(a.result.composites) for a in calcs])
        assigned_rate = assigned_count / input_size
        logging.info(
            f"After iteration {param_idx + 1} total assignment rate: {assigned_rate:.2%}, composites: {composite_count}"
        )
        param_idx += 1

    final_calc = merge_calculations(calcs)
    plot_composite_evolution(final_calc, freqs, band=(min_freq, max_freq))
    plot_histograms(final_calc.result)
    plot_rms_max_scatter(final_calc)
    plot_composite_evolution(final_calc, freqs, band=(min_freq, max_freq))
    plot_assigned_fan_curves(
        final_calc, freqs[(freqs >= min_freq) & (freqs <= max_freq)]
    )
    print_assignments(final_calc, catalogue)
    plot_rejected_by_reason(
        responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
        final_calc,
        freqs[(freqs >= min_freq) & (freqs <= max_freq)],
    )

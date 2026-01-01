import datetime
import logging
import sys
from collections import defaultdict

import numpy as np

from beqanalyser import BEQCompositeComputation, BEQFilter
from beqanalyser.analyser import build_beq_composites
from beqanalyser.loader import load
from beqanalyser.reporter import (
    plot_assigned_fan_curves,
    plot_composite_iterations,
    plot_histograms,
    plot_rejected_by_reason,
    plot_rms_max_scatter,
    print_assignments,
    summarize_assignments,
)

if __name__ == "__main__":
    rootLogger = logging.getLogger()
    rootLogger.addHandler(logging.StreamHandler(sys.stdout))
    rootLogger.setLevel(logging.INFO)

    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)
    rms_limit = 10.0
    max_limit = 10.0
    cosine_limit = 0.9
    derivative_limit = 0.9

    catalogue: list[BEQFilter] = load()
    by_author_by_year = defaultdict(lambda: defaultdict(int))

    freqs = catalogue[0].mag_freqs
    responses_db = np.array([f.mag_db - f.mag_db[-1] for f in catalogue])
    weights = np.ones_like(freqs)

    all_calc = build_beq_composites(
        responses_db=responses_db,
        freqs=freqs,
        weights=weights,
        band=(min_freq, max_freq),
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit,
        fan_counts=fan_counts,
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=3,
        hdbscan_cluster_selection_epsilon=2.0,
    )

    reject_only_calc = build_beq_composites(
        responses_db=responses_db[all_calc.result.rejected_entry_ids],
        freqs=freqs,
        weights=weights,
        band=(min_freq, max_freq),
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit,
        fan_counts=fan_counts,
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=5,
        hdbscan_cluster_selection_epsilon=3.0,
    )

    def dump_diagnostics(calculation: BEQCompositeComputation, is_all: bool = False):
        summarize_assignments(calculation)
        plot_composite_iterations(calculation, freqs, band=(min_freq, max_freq))
        if is_all:
            plot_histograms(calculation.result)
        plot_assigned_fan_curves(
            calculation, freqs[(freqs >= min_freq) & (freqs <= max_freq)]
        )
        if is_all:
            plot_rms_max_scatter(calculation)
            print_assignments(calculation, catalogue)
        else:
            plot_rejected_by_reason(
                responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
                calculation,
                freqs[(freqs >= min_freq) & (freqs <= max_freq)],
            )

    dump_diagnostics(all_calc, is_all=True)
    dump_diagnostics(reject_only_calc, is_all=False)

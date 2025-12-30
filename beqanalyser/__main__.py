import logging
import sys
from http.client import responses
from itertools import groupby

import numpy as np

from beqanalyser import BEQFilter, BEQCompositeComputation
from beqanalyser.analyser import build_beq_composites
from beqanalyser.loader import load
from beqanalyser.reporter import (
    plot_all_beq_curves,
    plot_histograms,
    plot_rms_max_scatter,
    print_assignments,
    summarize_assignments,
)

if __name__ == '__main__':
    rootLogger = logging.getLogger()
    rootLogger.addHandler(logging.StreamHandler(sys.stdout))
    rootLogger.setLevel(logging.INFO)

    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)
    rms_limit = 10.0
    max_limit = 10.0
    cosine_limit = 0.975
    derivative_limit = 1.0

    catalogue: list[BEQFilter] = load()

    by_author = groupby(catalogue, lambda x: x.entry.author)
    for author, filters_iter in by_author:
        filters = list(filters_iter)
        freqs = filters[0].mag_freqs
        if author != 'aron7awol':
            continue
        responses_db = np.array([f.mag_db - f.mag_db[-1] for f in filters])
        weights = np.ones_like(freqs)

        all_calc = build_beq_composites(
            responses_db=responses_db,
            freqs=freqs,
            weights=weights,
            band=(min_freq, max_freq),
            k=6,
            rms_limit=rms_limit,
            max_limit=max_limit,
            cosine_limit=cosine_limit,
            derivative_limit=derivative_limit,
            fan_counts=fan_counts,
            n_prototypes=1024
        )

        reject_only_calc = build_beq_composites(
            responses_db=responses_db[all_calc.result.rejected_entry_ids],
            freqs=freqs,
            weights=weights,
            band=(min_freq, max_freq),
            k=6,
            rms_limit=rms_limit,
            max_limit=max_limit,
            cosine_limit=cosine_limit,
            derivative_limit=derivative_limit,
            fan_counts=fan_counts,
            n_prototypes=1024
        )

        def dump_diagnostics(calculation: BEQCompositeComputation, do_all: bool = False):
            summarize_assignments(calculation)
            if do_all:
                plot_histograms(calculation.result)
            plot_all_beq_curves(responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
                                calculation, freqs[(freqs >= min_freq) & (freqs <= max_freq)])
            if do_all:
                plot_rms_max_scatter(calculation)
                print_assignments(calculation, filters)

        dump_diagnostics(all_calc, do_all=True)
        dump_diagnostics(reject_only_calc, do_all=False)

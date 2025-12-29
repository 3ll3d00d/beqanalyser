import logging
import sys

import numpy as np

from itertools import groupby

from beqanalyser import BEQFilter
from beqanalyser.analyser import build_beq_composites
from beqanalyser.loader import load
from beqanalyser.reporter import summarize_assignments, plot_histograms_from_table, plot_all_beq_curves, \
    plot_rms_max_scatter, print_assignments

if __name__ == '__main__':
    rootLogger = logging.getLogger()
    rootLogger.addHandler(logging.StreamHandler(sys.stdout))
    rootLogger.setLevel(logging.INFO)

    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)

    catalogue: list[BEQFilter] = load()

    by_author = groupby(catalogue, lambda x: x.entry.author)
    for author, filters_iter in by_author:
        filters = list(filters_iter)
        freqs = filters[0].mag_freqs
        if author != 'aron7awol':
            continue
        responses_db = np.array([f.mag_db - f.mag_db[-1] for f in filters])
        weights = np.ones_like(freqs)

        for i in range(7, 8, 1):
            result = build_beq_composites(
                responses_db=responses_db,
                freqs=freqs,
                weights=weights,
                band=(min_freq, max_freq),
                k=i,
                rms_limit=10.0,
                max_limit=10.0,
                cosine_limit=0.95,
                derivative_limit=2.0,
                fan_counts=fan_counts,
                n_prototypes=100
            )

            # Diagnostics
            summarize_assignments(result)
            plot_histograms_from_table(list(result.assignment_table.values()))
            plot_all_beq_curves(responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
                                result, freqs[(freqs >= min_freq) & (freqs <= max_freq)])
            plot_rms_max_scatter(result)
            print_assignments(result, filters)

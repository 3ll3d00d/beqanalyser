import logging
import sys
from collections import defaultdict

import numpy as np

from beqanalyser.analyser import (
    DistanceParams,
    HDBSCANParams,
    build_all_composites,
    compute_distance_matrix,
)
from beqanalyser.loader import load
from beqanalyser.reporter import (
    plot_assigned_fan_curves,
    plot_composite_evolution,
    plot_histograms,
    plot_rms_max_scatter,
    print_assignments,
    summarise_assignments,
    summarise_result,
)


def load_or_compute_distance_matrix(
    input_curves: np.ndarray,
    freqs: np.ndarray,
    distance_params: DistanceParams,
    data_hash: str,
    band: tuple[float, float] = (5, 50),
) -> np.ndarray:
    target_file = f"{data_hash}.npy"
    try:
        with open(target_file, "rb") as f:
            return np.load(f)
    except FileNotFoundError:
        logging.info("Distance matrix not found in , computing...")
        matrix = compute_distance_matrix(
            input_curves=input_curves,
            freqs=freqs,
            band=band,
            distance_params=distance_params,
        )
        np.save(target_file, matrix)
        return matrix


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - [%(threadName)s] - %(message)s",
    )

    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)
    use_constraints = True

    catalogue, data_hash = load()
    by_author_by_year = defaultdict(lambda: defaultdict(int))

    freqs = catalogue[0].mag_freqs
    responses_db = np.array([f.mag_db - f.mag_db[-1] for f in catalogue])

    params = [
        HDBSCANParams(
            min_cluster_size=30, min_samples=20, cluster_selection_epsilon=10.0
        ),
        HDBSCANParams(
            min_cluster_size=30, min_samples=20, cluster_selection_epsilon=10.0
        ),
        HDBSCANParams(
            min_cluster_size=30, min_samples=20, cluster_selection_epsilon=10.0
        ),
        HDBSCANParams(
            min_cluster_size=30, min_samples=10, cluster_selection_epsilon=10.0
        ),
        HDBSCANParams(
            min_cluster_size=10, min_samples=5, cluster_selection_epsilon=10.0
        ),
        HDBSCANParams(
            min_cluster_size=10, min_samples=5, cluster_selection_epsilon=10.0
        ),
    ]

    distance_params = DistanceParams()

    full_distance_matrix = load_or_compute_distance_matrix(
        input_curves=responses_db,
        freqs=freqs,
        band=(min_freq, max_freq),
        distance_params=distance_params,
        data_hash=data_hash,
    )

    result = build_all_composites(
        input_curves=responses_db,
        freqs=freqs,
        band=(min_freq, max_freq),
        fan_counts=fan_counts,
        iteration_params=params,
        distance_params=distance_params,
        full_distance_matrix=full_distance_matrix,
        final_assignment_threshold_multiplier=-1
    )

    plot_histograms(result.composites)
    plot_rms_max_scatter(result.composites)
    plot_assigned_fan_curves(
        result.composites, freqs[(freqs >= min_freq) & (freqs <= max_freq)]
    )
    print_assignments(result.composites, catalogue)
    # plot_rejected_by_reason(
    #     responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
    #     result[-1],
    #     freqs[(freqs >= min_freq) & (freqs <= max_freq)],
    # )
    summarise_result(result)
    for i, c in enumerate(result.calculations, start=1):
        summarise_assignments(i, c, level=logging.INFO)
    plot_composite_evolution(result, freqs, band=(min_freq, max_freq))

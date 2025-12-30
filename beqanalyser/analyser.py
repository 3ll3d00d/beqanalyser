import copy
import math
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from scipy.cluster.hierarchy import fcluster, linkage

from beqanalyser import (
    BEQComposite,
    BEQCompositeComputation,
    BEQFilterMapping,
    ComputationCycle,
    RejectionReason,
    cosine_similarity,
    derivative_rms,
    rms,
)


# ------------------------------
# Pure NumPy K-medoids
# ------------------------------
def k_medoids(
    X: np.ndarray, n_clusters: int, max_iter: int = 100, random_state: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed=random_state)
    n_samples = X.shape[0]
    medoid_indices = rng.choice(n_samples, size=n_clusters, replace=False)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - X[medoid_indices][None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_medoids = np.copy(medoid_indices)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            total_dists = np.sum(
                np.linalg.norm(
                    cluster_points[:, None, :] - cluster_points[None, :, :], axis=2
                ),
                axis=1,
            )
            min_idx = np.argmin(total_dists)
            new_medoids[i] = np.where((X == cluster_points[min_idx]).all(axis=1))[0][0]

        if np.all(new_medoids == medoid_indices):
            break
        medoid_indices = new_medoids

    return medoid_indices


# ------------------------------
# Assignment function
# ------------------------------
def map_to_best_composite(
    entry: np.ndarray,
    composites: list[BEQComposite],
    rms_limit: float,
    max_limit: float,
    cosine_limit: float,
    derivative_limit: float,
    entry_idx: int,
    weights: np.ndarray | None = None,
    rms_epsilon: float = 1.0,
) -> None:
    """
    Assign entry to composites using two-stage selection:
    1. Find composites within rms_epsilon of minimum RMS
    2. Among those, select the one with highest cosine similarity

    Args:
        entry: Input curve to assign
        composites: List of composite candidates
        rms_limit: Maximum acceptable RMS deviation
        max_limit: Maximum acceptable absolute deviation
        cosine_limit: Minimum acceptable cosine similarity
        derivative_limit: Maximum acceptable derivative RMS
        entry_idx: Index of entry in catalogue
        weights: Optional frequency weights for RMS calculation
        rms_epsilon: Tolerance for selecting near-minimum RMS composites

    """
    # Store all composite evaluations
    mappings: list[BEQFilterMapping] = []

    for comp in composites:
        delta = entry - comp.mag_response
        mapping = BEQFilterMapping(
            composite_id=comp.id,
            entry_id=entry_idx,
            rms_delta=rms(delta, weights),
            max_delta=float(np.max(np.abs(delta))),
            derivative_delta=derivative_rms(delta),
            cosine_similarity=cosine_similarity(entry, comp.mag_response),
        )
        mapping.assess(rms_limit, max_limit, cosine_limit, derivative_limit)
        mappings.append(mapping)
        comp.mappings.append(mapping)

    # Find minimum RMS
    min_rms = min(m.rms_delta for m in mappings)

    # Filter composites within epsilon of minimum RMS
    candidates = [m for m in mappings if m.rms_delta <= min_rms + rms_epsilon]

    # Among candidates, select the candidate with the highest cosine similarity
    best_metrics = max(candidates, key=lambda m: m.cosine_similarity)

    # flag all other composites as suboptimal unless already rejected
    for mapping in mappings:
        if mapping.composite_id != best_metrics.composite_id:
            if not mapping.rejected:
                mapping.rejection_reason = RejectionReason.SUBOPTIMAL
        else:
            mapping.is_best = True


# ------------------------------
# computes the composites for the next cycle based on the current cycle results
# ------------------------------
def compute_next_cycle(
    catalogue: np.ndarray, cycle: ComputationCycle, copy_forward: bool = False
) -> ComputationCycle:
    def _compute(c: BEQComposite):
        assigned = catalogue[c.assigned_entry_ids, :]
        return np.median(assigned, axis=0)

    next_composites = [BEQComposite(c.id, _compute(c)) for c in cycle.composites]
    next_cycle = ComputationCycle(cycle.iteration + 1, next_composites)
    if copy_forward:
        for comp in next_composites:
            comp.mappings = [
                copy.deepcopy(m) for m in cycle.composites[comp.id].mappings
            ]
    return next_cycle


# ------------------------------
# Compute non-overlapping fan curves
# ------------------------------
def compute_fan_curves(
    catalogue: np.ndarray,
    composites: list[BEQComposite],
    fan_counts: tuple[int, ...] = (5,),
) -> None:
    """
    For each composite, compute non-overlapping fan levels of assigned curves.
    Each fan level contains curves not in previous levels.
    """
    for comp in composites:
        comp.fan_envelopes = []
        if comp.mappings:
            assigned_filters = catalogue[comp.assigned_entry_ids, :]
            # Rank by RMS distance to the composite
            rms_dists = np.array([rms(c - comp.mag_response) for c in assigned_filters])
            sorted_idx = np.argsort(rms_dists)

            previous = 0
            for n in fan_counts:
                n_curves = min(n, len(sorted_idx))
                if n_curves > previous:
                    comp.fan_envelopes.append(
                        assigned_filters[sorted_idx[previous:n_curves], :]
                    )
                    previous = n_curves
                else:
                    # If not enough new curves, append empty array
                    comp.fan_envelopes.append(np.empty((0, assigned_filters.shape[1])))
        else:
            for _ in fan_counts:
                comp.fan_envelopes.append(np.array([comp.mag_response]))


# ------------------------------
# Full pipeline
# ------------------------------
def build_beq_composites(
    responses_db: np.ndarray,
    freqs: np.ndarray,
    weights: np.ndarray | None = None,
    band: tuple[float, float] = (5, 50),
    k: int = 5,
    rms_limit: float = 5.0,
    max_limit: float = 5.0,
    cosine_limit: float = 0.95,
    derivative_limit: float = 2.0,
    fan_counts: tuple[int, ...] = (5,),
    n_prototypes: int = 50,
    rms_epsilon: float = 0.5,
    max_iterations: int = 10,
    min_reject_rate_delta: float = 0.005,
) -> BEQCompositeComputation:
    band_mask: np.ndarray = (freqs >= band[0]) & (freqs <= band[1])
    catalogue: np.ndarray = responses_db[:, band_mask]
    band_weights: np.ndarray | None = (
        weights[band_mask] if weights is not None else None
    )

    # Step 1: k-medoids prototypes
    if catalogue.shape[0] <= n_prototypes:
        prototypes: np.ndarray = catalogue.copy()
    else:
        medoid_indices: np.ndarray = k_medoids(
            catalogue, n_prototypes, max_iter=100, random_state=0
        )
        prototypes = catalogue[medoid_indices]

    # Step 2: Ward clustering
    linkage_matrix: np.ndarray = linkage(prototypes, method="ward")
    labels: np.ndarray = fcluster(linkage_matrix, t=k, criterion="maxclust")

    # Step 3: median per cluster → initial estimate of composite curve
    composites: list[BEQComposite] = [
        BEQComposite(i - 1, np.median(prototypes[labels == i], axis=0))
        for i in range(1, k + 1)
    ]

    iterations: list[ComputationCycle] = [ComputationCycle(0, composites)]

    # iterate until rejection rate stops improving
    prev_reject_rate: float | None = None
    for attempt in range(max_iterations):
        # Step 4: assign all entries
        for i, entry in enumerate(catalogue):
            map_to_best_composite(
                entry,
                iterations[-1].composites,
                rms_limit,
                max_limit,
                cosine_limit,
                derivative_limit,
                i,
                weights=band_weights,
                rms_epsilon=rms_epsilon,
            )

        # Step 5: compute reject rate
        reject_rate = iterations[-1].reject_rate(catalogue.shape[0])

        # Halts iteration when convergence or limit reached
        if (
            prev_reject_rate is not None
            and (
                abs(prev_reject_rate - reject_rate) <= min_reject_rate_delta
                or reject_rate > prev_reject_rate
            )
        ) or attempt == max_iterations - 1:
            iterations.append(
                compute_next_cycle(catalogue, iterations[-1], copy_forward=True)
            )
            break

        # Step 6: recompute median composite shapes for the next iteration using assigned entries only
        prev_reject_rate = reject_rate
        iterations.append(compute_next_cycle(catalogue, iterations[-1]))

    # Step 7: where multiple valid mappings exist, flag all but the best as suboptimal and recompute curves
    mark_suboptimal_mappings(catalogue, iterations)
    compute_next_cycle(catalogue, iterations[-1], copy_forward=True)

    # Step 8: compute fans
    compute_fan_curves(catalogue, iterations[-1].composites, fan_counts)

    return BEQCompositeComputation(
        inputs=responses_db,
        cycles=iterations,
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit,
    )


def mark_suboptimal_mappings(
    catalogue: ndarray[tuple[Any, ...], dtype[Any]], iterations: list[ComputationCycle]
):
    # isolate "best" mappings for cases where a filter can be mapped to multiple composites
    from collections import defaultdict

    mappings_by_entry_id: dict[int, list[BEQFilterMapping]] = defaultdict(list)
    for c in iterations[-1].composites:
        for m in c.mappings:
            if not m.rejected:
                mappings_by_entry_id[m.entry_id].append(m)
    if mappings_by_entry_id and any(
        len(mappings) > 1 for mappings in mappings_by_entry_id.values()
    ):
        iterations.append(
            compute_next_cycle(catalogue, iterations[-1], copy_forward=True)
        )
        for entry_id, mappings in mappings_by_entry_id.items():
            # Convert non‑optimal mappings to rejected
            if len(mappings) > 1:
                best_mapping_by_rms_delta = min(mappings, key=lambda m: m.rms_delta)
                for m in mappings:
                    if m.composite_id != best_mapping_by_rms_delta.composite_id:
                        m.rejection_reason = RejectionReason.SUBOPTIMAL

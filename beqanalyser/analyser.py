import copy
import logging
import time
from multiprocessing import Pool, cpu_count
from typing import Any

import hdbscan
import numpy as np
from numpy import dtype, ndarray

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

logger = logging.getLogger(__name__)


# ------------------------------
# Precompute distance matrix for BEQ curves
# ------------------------------
def _compute_distance_chunk(args):
    """
    Worker function for parallel distance computation.

    Args:
        args: Tuple of (i, end_i, X_float32, X_normalized, distance_params)

    Returns:
        Tuple of (i, end_i, rms_results, cos_results, max_results, deriv_results)
    """
    a = time.time()
    i, end_i, X_float32, X_normalized, params = args

    chunk_i_float = X_float32[i:end_i]
    chunk_i_norm = X_normalized[i:end_i]
    n_samples = X_float32.shape[0]

    # Initialise result arrays for this chunk
    rms_row = np.zeros((end_i - i, n_samples), dtype=np.float32)
    cos_row = np.zeros((end_i - i, n_samples), dtype=np.float32)
    max_row = np.zeros((end_i - i, n_samples), dtype=np.float32)
    deriv_row = (
        np.zeros((end_i - i, n_samples), dtype=np.float32)
        if params["compute_derivative"]
        else None
    )

    # Process in sub-chunks to manage memory
    for j in range(0, n_samples, params["sub_chunk_size"]):
        end_j = min(j + params["sub_chunk_size"], n_samples)
        chunk_j_float = X_float32[j:end_j]
        chunk_j_norm = X_normalized[j:end_j]

        # Compute differences
        diff = chunk_i_float[:, None, :] - chunk_j_float[None, :, :]

        # RMS distance
        rms_dist = np.sqrt(np.mean(diff**2, axis=2))
        rms_row[:, j:end_j] = rms_dist

        # Max absolute deviation
        max_abs_diff = np.max(np.abs(diff), axis=2)
        max_row[:, j:end_j] = max_abs_diff

        # Cosine similarity
        cos_sim = np.dot(chunk_i_norm, chunk_j_norm.T)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        cos_row[:, j:end_j] = cos_sim

        # Derivative RMS if requested
        if params["compute_derivative"]:
            # Compute derivative of differences (finite difference)
            diff_deriv = np.diff(diff, axis=2)
            deriv_rms = np.sqrt(np.mean(diff_deriv**2, axis=2))
            deriv_row[:, j:end_j] = deriv_rms

    b = time.time()
    logger.info(f"Computed chunk {i}-{end_i} in {b-a:.3f}s")
    return i, end_i, rms_row, cos_row, max_row, deriv_row


def compute_beq_distance_matrix(
    X: np.ndarray,
    chunk_size: int = 1000,
    rms_weight: float = 0.5,
    cosine_weight: float = 0.5,
    cosine_scale: float = 10.0,
    rms_limit: float | None = None,
    cosine_limit: float | None = None,
    max_limit: float | None = None,
    derivative_limit: float | None = None,
    penalty_scale: float = 100.0,
    # tolerance parameters
    rms_undershoot_tolerance: float = 2.0,
    rms_close_threshold: float = 2.0,
    cosine_boost_in_close_range: float = 2.0,
    soft_limit_factor: float = 0.7,
    soft_penalty_scale: float = 10.0,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Precompute pairwise distance matrix with a sophisticated penalty structure.

    Features:
    1. Asymmetric RMS penalties (tolerant to undershoot, harsh on overshoot)
    2. Cosine boosting when RMS is close
    3. Tiered penalty system (soft/hard limits)
    4. Parallel processing

    Args:
        X: Input data matrix (n_samples, n_features)
        chunk_size: Number of rows to process at once
        rms_weight: Base weight for RMS component
        cosine_weight: Base weight for cosine component
        cosine_scale: Scaling factor for cosine distance
        rms_limit: Hard upper limit for RMS
        cosine_limit: Hard lower limit for cosine similarity
        max_limit: Hard upper limit for max absolute deviation
        derivative_limit: Hard upper limit for derivative RMS
        penalty_scale: Penalty multiplier for hard limit violations
        rms_undershoot_tolerance: Extra tolerance for RMS undershoot (default: 2.0)
        rms_close_threshold: RMS threshold for "close" range (default: 2.0)
        cosine_boost_in_close_range: Cosine weight multiplier when RMS is close (default: 2.0)
        soft_limit_factor: Multiplier for soft limit (default: 0.7 = 70% of hard limit)
        soft_penalty_scale: Penalty for soft limit violations (default: 10.0)
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = no parallelism)

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    n_cores = cpu_count() if n_jobs == -1 else max(1, n_jobs)

    logger.info(
        f"Computing distance matrix for {n_samples} samples using {n_cores} cores..."
    )
    logger.info(
        f"Base weights: RMS={rms_weight}, Cosine={cosine_weight} (scale={cosine_scale})"
    )
    logger.info(
        f"RMS undershoot tolerance: {rms_undershoot_tolerance}, close threshold: {rms_close_threshold}"
    )
    logger.info(f"Cosine boost in close range: {cosine_boost_in_close_range}x")

    if rms_limit or cosine_limit or max_limit or derivative_limit:
        soft_limits = {
            "rms": rms_limit * soft_limit_factor if rms_limit else None,
            "cosine": cosine_limit + (1.0 - cosine_limit) * (1.0 - soft_limit_factor)
            if cosine_limit
            else None,
            "max": max_limit * soft_limit_factor if max_limit else None,
            "derivative": derivative_limit * soft_limit_factor
            if derivative_limit
            else None,
        }
        logger.info(
            f"Hard limits: RMS={rms_limit}, Cosine={cosine_limit}, Max={max_limit}, Deriv={derivative_limit}"
        )
        logger.info(
            f"Soft limits: RMS={soft_limits['rms']}, Cosine={soft_limits['cosine']}, "
            f"Max={soft_limits['max']}, Deriv={soft_limits['derivative']}"
        )
        logger.info(f"Penalties: Soft={soft_penalty_scale}, Hard={penalty_scale}")

    # Preallocate distance matrix as float64
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    # Precompute normalised vectors for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normalized = (X / norms).astype(np.float32)
    X_float32 = X.astype(np.float32)

    # Prepare worker parameters
    compute_derivative = derivative_limit is not None
    worker_params = {
        "sub_chunk_size": min(int(chunk_size / 2), min(chunk_size, 2500)),
        "compute_derivative": compute_derivative,
    }

    # Prepare chunk arguments for parallel processing
    chunk_args = []
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        chunk_args.append((i, end_i, X_float32, X_normalized, worker_params))

    # Process chunks in parallel
    logger.info(f"Computing base distances on {n_cores} cores in {len(chunk_args)} chunks")
    a = time.time()
    if n_cores > 1:
        with Pool(processes=min(n_cores, len(chunk_args))) as pool:
            results = pool.map(_compute_distance_chunk, chunk_args)
    else:
        results = [_compute_distance_chunk(args) for args in chunk_args]
    b = time.time()
    logger.info(f"Computed base distances in {b-a:.3f}s")

    # Assemble results and compute sophisticated distance
    logger.info("Assembling distance matrix with sophisticated penalties...")

    for i, end_i, rms_row, cos_row, max_row, deriv_row in results:
        n_rows = end_i - i

        # Convert cosine similarity to distance
        cos_dist = 1.0 - cos_row

        # 1. Asymmetric RMS handling
        # For each pair, check if it undershoots or overshoots
        # Undershoot: curve i is lower than curve j (negative mean difference)
        mean_diff = np.mean(X_float32[i:end_i, None, :] - X_float32[None, :, :], axis=2)
        is_undershoot = mean_diff < 0

        # Apply tolerance to undershoot
        rms_adjusted = rms_row.copy()
        if rms_undershoot_tolerance > 1.0:
            undershoot_factor = 1.0 / rms_undershoot_tolerance
            rms_adjusted = np.where(is_undershoot, rms_row * undershoot_factor, rms_row)

        # 2. Detect "close" RMS range and boost cosine weight
        is_close = rms_row < rms_close_threshold
        cosine_weight_adjusted = np.where(
            is_close, cosine_weight * cosine_boost_in_close_range, cosine_weight
        )

        # Compute base distance with adaptive weighting
        base_distance = (
            rms_weight * rms_adjusted + cosine_weight_adjusted * cos_dist * cosine_scale
        )

        # 3. Tiered penalty system
        penalty = np.zeros((n_rows, n_samples), dtype=np.float32)

        if rms_limit is not None:
            soft_rms = soft_limits["rms"]
            # Soft penalty zone
            soft_violations = (rms_row > soft_rms) & (rms_row <= rms_limit)
            penalty += soft_violations.astype(np.float32) * soft_penalty_scale
            # Hard penalty zone (overshoot only - undershoot gets tolerance)
            hard_violations = (rms_row > rms_limit) & ~is_undershoot
            penalty += hard_violations.astype(np.float32) * penalty_scale
            # Undershoot hard violations get a reduced penalty
            hard_undershoot = (rms_row > rms_limit) & is_undershoot
            penalty += hard_undershoot.astype(np.float32) * (
                penalty_scale / rms_undershoot_tolerance
            )

        if cosine_limit is not None:
            soft_cos = soft_limits["cosine"]
            cos_sim = 1.0 - cos_dist
            # Soft penalty zone
            soft_violations = (cos_sim < soft_cos) & (cos_sim >= cosine_limit)
            penalty += soft_violations.astype(np.float32) * soft_penalty_scale
            # Hard penalty zone
            hard_violations = cos_sim < cosine_limit
            penalty += hard_violations.astype(np.float32) * penalty_scale

        if max_limit is not None:
            soft_max = soft_limits["max"]
            # Soft penalty zone
            soft_violations = (max_row > soft_max) & (max_row <= max_limit)
            penalty += soft_violations.astype(np.float32) * soft_penalty_scale
            # Hard penalty zone
            hard_violations = max_row > max_limit
            penalty += hard_violations.astype(np.float32) * penalty_scale

        if derivative_limit is not None and deriv_row is not None:
            soft_deriv = soft_limits["derivative"]
            # Soft penalty zone
            soft_violations = (deriv_row > soft_deriv) & (deriv_row <= derivative_limit)
            penalty += soft_violations.astype(np.float32) * soft_penalty_scale
            # Hard penalty zone
            hard_violations = deriv_row > derivative_limit
            penalty += hard_violations.astype(np.float32) * penalty_scale

        # Combine base distance with penalties
        distance_matrix[i:end_i, :] = (base_distance + penalty).astype(np.float64)

        logger.info(f"  Processed {end_i}/{n_samples} rows")

    c = time.time()
    logger.info(f"Computed distance final in {c-b:.3f}s")

    # Log penalty statistics
    if rms_limit or cosine_limit or max_limit or derivative_limit:
        n_soft = np.sum((distance_matrix > 0) & (distance_matrix < penalty_scale))
        n_hard = np.sum(distance_matrix >= penalty_scale)
        total_pairs = n_samples * n_samples
        logger.info(
            f"Penalties: {n_soft:,} soft violations ({100.0 * n_soft / total_pairs:.2f}%), "
            f"{n_hard:,} hard violations ({100.0 * n_hard / total_pairs:.2f}%)"
        )

    logger.info(
        f"Distance matrix computed in {c-a:.3f}s. Range: [{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]"
    )
    return distance_matrix


# ------------------------------
# HDBSCAN clustering
# ------------------------------
def hdbscan_clustering(
    X: np.ndarray,
    min_cluster_size: int = 500,
    min_samples: int = 50,
    cluster_selection_epsilon: float = 0.0,
    distance_chunk_size: int = 1000,
    distance_rms_weight: float = 0.5,
    distance_cosine_weight: float = 0.5,
    distance_cosine_scale: float = 10.0,
    distance_rms_limit: float | None = None,
    distance_cosine_limit: float | None = None,
    distance_max_limit: float | None = None,
    distance_derivative_limit: float | None = None,
    distance_penalty_scale: float = 100.0,
    distance_rms_undershoot_tolerance: float = 2.0,
    distance_rms_close_threshold: float = 2.0,
    distance_cosine_boost_in_close_range: float = 2.0,
    distance_soft_limit_factor: float = 0.7,
    distance_soft_penalty_scale: float = 10.0,
    distance_n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use HDBSCAN to identify natural cluster structure directly.

    Args:
        X: Input data matrix (n_samples, n_features)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples parameter for HDBSCAN
        cluster_selection_epsilon: Distance threshold for cluster merging (0.0 = no merging)
        distance_chunk_size: Chunk size for distance matrix computation (lower = less memory)
        distance_rms_weight: Weight for RMS component in distance metric (default: 0.5)
        distance_cosine_weight: Weight for cosine component in distance metric (default: 0.5)
        distance_cosine_scale: Scaling factor for cosine distance (default: 10.0)
        distance_rms_limit: If set, penalize pairs with RMS > limit
        distance_cosine_limit: If set, penalize pairs with cosine similarity < limit
        distance_max_limit: If set, penalize pairs with max absolute deviation > limit
        distance_derivative_limit: If set, penalize pairs with derivative RMS > limit
        distance_penalty_scale: Penalty multiplier for hard limit violations (default: 100.0)
        distance_rms_undershoot_tolerance: Extra tolerance for RMS undershoot (default: 2.0)
        distance_rms_close_threshold: RMS threshold for "close" range (default: 2.0)
        distance_cosine_boost_in_close_range: Cosine weight boost when RMS is close (default: 2.0)
        distance_soft_limit_factor: Soft limit as fraction of hard limit (default: 0.7)
        distance_soft_penalty_scale: Penalty for soft limit violations (default: 10.0)
        distance_n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)

    Returns:
        Tuple of (cluster_labels, cluster_medoids)
        - cluster_labels: Array of cluster assignments for each input
        - cluster_medoids: Representative curves (medoid) for each cluster
    """
    # Precompute distance matrix with sophisticated penalties
    distance_matrix = compute_beq_distance_matrix(
        X,
        chunk_size=distance_chunk_size,
        rms_weight=distance_rms_weight,
        cosine_weight=distance_cosine_weight,
        cosine_scale=distance_cosine_scale,
        rms_limit=distance_rms_limit,
        cosine_limit=distance_cosine_limit,
        max_limit=distance_max_limit,
        derivative_limit=distance_derivative_limit,
        penalty_scale=distance_penalty_scale,
        rms_undershoot_tolerance=distance_rms_undershoot_tolerance,
        rms_close_threshold=distance_rms_close_threshold,
        cosine_boost_in_close_range=distance_cosine_boost_in_close_range,
        soft_limit_factor=distance_soft_limit_factor,
        soft_penalty_scale=distance_soft_penalty_scale,
        n_jobs=distance_n_jobs,
    )

    # Run HDBSCAN clustering with precomputed distances
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom",  # Excess of Mass
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=False,
    )

    labels = clusterer.fit_predict(distance_matrix)

    # Get unique cluster labels (excluding noise labeled as -1)
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1)

    if n_noise > 0:
        logger.info(
            f"HDBSCAN identified {n_noise} noise points ({100.0 * n_noise / X.shape[0]:.1f}%)"
        )

    if n_clusters == 0:
        # If HDBSCAN found no clusters (all noise), create a single cluster
        logger.warning(
            "HDBSCAN found no clusters. Creating single cluster from all data."
        )
        labels = np.zeros(X.shape[0], dtype=int)
        unique_labels = np.array([0])
        n_clusters = 1

    # Compute medoid for each cluster (point closest to cluster median)
    cluster_medoids = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = X[cluster_mask]

        # Find medoid: point in cluster closest to the median
        cluster_median = np.median(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - cluster_median, axis=1)
        medoid_idx = np.argmin(distances)
        cluster_medoids.append(cluster_points[medoid_idx])

    # Keep noise points as noise (label=-1) - they will be handled during assignment phase
    # The iterative refinement will attempt to assign them, and they'll be rejected if they
    # don't meet the acceptance criteria

    return labels, np.array(cluster_medoids)


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
    best_metrics = (
        max(candidates, key=lambda m: m.cosine_similarity) if candidates else None
    )

    # flag all other composites as suboptimal unless already rejected
    for mapping in mappings:
        if best_metrics and mapping.composite_id != best_metrics.composite_id:
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
    rms_limit: float = 5.0,
    max_limit: float = 5.0,
    cosine_limit: float = 0.95,
    derivative_limit: float = 2.0,
    fan_counts: tuple[int, ...] = (5,),
    rms_epsilon: float = 0.5,
    max_iterations: int = 10,
    min_reject_rate_delta: float = 0.005,
    hdbscan_min_cluster_size: int = 500,
    hdbscan_min_samples: int | None = 50,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_distance_chunk_size: int = 5000,
    hdbscan_distance_rms_weight: float = 0.8,
    hdbscan_distance_cosine_weight: float = 0.2,
    hdbscan_distance_cosine_scale: float = 10.0,
    hdbscan_use_constraints: bool = True,
    hdbscan_distance_penalty_scale: float = 100.0,
    hdbscan_distance_rms_undershoot_tolerance: float = 2.0,
    hdbscan_distance_rms_close_threshold: float = 2.0,
    hdbscan_distance_cosine_boost_in_close_range: float = 2.0,
    hdbscan_distance_soft_limit_factor: float = 0.7,
    hdbscan_distance_soft_penalty_scale: float = 10.0,
    hdbscan_distance_n_jobs: int = -1,
) -> BEQCompositeComputation:
    """
    Build BEQ composite filters using HDBSCAN clustering.

    Args:
        responses_db: Full frequency response database
        freqs: Frequency array
        weights: Optional frequency weights
        band: Frequency band to analyze (min_freq, max_freq)
        rms_limit: Maximum acceptable RMS deviation for assignment
        max_limit: Maximum acceptable absolute deviation
        cosine_limit: Minimum acceptable cosine similarity
        derivative_limit: Maximum acceptable derivative RMS
        fan_counts: Number of curves in each fan level
        rms_epsilon: Tolerance for near-minimum RMS selection
        max_iterations: Maximum refinement iterations
        min_reject_rate_delta: Minimum change in reject rate to continue iterating
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN (controls number of clusters)
        hdbscan_min_samples: Minimum samples for core points (controls cluster density)
        hdbscan_cluster_selection_epsilon: Distance threshold for merging clusters (0.0 = no merging)
        hdbscan_distance_chunk_size: Chunk size for distance computation (lower = less memory, default 1000)
        hdbscan_distance_rms_weight: Weight for RMS component in distance (default: 0.5)
        hdbscan_distance_cosine_weight: Weight for cosine component in distance (default: 0.5)
        hdbscan_distance_cosine_scale: Scaling factor for cosine to match RMS range (default: 10.0)
        hdbscan_use_constraints: If True, penalize pairs that violate acceptance criteria (default: True)
        hdbscan_distance_penalty_scale: Hard penalty multiplier for constraint violations (default: 100.0)
        hdbscan_distance_rms_undershoot_tolerance: Tolerance factor for RMS undershoot (default: 2.0)
        hdbscan_distance_rms_close_threshold: RMS threshold for cosine boosting (default: 2.0)
        hdbscan_distance_cosine_boost_in_close_range: Cosine weight multiplier when close (default: 2.0)
        hdbscan_distance_soft_limit_factor: Soft limit as fraction of hard limit (default: 0.7)
        hdbscan_distance_soft_penalty_scale: Soft penalty multiplier (default: 10.0)
        hdbscan_distance_n_jobs: Number of parallel jobs for distance computation (default: -1 = all CPUs)

    Returns:
        BEQCompositeComputation with all cycles and final composites

    Note:
        The number of final composites is determined by HDBSCAN parameters:
        - Increase min_cluster_size to get fewer, larger clusters
        - Increase min_samples to get more conservative, tighter clusters
        - Increase cluster_selection_epsilon to merge similar clusters
    """
    band_mask: np.ndarray = (freqs >= band[0]) & (freqs <= band[1])
    catalogue: np.ndarray = responses_db[:, band_mask]
    band_weights: np.ndarray | None = (
        weights[band_mask] if weights is not None else None
    )

    # Step 1: HDBSCAN clustering (replaces k-medoids + Ward)
    labels, cluster_medoids = hdbscan_clustering(
        catalogue,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        distance_chunk_size=hdbscan_distance_chunk_size,
        distance_rms_weight=hdbscan_distance_rms_weight,
        distance_cosine_weight=hdbscan_distance_cosine_weight,
        distance_cosine_scale=hdbscan_distance_cosine_scale,
        distance_rms_limit=rms_limit if hdbscan_use_constraints else None,
        distance_cosine_limit=cosine_limit if hdbscan_use_constraints else None,
        distance_max_limit=max_limit if hdbscan_use_constraints else None,
        distance_derivative_limit=derivative_limit if hdbscan_use_constraints else None,
        distance_penalty_scale=hdbscan_distance_penalty_scale,
        distance_rms_undershoot_tolerance=hdbscan_distance_rms_undershoot_tolerance,
        distance_rms_close_threshold=hdbscan_distance_rms_close_threshold,
        distance_cosine_boost_in_close_range=hdbscan_distance_cosine_boost_in_close_range,
        distance_soft_limit_factor=hdbscan_distance_soft_limit_factor,
        distance_soft_penalty_scale=hdbscan_distance_soft_penalty_scale,
        distance_n_jobs=hdbscan_distance_n_jobs,
    )

    n_clusters = len(cluster_medoids)
    logger.info(f"HDBSCAN found {n_clusters} natural clusters")

    # Step 2: Create initial composites from cluster medians
    composites: list[BEQComposite] = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_curves = catalogue[cluster_mask]
        # Use median of cluster rather than medoid for initial composite
        composite_curve = np.median(cluster_curves, axis=0)
        composites.append(BEQComposite(cluster_id, composite_curve))

    iterations: list[ComputationCycle] = [ComputationCycle(0, composites)]

    # Log initial cluster statistics
    n_noise = np.sum(labels == -1)
    if n_noise > 0:
        logger.info(
            f"Initial clusters: {n_clusters} composites cover {catalogue.shape[0] - n_noise} curves "
            f"({n_noise} noise points will be assigned during iteration)"
        )

    # iterate until rejection rate stops improving
    prev_reject_rate: float | None = None
    for attempt in range(max_iterations):
        # Step 3: assign all entries
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

        # Step 4: compute reject rate
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

        # Step 5: recompute median composite shapes for the next iteration using assigned entries only
        prev_reject_rate = reject_rate
        iterations.append(compute_next_cycle(catalogue, iterations[-1]))

    # Step 6: where multiple valid mappings exist, flag all but the best as suboptimal and recompute curves
    mark_suboptimal_mappings(catalogue, iterations)
    compute_next_cycle(catalogue, iterations[-1], copy_forward=True)

    # Step 7: compute fans
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

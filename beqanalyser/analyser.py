import copy
import dataclasses
import logging
from collections import defaultdict

import hdbscan
import numpy as np

from beqanalyser import (
    BEQComposite,
    BEQCompositeComputation,
    BEQFilterMapping,
    BEQResult,
    ComputationCycle,
    RejectionReason,
    cosine_similarity,
    derivative_rms,
    rms, DistanceParams, HDBSCANParams, )
from beqanalyser.loader import compute_beq_distance_matrix, compute_distance_components

logger = logging.getLogger(__name__)


def compute_composite_distance(
    entry: np.ndarray,
    composite: np.ndarray,
    params: DistanceParams,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute sophisticated distance score for a single entry-composite pair.

    Uses the same logic as compute_beq_distance_matrix for consistency.

    Args:
        entry: Input curve to compare
        composite: Composite curve to compare against
        params: Parameters controlling distance computation
        weights: Optional frequency weights for RMS calculation

    Returns:
        Tuple of (distance_score, rms_delta, max_delta, cosine_similarity, derivative_delta)
    """
    # Compute basic metrics
    delta = entry - composite
    rms_delta = rms(delta, weights)
    max_delta = float(np.max(np.abs(delta)))
    cos_sim = cosine_similarity(entry, composite)
    deriv_delta = derivative_rms(delta)
    mean_diff = float(np.mean(delta))

    # Use common distance computation logic
    distance_score = compute_distance_components(
        rms_vals=rms_delta,
        cos_sim_vals=cos_sim,
        max_vals=max_delta,
        deriv_vals=deriv_delta,
        mean_diff_vals=mean_diff,
        params=params,
    )

    return float(distance_score), rms_delta, max_delta, cos_sim, deriv_delta


# ------------------------------
# HDBSCAN clustering
# ------------------------------
def hdbscan_clustering(
    X: np.ndarray,
    distance_params: DistanceParams,
    hdbscan_params: HDBSCANParams,
    precomputed_distance_matrix: np.ndarray | None = None,
    entry_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use HDBSCAN to identify natural cluster structure directly.

    Args:
        X: Input data matrix (n_samples, n_features)
        distance_params: Distance computation parameters
        hdbscan_params: HDBSCAN parameters
        precomputed_distance_matrix: Optional precomputed full distance matrix
        entry_indices: Optional indices mapping X rows to full distance matrix

    Returns:
        Tuple of (cluster_labels, cluster_medoids, distance_matrix)
        - cluster_labels: Array of cluster assignments for each input
        - cluster_medoids: Representative curves (medoid) for each cluster
        - distance_matrix: The distance matrix used (for reuse in subsequent calls)
    """
    if precomputed_distance_matrix is not None and entry_indices is not None:
        # Extract submatrix for the given indices
        logger.info(
            f"Reusing precomputed distance matrix, extracting submatrix for {len(entry_indices)} indices"
        )
        distance_matrix = precomputed_distance_matrix[
            np.ix_(entry_indices, entry_indices)
        ]
    else:
        # Compute full distance matrix
        logger.info("Computing distance matrix from scratch")
        distance_matrix = compute_beq_distance_matrix(
            X,
            chunk_size=distance_params.distance_chunk_size,
            rms_weight=distance_params.distance_rms_weight,
            cosine_weight=distance_params.distance_cosine_weight,
            cosine_scale=distance_params.distance_cosine_scale,
            rms_limit=distance_params.rms_limit
            if distance_params.use_constraints
            else None,
            cosine_limit=distance_params.cosine_limit
            if distance_params.use_constraints
            else None,
            max_limit=distance_params.max_limit
            if distance_params.use_constraints
            else None,
            derivative_limit=distance_params.derivative_limit
            if distance_params.use_constraints
            else None,
            penalty_scale=distance_params.distance_penalty_scale,
            rms_undershoot_tolerance=distance_params.distance_rms_undershoot_tolerance,
            rms_close_threshold=distance_params.distance_rms_close_threshold,
            cosine_boost_in_close_range=distance_params.distance_cosine_boost_in_close_range,
            soft_limit_factor=distance_params.distance_soft_limit_factor,
            soft_penalty_scale=distance_params.distance_soft_penalty_scale,
            n_jobs=distance_params.distance_n_jobs,
        )

    # Run HDBSCAN clustering with precomputed distances
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_params.min_cluster_size,
        min_samples=hdbscan_params.min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
        cluster_selection_epsilon=hdbscan_params.cluster_selection_epsilon,
        allow_single_cluster=False,
    )

    labels = clusterer.fit_predict(distance_matrix)

    # Get unique cluster labels (excluding noise labeled as -1)
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1)

    logger.info(
        f"hdbscan identified {n_noise} of {X.shape[0]} as noise({100.0 * n_noise / X.shape[0]:.1f}%) and {n_clusters} clusters"
    )

    if n_clusters == 0:
        logger.warning(
            "HDBSCAN found no clusters. Creating single cluster from all data."
        )
        labels = np.zeros(X.shape[0], dtype=int)
        unique_labels = np.array([0])

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

    return labels, np.array(cluster_medoids), distance_matrix


# ------------------------------
# Assignment function
# ------------------------------
def map_to_best_composite(
    entry: np.ndarray,
    composites: list[BEQComposite],
    params: DistanceParams,
    entry_id: int,
    cluster_label: int,
    weights: np.ndarray | None = None,
) -> None:
    """
    Records the

    The composite with the lowest distance score is selected as best. Entries are rejected
    if their distance score indicates hard limit violations (determined by penalty thresholds),
    otherwise all others are marked as suboptimal.

    Args:
        entry: Input curve to assign
        composites: List of composite candidates
        params: Parameters controlling distance computation and limits
        entry_id: Index of entry in catalogue
        cluster_label: Cluster label assigned by HDBSCAN, -1 if identified as noise
        weights: Optional frequency weights for RMS calculation
    """
    composite_scores: list[tuple[float, BEQFilterMapping]] = []

    # Define the rejection threshold based on hard penalties
    # If the distance score includes hard penalties, it indicates constraint violations
    rejection_threshold = (
        params.distance_penalty_scale if params.use_constraints else float("inf")
    )

    for comp in composites:
        distance_score, rms_delta, max_delta, cos_sim, deriv_delta = (
            compute_composite_distance(entry, comp.mag_response, params, weights)
        )

        mapping = BEQFilterMapping(
            composite_id=comp.id,
            entry_id=entry_id,
            rms_delta=rms_delta,
            max_delta=max_delta,
            derivative_delta=deriv_delta,
            cosine_similarity=cos_sim,
            distance_score=distance_score,
        )

        # Reject if the distance score indicates a hard limit violation or if it was already identified as noise
        if cluster_label == -1:
            mapping.rejection_reason = RejectionReason.NOISE
        # elif distance_score >= rejection_threshold:
        #     mapping.rejection_reason = RejectionReason.HARD_LIMIT

        composite_scores.append((distance_score, mapping))
        comp.mappings.append(mapping)

    # Find the composite with the minimum distance score and mark it as best
    best_score, best_mapping = min(composite_scores, key=lambda x: x[0])
    best_mapping.is_best = True

    # Mark all others as suboptimal
    for distance_score, mapping in composite_scores:
        if mapping.composite_id != best_mapping.composite_id and not mapping.rejected:
            mapping.rejection_reason = RejectionReason.SUBOPTIMAL

    best_composites = [
        c
        for c in composites
        if any(m.is_best is True for m in c.mappings if m.entry_id == entry_id)
    ]
    assert len(best_composites) == 1, (
        f"Expected 1 best mapping, got {len(best_composites)}"
    )


def assign_remaining_entries(
    input_curves: np.ndarray,
    remaining_entry_ids: list[int],
    all_composites: list[BEQComposite],
    freqs: np.ndarray,
    params: DistanceParams,
    weights: np.ndarray | None = None,
    band: tuple[float, float] = (5, 50),
    distance_threshold_multiplier: float = 1.5,
) -> tuple[int, int]:
    """
    Attempt to assign remaining rejected/noise entries to discovered composites.

    This is the final assignment phase that runs after all cluster discovery iterations.
    It uses a relaxed distance threshold to catch "near-miss" outliers while still
    rejecting true outliers that don't fit any discovered pattern.

    Args:
        input_curves: Full frequency response database
        remaining_entry_ids: Entry IDs that were not assigned during clustering
        all_composites: All composites discovered across all iterations
        freqs: Frequency array
        params: Base parameters for distance computation
        weights: Optional frequency weights
        band: Frequency band to analyze
        distance_threshold_multiplier: Multiplier for acceptable distance threshold.
            Values > 1.0 allow more lenient assignment of outliers.

    Returns:
        Tuple of (newly_assigned_count, still_rejected_count)
    """
    if not remaining_entry_ids or not all_composites:
        return 0, len(remaining_entry_ids) if remaining_entry_ids else 0

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    masked_catalogue = input_curves[:, band_mask]
    band_weights = weights[band_mask] if weights is not None else None

    logger.info(
        f"Attempting final assignment of {len(remaining_entry_ids)} remaining entries "
        f"to {len(all_composites)} composites with threshold multiplier {distance_threshold_multiplier}"
    )

    # Create relaxed parameters for final assignment
    relaxed_params = copy.deepcopy(params)
    if relaxed_params.rms_limit is not None:
        relaxed_params.rms_limit *= distance_threshold_multiplier
    if relaxed_params.max_limit is not None:
        relaxed_params.max_limit *= distance_threshold_multiplier
    if relaxed_params.derivative_limit is not None:
        relaxed_params.derivative_limit *= distance_threshold_multiplier
    if relaxed_params.cosine_limit is not None:
        # For cosine, move threshold toward more lenient (lower value)
        relaxed_params.cosine_limit = max(
            0.0,
            relaxed_params.cosine_limit
            - (1.0 - relaxed_params.cosine_limit)
            * (distance_threshold_multiplier - 1.0),
        )

    # Rejection threshold for relaxed parameters
    rejection_threshold = (
        relaxed_params.distance_penalty_scale
        if relaxed_params.use_constraints
        else float("inf")
    )

    still_rejected = 0
    newly_assigned_counts_by_composite = defaultdict(int)
    previously_assigned_counts_by_composite = {
        c.id: len(c.assigned_entry_ids)
        for c in sorted(all_composites, key=lambda c: c.id)
    }

    for entry_id in remaining_entry_ids:
        entry = masked_catalogue[entry_id]

        # Compute distance to all composites
        best_distance = float("inf")
        best_composite = None
        best_mapping = None

        for comp in all_composites:
            distance_score, rms_delta, max_delta, cos_sim, deriv_delta = (
                compute_composite_distance(
                    entry, comp.mag_response, relaxed_params, band_weights
                )
            )

            if distance_score < best_distance:
                best_distance = distance_score
                best_composite = comp
                best_mapping = BEQFilterMapping(
                    composite_id=comp.id,
                    entry_id=entry_id,
                    rms_delta=rms_delta,
                    max_delta=max_delta,
                    derivative_delta=deriv_delta,
                    cosine_similarity=cos_sim,
                    distance_score=distance_score,
                )

        # Check if rejected based on distance score
        if best_mapping:
            if best_distance >= rejection_threshold:
                best_mapping.rejection_reason = RejectionReason.HARD_LIMIT
                still_rejected += 1
            else:
                best_mapping.is_best = True
                newly_assigned_counts_by_composite[best_composite.id] = (
                    newly_assigned_counts_by_composite[best_composite.id] + 1
                )

            best_composite.mappings.append(best_mapping)

    for composite_id, count in previously_assigned_counts_by_composite.items():
        new_count = newly_assigned_counts_by_composite[composite_id]
        logger.info(
            f"  {composite_id}: {count} + {new_count} = {count + new_count} entries assigned (+{new_count / (count + new_count):.1%})"
        )

    newly_assigned = sum(newly_assigned_counts_by_composite.values())
    logger.info(
        f"Final assignment: {newly_assigned} newly assigned, {still_rejected} still rejected "
        f"({100.0 * newly_assigned / len(remaining_entry_ids):.1f}% success rate)"
    )

    return newly_assigned, still_rejected


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
        next_cycle.is_copy = True
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
def build_all_composites(
    input_curves: np.ndarray,
    freqs: np.ndarray,
    full_distance_matrix: np.ndarray,
    distance_params: DistanceParams,
    iteration_params: list[HDBSCANParams],
    weights: np.ndarray | None = None,
    band: tuple[float, float] = (5, 50),
    fan_counts: tuple[int, ...] = (5,),
    max_iterations: int = 20,
    min_reject_rate_delta: float = 0.005,
    final_assignment_threshold_multiplier: float = 1.0,
) -> BEQResult:
    """
    Build BEQ composite filters via an iterative HDBSCAN clustering with final assignment phase.

    This function now precomputes the full distance matrix once and reuses it across all
    iterations by extracting relevant submatrices.

    Args:
        input_curves: Full frequency response database
        freqs: Frequency array.
        full_distance_matrix: Precomputed full distance matrix.
        distance_params: Parameters controlling distance computation and limits
        iteration_params: parameters for each iteration, predominantly aimed at hdbscan configuration.
        weights: Optional frequency weights
        band: Frequency band to analyse (min_freq, max_freq)
        fan_counts: Number of curves in each fan level
        max_iterations: Maximum refinement iterations per HDBSCAN run
        min_reject_rate_delta: Minimum change in reject rate to continue iterating
        final_assignment_threshold_multiplier: Multiplier for the distance threshold in final assignment.

    Returns:
        a BEQResult object containing all cycles and final composites
    """
    calcs: list[BEQCompositeComputation] = []
    weights = weights if weights is not None else np.ones_like(freqs)
    assigned_rate = 1.0
    param_idx = 0

    all_entry_ids = list(range(0, input_curves.shape[0]))

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    full_masked_catalogue = input_curves[:, band_mask]

    # Phase 1: Discovery - iteratively find clusters, keep noise as rejected
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1: CLUSTER DISCOVERY (reusing precomputed distance matrix)")
    logger.info("=" * 80)

    while assigned_rate >= 0.01 and param_idx < len(iteration_params):
        if calcs:
            entry_ids = calcs[-1].result.rejected_entry_ids
        else:
            entry_ids = all_entry_ids

        input_size = len(entry_ids)

        logger.info(
            f"Running iteration {param_idx + 1} for {input_size} entries with params {iteration_params[param_idx]}"
        )

        last_calc = build_beq_composites(
            input_curves=input_curves,
            entry_ids=entry_ids,
            freqs=freqs,
            weights=weights,
            band=band,
            fan_counts=fan_counts,
            distance_params=distance_params,
            hdbscan_params=iteration_params[param_idx],
            max_iterations=max_iterations,
            min_reject_rate_delta=min_reject_rate_delta,
            precomputed_distance_matrix=full_distance_matrix,
        )
        calcs.append(last_calc)
        assigned_count = sum([a.total_assigned_count for a in calcs])
        composite_count = sum([len(a.result.composites) for a in calcs])
        assigned_rate = assigned_count / input_curves.shape[0]
        logger.info(
            f"After iteration {param_idx + 1} total assignment rate: {assigned_rate:.2%}, composites: {composite_count}"
        )
        param_idx += 1

    # Phase 2: Final assignment of remaining entries
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: FINAL ASSIGNMENT (attempt to map remaining entries)")
    logger.info("=" * 80)

    result = create_final_result(calcs)
    assigned_ids = result.assigned_entry_ids
    remaining_ids = set(all_entry_ids) - assigned_ids
    if remaining_ids and final_assignment_threshold_multiplier > 0:
        newly_assigned, still_rejected = assign_remaining_entries(
            input_curves=input_curves,
            remaining_entry_ids=list(remaining_ids),
            all_composites=result.composites,
            freqs=freqs,
            params=distance_params,
            weights=weights,
            band=band,
            distance_threshold_multiplier=final_assignment_threshold_multiplier,
        )
        logger.info(
            f"Remaining entry assignment results: {newly_assigned} newly assigned, {still_rejected} still rejected"
        )

        if newly_assigned > 0:
            # Recompute composite curves with newly assigned entries
            for comp in result.composites:
                if comp.assigned_entry_ids:
                    assigned = full_masked_catalogue[comp.assigned_entry_ids, :]
                    comp.mag_response = np.median(assigned, axis=0)

            # Recompute fan curves
            compute_fan_curves(full_masked_catalogue, result.composites, fan_counts)

        # Update result statistics
        total_assigned = sum(len(c.assigned_entry_ids) for c in result.composites)
        final_rate = total_assigned / input_curves.shape[0]
        logger.info(
            f"Final assignment rate: {final_rate:.2%} "
            f"({total_assigned}/{input_curves.shape[0]} entries assigned)"
        )
    else:
        logger.info("No remaining entries to assign or final assignment disabled")

    return result


def build_beq_composites(
    input_curves: np.ndarray,
    entry_ids: list[int],
    freqs: np.ndarray,
    distance_params: DistanceParams,
    hdbscan_params: HDBSCANParams,
    weights: np.ndarray | None = None,
    band: tuple[float, float] = (5, 50),
    fan_counts: tuple[int, ...] = (5,),
    max_iterations: int = 10,
    min_reject_rate_delta: float = 0.005,
    precomputed_distance_matrix: np.ndarray | None = None,
) -> BEQCompositeComputation:
    """
    Build BEQ composite filters using HDBSCAN clustering.

    Args:
        input_curves: Full frequency response database
        entry_ids: ids of the input curves
        freqs: Frequency array
        hdbscan_params: HDBSCAN parameters
        weights: Optional frequency weights
        band: Frequency band to analyze (min_freq, max_freq)
        fan_counts: Number of curves in each fan level
        max_iterations: Maximum refinement iterations
        min_reject_rate_delta: Minimum change in reject rate to continue iterating
        precomputed_distance_matrix: Optional precomputed distance matrix for full catalogue

    Returns:
        BEQCompositeComputation with all cycles and final composites
    """
    band_mask: np.ndarray = (freqs >= band[0]) & (freqs <= band[1])
    full_masked_catalogue = input_curves[:, band_mask]
    scoped_masked_catalogue: np.ndarray = input_curves[entry_ids][:, band_mask]
    band_weights: np.ndarray | None = (
        weights[band_mask] if weights is not None else None
    )

    # Convert entry_ids to numpy array for efficient indexing
    entry_indices = np.array(entry_ids)

    # Step 1: HDBSCAN clustering (replaces k-medoids + Ward)
    labels, cluster_medoids, _ = hdbscan_clustering(
        scoped_masked_catalogue,
        distance_params,
        hdbscan_params,
        precomputed_distance_matrix=precomputed_distance_matrix,
        entry_indices=entry_indices,
    )

    # Step 2: Create initial composites from cluster medians
    composites: list[BEQComposite] = []
    for cluster_id in range(len(cluster_medoids)):
        cluster_mask = labels == cluster_id
        cluster_curves = scoped_masked_catalogue[cluster_mask]
        # prefer cluster medoid as the composite unless it's a long way from the median (unlikely)
        median_curve = np.median(cluster_curves, axis=0)
        medoid_curve = cluster_medoids[cluster_id]
        medoid_distance, _, _, _, _ = compute_composite_distance(
            medoid_curve, median_curve, distance_params
        )
        # Median drifted too far, fall back to medoid
        medoid_too_far = medoid_distance > distance_params.distance_penalty_scale
        composite_curve = median_curve if medoid_too_far else medoid_curve
        composites.append(BEQComposite(cluster_id, composite_curve))

    iterations: list[ComputationCycle] = [ComputationCycle(0, composites)]

    # Log initial cluster statistics
    n_noise = np.sum(labels == -1)
    in_scope_catalogue_size = scoped_masked_catalogue[labels != -1].shape[0]
    if n_noise > 0:
        old_cat_size = scoped_masked_catalogue.shape[0]
        logger.info(
            f"Ignoring noise: catalogue reduced from {old_cat_size} entries to {in_scope_catalogue_size}"
        )
    else:
        logger.info("No noise points found, entire catalogue will be assigned")

    # iterate until the rejection rate stops improving
    prev_reject_rate: float = 100.0
    for attempt in range(max_iterations):
        # Step 3: assign all entries
        for i, entry in enumerate(scoped_masked_catalogue):
            map_to_best_composite(
                entry,
                iterations[-1].composites,
                distance_params,
                entry_ids[i],
                labels[i],
                weights=band_weights,
            )

        # Step 4: compute reject rate
        reject_rate = iterations[-1].reject_rate(in_scope_catalogue_size)

        # Halts iteration when convergence or limit reached
        if (
            (
                abs(prev_reject_rate - reject_rate) <= min_reject_rate_delta
                or reject_rate > prev_reject_rate
            )
            or attempt == max_iterations - 1
            or reject_rate == 0.0
        ):
            if len(iterations) > 1:
                copy_from = -2 if reject_rate > prev_reject_rate else -1
                iterations.append(
                    compute_next_cycle(
                        full_masked_catalogue, iterations[copy_from], copy_forward=True
                    )
                )
            break

        # Step 5: recompute median composite shapes for the next iteration using assigned entries only
        prev_reject_rate = reject_rate
        iterations.append(compute_next_cycle(full_masked_catalogue, iterations[-1]))

    # Step 6: compute fans
    compute_fan_curves(full_masked_catalogue, iterations[-1].composites, fan_counts)

    computation = BEQCompositeComputation(
        inputs=input_curves,
        cycles=iterations,
    )

    logging.info(
        f"Entries: {scoped_masked_catalogue.shape[0]}, in scope: {in_scope_catalogue_size}, composites: {len(composites)}, assigned: {computation.total_assigned_count}, rejected {computation.total_rejected_count}"
    )

    delta = (
        scoped_masked_catalogue.shape[0]
        - computation.total_assigned_count
        - computation.total_rejected_count
    )
    assert delta == 0, f"Delta should be 0, got {delta}"

    return computation


def create_final_result(
    calcs: list[BEQCompositeComputation],
) -> BEQResult:
    """
    Creates a new computation that collates all the composites from each iteration into a single cycle. All catalogue
    references are remapped to the original input catalogue correctly.
    :param calcs: the output of each iteration.
    :return: the original calcs with the new computation appended.
    """
    if not calcs:
        raise ValueError("No computations to merge")

    composites: list[BEQComposite] = []
    inputs: np.ndarray | None = None

    # extract final composites from every calculation cycle into a single cycle
    for c in calcs:
        if inputs is None:
            inputs = c.inputs
            composites.extend(c.result.composites)
        else:
            for comp in c.result.composites:
                comp_id = len(composites)
                composites.append(
                    BEQComposite(
                        comp_id,
                        comp.mag_response,
                        mappings=[
                            dataclasses.replace(m, composite_id=comp_id)
                            for m in comp.mappings
                        ],
                        fan_envelopes=comp.fan_envelopes,
                    )
                )

    return BEQResult(inputs, composites, calcs)

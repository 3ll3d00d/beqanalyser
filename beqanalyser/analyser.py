import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from beqanalyser import BEQComposite, AssignmentRecord, RejectionReason, rms, derivative_rms, cosine_similarity, \
    BEQCompositePipelineResult


# ------------------------------
# Pure NumPy K-medoids
# ------------------------------
def k_medoids(X: np.ndarray, n_clusters: int, max_iter: int = 100, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
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
            total_dists = np.sum(np.linalg.norm(cluster_points[:, None, :] - cluster_points[None, :, :], axis=2),
                                 axis=1)
            min_idx = np.argmin(total_dists)
            new_medoids[i] = np.where((X == cluster_points[min_idx]).all(axis=1))[0][0]

        if np.all(new_medoids == medoid_indices):
            break
        medoid_indices = new_medoids

    return medoid_indices


# ------------------------------
# Assignment function
# ------------------------------
def assign_to_composites_with_record(
        entry: np.ndarray,
        composites: list[BEQComposite],
        rms_limit: float,
        max_limit: float,
        cosine_limit: float,
        derivative_limit: float,
        entry_idx: int,
        weights: np.ndarray | None = None
) -> AssignmentRecord:
    best_comp: BEQComposite | None = None
    best_comp_idx: int | None = None

    best_rms: float = np.inf
    best_max: float = np.inf
    best_deriv: float = np.inf
    best_cos: float = -1.0

    rejection: RejectionReason | None = None
    rms_val = max_val = deriv_val = cos_val = None

    for comp_idx, comp in enumerate(composites):
        delta = entry - comp.shape
        entry_rms = rms(delta, weights)
        entry_max = float(np.max(np.abs(delta)))
        entry_deriv = derivative_rms(delta)
        entry_cos = cosine_similarity(entry, comp.shape)

        reject: RejectionReason | None = None
        if entry_rms > rms_limit and entry_max > max_limit:
            reject = RejectionReason.BOTH_EXCEEDED
        elif entry_rms > rms_limit:
            reject = RejectionReason.RMS_EXCEEDED
        elif entry_max > max_limit:
            reject = RejectionReason.MAX_EXCEEDED
        elif entry_cos < cosine_limit:
            reject = RejectionReason.COSINE_TOO_LOW
        elif entry_deriv > derivative_limit:
            reject = RejectionReason.DERIVATIVE_TOO_HIGH

        # Track closest composite regardless
        if entry_rms < best_rms:
            best_rms = entry_rms
            best_max = entry_max
            best_deriv = entry_deriv
            best_cos = entry_cos
            best_comp = comp
            best_comp_idx = comp_idx
            rejection = reject
            rms_val, max_val, deriv_val, cos_val = entry_rms, entry_max, entry_deriv, entry_cos

    # --- ASSIGN OR REJECT (but ALWAYS RECORD composite) ---
    if rejection is None and best_comp is not None:
        comp = best_comp
        comp.assigned_indices.append(entry_idx)
        comp.deltas.append(best_rms)
        comp.max_deltas.append(best_max)
        comp.derivative_deltas.append(best_deriv)
        comp.cosine_similarities.append(best_cos)

        if comp.worst_rms_index is None or best_rms > comp.deltas[comp.worst_rms_index]:
            comp.worst_rms_index = len(comp.deltas) - 1
        if comp.worst_max_index is None or best_max > comp.max_deltas[comp.worst_max_index]:
            comp.worst_max_index = len(comp.max_deltas) - 1

        return AssignmentRecord(
            entry_index=entry_idx,
            assigned_composite=best_comp_idx,
            rejected=False,
            rms_value=best_rms,
            max_value=best_max,
            derivative_value=best_deriv,
            cosine_value=best_cos
        )

    # --- REJECTED, BUT COMPOSITE ATTRIBUTED ---
    return AssignmentRecord(
        entry_index=entry_idx,
        assigned_composite=best_comp_idx,  # <<< FIX
        rejected=True,
        rejection_reason=rejection,
        rms_value=rms_val,
        max_value=max_val,
        derivative_value=deriv_val,
        cosine_value=cos_val
    )


# ------------------------------
# Update composites
# ------------------------------
def update_composite_shapes(catalogue: np.ndarray, composites: list[BEQComposite]) -> None:
    for comp in composites:
        if comp.assigned_indices:
            assigned = catalogue[comp.assigned_indices, :]
            comp.shape = np.median(assigned, axis=0)


# ------------------------------
# Compute non-overlapping fan curves
# ------------------------------
def compute_fan_curves(catalogue: np.ndarray, composites: list[BEQComposite],
                       fan_counts: tuple[int, ...] = (5,)) -> None:
    """
    For each composite, compute non-overlapping fan levels of assigned curves.
    Each fan level contains curves not in previous levels.
    """
    for comp in composites:
        comp.fan_envelopes = []
        if comp.assigned_indices:
            assigned = catalogue[comp.assigned_indices, :]
            # Rank by RMS distance to the composite
            rms_dists = np.array([rms(c - comp.shape) for c in assigned])
            sorted_idx = np.argsort(rms_dists)

            previous = 0
            for n in fan_counts:
                n_curves = min(n, len(sorted_idx))
                if n_curves > previous:
                    comp.fan_envelopes.append(assigned[sorted_idx[previous:n_curves], :])
                    previous = n_curves
                else:
                    # If not enough new curves, append empty array
                    comp.fan_envelopes.append(np.empty((0, assigned.shape[1])))
        else:
            for _ in fan_counts:
                comp.fan_envelopes.append(np.array([comp.shape]))


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
        n_prototypes: int = 50
) -> BEQCompositePipelineResult:
    band_mask: np.ndarray = (freqs >= band[0]) & (freqs <= band[1])
    catalogue: np.ndarray = responses_db[:, band_mask]
    band_weights: np.ndarray | None = weights[band_mask] if weights is not None else None

    # Step 1: k-medoids prototypes
    if catalogue.shape[0] <= n_prototypes:
        prototypes: np.ndarray = catalogue.copy()
    else:
        medoid_indices: np.ndarray = k_medoids(catalogue, n_prototypes, max_iter=100, random_state=0)
        prototypes = catalogue[medoid_indices]

    # Step 2: Ward clustering
    linkage_matrix: np.ndarray = linkage(prototypes, method='ward')
    labels: np.ndarray = fcluster(linkage_matrix, t=k, criterion='maxclust')

    # Step 3: median per cluster → composites
    composite_shapes: list[np.ndarray] = []
    for i in range(1, k + 1):
        cluster_curves: np.ndarray = prototypes[labels == i]
        median_shape: np.ndarray = np.median(cluster_curves, axis=0)
        composite_shapes.append(median_shape)
    composites: list[BEQComposite] = [BEQComposite(shape=c) for c in composite_shapes]

    # Step 4: assign all entries
    assignment_table: list[AssignmentRecord] = []
    for i, entry in enumerate(catalogue):
        record: AssignmentRecord = assign_to_composites_with_record(entry, composites,
                                                                    rms_limit, max_limit,
                                                                    cosine_limit, derivative_limit,
                                                                    i, weights=band_weights)
        assignment_table.append(record)

    # Step 5: recompute median composite shapes
    update_composite_shapes(catalogue, composites)

    # Step 6: compute fans
    compute_fan_curves(catalogue, composites, fan_counts)

    return BEQCompositePipelineResult(
        composites=composites,
        assignment_table={a.entry_index: a for a in assignment_table},
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit
    )

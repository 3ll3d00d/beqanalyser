import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import requests
from scipy.signal import freqz, sosfilt, unit_impulse

from beqanalyser import BEQFilter, CatalogueEntry, ComplexFilter, DistanceParams

logger = logging.getLogger()


def convert(entry: CatalogueEntry, fs=1000) -> BEQFilter | None:
    u_i = unit_impulse(fs * 4, "mid") * 23453.66
    f = ComplexFilter(
        fs=fs, filters=entry.iir_filters(fs=fs), description=f"{entry.digest}"
    )
    try:
        filtered = sosfilt(f.get_sos(), u_i)
        w, h = freqz(filtered, worN=1 << (int(fs / 2) - 1).bit_length())
        x = w * fs * 1.0 / (2 * np.pi)
        h[h == 0] = 0.000000001
        return BEQFilter(x, 20 * np.log10(abs(h)), entry)
    except Exception:
        logger.exception(f"Unable to process entry {entry.title}")
        return None


class CatalogueEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, CatalogueEntry):
            return obj.for_search
        return super().default(obj)


class CatalogueDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(dct):
        if "mag_freqs" in dct and "mag_db" in dct and "entry" in dct:
            return BEQFilter(
                np.array(dct["mag_freqs"]),
                np.array(dct["mag_db"]),
                CatalogueEntry("0", dct["entry"]),
            )
        return dct


def load() -> tuple[list[BEQFilter], str]:
    """
    Loads a list of BEQFilter objects and an associated hash, either from a local
    binary file or through a GitHub repository if the local file is unavailable
    or raises an exception during loading. The method ensures data integrity by
    maintaining and validating a SHA-256 hash of the data.

    :return:
        A tuple containing the list of BEQFilter objects and its corresponding
        SHA-256 hash value as a string.
    :rtype: tuple[list[BEQFilter], str]

    :raises FileNotFoundError:
        If the file `database.bin` is not found during the first load attempt.

    :raises requests.exceptions.HTTPError:
        If an HTTP error occurs when attempting to fetch the database from GitHub.
    """
    a = time.time()

    try:
        with open("database.bin", "r") as f:
            content = f.read()
            data: list[BEQFilter] = json.loads(content, cls=CatalogueDecoder)["data"]
            with open("database.bin.sha256", "r") as h:
                data_hash = h.read()
                import hashlib

                actual_hash = hashlib.sha256(content.encode()).hexdigest()
                assert data_hash == actual_hash, (
                    f"Data hash mismatch {data_hash} != {actual_hash}"
                )
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.exception(
                "Unable to load catalogue from database.bin, trying github"
            )
        try:
            logger.info("Loading catalogue from github")
            r = requests.get(
                "https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json",
                allow_redirects=True,
            )
            r.raise_for_status()
            with ProcessPoolExecutor() as executor:
                data: list[BEQFilter] = list(
                    executor.map(
                        convert,
                        (
                            CatalogueEntry(f"{idx}", e)
                            for idx, e in enumerate(json.loads(r.content))
                            if e.get("filters", [])
                        ),
                    )
                )
            with open("database.bin", "w") as f:
                output = json.dumps({"data": data}, cls=CatalogueEncoder)
                f.write(output)
                with open("database.bin.sha256", "w") as h:
                    import hashlib

                    data_hash = hashlib.sha256(output.encode()).hexdigest()
                    h.write(data_hash)
        except requests.exceptions.HTTPError as e:
            logger.exception("Unable to load catalogue from database")
            raise e

    b = time.time()
    logger.info(f"Loaded catalogue in {b - a:.3g}s")

    return data, data_hash


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

    logger.debug(
        f"Computing distance matrix for {n_samples} samples using {n_cores} cores..."
    )
    logger.debug(
        f"Base weights: RMS={rms_weight}, Cosine={cosine_weight} (scale={cosine_scale})"
    )
    logger.debug(
        f"RMS undershoot tolerance: {rms_undershoot_tolerance}, close threshold: {rms_close_threshold}"
    )
    logger.debug(f"Cosine boost in close range: {cosine_boost_in_close_range}x")

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
        logger.debug(
            f"Hard limits: RMS={rms_limit}, Cosine={cosine_limit}, Max={max_limit}, Deriv={derivative_limit}"
        )
        logger.debug(
            f"Soft limits: RMS={soft_limits['rms']}, Cosine={soft_limits['cosine']}, "
            f"Max={soft_limits['max']}, Deriv={soft_limits['derivative']}"
        )
        logger.debug(f"Penalties: Soft={soft_penalty_scale}, Hard={penalty_scale}")

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
    logger.info(
        f"Computing base distances on {n_cores} cores in {len(chunk_args)} chunks for {n_samples} samples"
    )
    a = time.time()
    if n_cores > 1:
        with Pool(processes=min(n_cores, len(chunk_args))) as pool:
            results = pool.map(compute_distance_chunk, chunk_args)
    else:
        results = [compute_distance_chunk(args) for args in chunk_args]
    b = time.time()
    logger.info(f"Computed base distances in {b - a:.3f}s")

    temp_params = DistanceParams(
        rms_limit=rms_limit,
        cosine_limit=cosine_limit,
        max_limit=max_limit,
        derivative_limit=derivative_limit,
        distance_rms_weight=rms_weight,
        distance_cosine_weight=cosine_weight,
        distance_cosine_scale=cosine_scale,
        distance_penalty_scale=penalty_scale,
        distance_rms_undershoot_tolerance=rms_undershoot_tolerance,
        distance_rms_close_threshold=rms_close_threshold,
        distance_cosine_boost_in_close_range=cosine_boost_in_close_range,
        distance_soft_limit_factor=soft_limit_factor,
        distance_soft_penalty_scale=soft_penalty_scale,
        use_constraints=True
        if (rms_limit or cosine_limit or max_limit or derivative_limit)
        else False,
    )

    # Assemble results using common distance computation logic
    logger.debug("Assembling distance matrix with sophisticated penalties...")

    for i, end_i, rms_row, cos_row, max_row, deriv_row in results:
        # Compute mean difference for undershoot detection
        mean_diff = np.mean(X_float32[i:end_i, None, :] - X_float32[None, :, :], axis=2)

        # Use common distance computation logic
        # Note: cos_row contains cosine SIMILARITY (not distance)
        distance_matrix[i:end_i, :] = compute_distance_components(
            rms_vals=rms_row,
            cos_sim_vals=cos_row,  # cos_row already contains similarity
            max_vals=max_row,
            deriv_vals=deriv_row,
            mean_diff_vals=mean_diff,
            params=temp_params,
        ).astype(np.float64)

        logger.debug(f"  Processed {end_i}/{n_samples} rows")

    c = time.time()
    logger.info(f"Computed distance final in {c - b:.3f}s")

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
        f"Distance matrix computed in {c - a:.3f}s. Range: [{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]"
    )
    return distance_matrix


# ------------------------------
# Precompute distance matrix for BEQ curves
# ------------------------------
def compute_distance_components(
    rms_vals: np.ndarray | float,
    cos_sim_vals: np.ndarray | float,
    max_vals: np.ndarray | float,
    deriv_vals: np.ndarray | float | None,
    mean_diff_vals: np.ndarray | float,
    params: DistanceParams,
) -> np.ndarray | float:
    """
    Core distance computation logic shared between matrix and single-pair calculations.

    Computes sophisticated distance scores with:
    - Asymmetric RMS handling (tolerant to undershoot)
    - Adaptive cosine weighting (boosted when RMS is close)
    - Tiered penalty system (soft/hard limits)

    Args:
        rms_vals: RMS distance values (can be scalar or array)
        cos_sim_vals: Cosine similarity values (can be scalar or array)
        max_vals: Maximum absolute deviation values (can be scalar or array)
        deriv_vals: Derivative RMS values (can be scalar or array, or None)
        mean_diff_vals: Mean difference values for undershoot detection (can be scalar or array)
        params: Parameters controlling distance computation

    Returns:
        Distance scores (same shape as input arrays)
    """
    # Convert cosine similarity to distance
    cos_dist = 1.0 - cos_sim_vals

    # 1. Asymmetric RMS handling
    is_undershoot = mean_diff_vals < 0
    rms_adjusted = rms_vals

    if params.distance_rms_undershoot_tolerance > 1.0:
        undershoot_factor = 1.0 / params.distance_rms_undershoot_tolerance
        if isinstance(rms_vals, np.ndarray):
            rms_adjusted = np.where(
                is_undershoot, rms_vals * undershoot_factor, rms_vals
            )
        elif is_undershoot:
            rms_adjusted = rms_vals * undershoot_factor

    # 2. Detect "close" RMS range and boost cosine weight
    is_close = rms_vals < params.distance_rms_close_threshold
    if isinstance(is_close, np.ndarray):
        cosine_weight_adjusted = np.where(
            is_close,
            params.distance_cosine_weight * params.distance_cosine_boost_in_close_range,
            params.distance_cosine_weight,
        )
    else:
        cosine_weight_adjusted = (
            params.distance_cosine_weight * params.distance_cosine_boost_in_close_range
            if is_close
            else params.distance_cosine_weight
        )

    # Compute base distance with adaptive weighting
    base_distance = (
        params.distance_rms_weight * rms_adjusted
        + cosine_weight_adjusted * cos_dist * params.distance_cosine_scale
    )

    # 3. Tiered penalty system
    if isinstance(base_distance, np.ndarray):
        penalty = np.zeros_like(base_distance, dtype=np.float32)
    else:
        penalty = 0.0

    if params.use_constraints:
        # RMS penalties
        if params.rms_limit is not None:
            soft_rms = params.rms_limit * params.distance_soft_limit_factor

            # Soft penalty zone
            soft_violations = (rms_vals > soft_rms) & (rms_vals <= params.rms_limit)
            if isinstance(soft_violations, np.ndarray):
                penalty = (
                    penalty
                    + soft_violations.astype(np.float32)
                    * params.distance_soft_penalty_scale
                )
            elif soft_violations:
                penalty = penalty + params.distance_soft_penalty_scale

            # Hard penalty zone with asymmetric handling
            hard_violations_overshoot = (rms_vals > params.rms_limit) & ~is_undershoot
            if isinstance(hard_violations_overshoot, np.ndarray):
                penalty = (
                    penalty
                    + hard_violations_overshoot.astype(np.float32)
                    * params.distance_penalty_scale
                )
            elif hard_violations_overshoot:
                penalty = penalty + params.distance_penalty_scale

            # Reduced penalty for undershoot violations
            hard_violations_undershoot = (rms_vals > params.rms_limit) & is_undershoot
            reduced_penalty = (
                params.distance_penalty_scale / params.distance_rms_undershoot_tolerance
            )
            if isinstance(hard_violations_undershoot, np.ndarray):
                penalty = (
                    penalty
                    + hard_violations_undershoot.astype(np.float32) * reduced_penalty
                )
            elif hard_violations_undershoot:
                penalty = penalty + reduced_penalty

        # Cosine penalties
        if params.cosine_limit is not None:
            soft_cos = params.cosine_limit + (1.0 - params.cosine_limit) * (
                1.0 - params.distance_soft_limit_factor
            )

            # Soft penalty zone
            soft_violations = (cos_sim_vals < soft_cos) & (
                cos_sim_vals >= params.cosine_limit
            )
            if isinstance(soft_violations, np.ndarray):
                penalty = (
                    penalty
                    + soft_violations.astype(np.float32)
                    * params.distance_soft_penalty_scale
                )
            elif soft_violations:
                penalty = penalty + params.distance_soft_penalty_scale

            # Hard penalty zone
            hard_violations = cos_sim_vals < params.cosine_limit
            if isinstance(hard_violations, np.ndarray):
                penalty = (
                    penalty
                    + hard_violations.astype(np.float32) * params.distance_penalty_scale
                )
            elif hard_violations:
                penalty = penalty + params.distance_penalty_scale

        # Max deviation penalties
        if params.max_limit is not None:
            soft_max = params.max_limit * params.distance_soft_limit_factor

            # Soft penalty zone
            soft_violations = (max_vals > soft_max) & (max_vals <= params.max_limit)
            if isinstance(soft_violations, np.ndarray):
                penalty = (
                    penalty
                    + soft_violations.astype(np.float32)
                    * params.distance_soft_penalty_scale
                )
            elif soft_violations:
                penalty = penalty + params.distance_soft_penalty_scale

            # Hard penalty zone
            hard_violations = max_vals > params.max_limit
            if isinstance(hard_violations, np.ndarray):
                penalty = (
                    penalty
                    + hard_violations.astype(np.float32) * params.distance_penalty_scale
                )
            elif hard_violations:
                penalty = penalty + params.distance_penalty_scale

        # Derivative penalties
        if params.derivative_limit is not None and deriv_vals is not None:
            soft_deriv = params.derivative_limit * params.distance_soft_limit_factor

            # Soft penalty zone
            soft_violations = (deriv_vals > soft_deriv) & (
                deriv_vals <= params.derivative_limit
            )
            if isinstance(soft_violations, np.ndarray):
                penalty = (
                    penalty
                    + soft_violations.astype(np.float32)
                    * params.distance_soft_penalty_scale
                )
            elif soft_violations:
                penalty = penalty + params.distance_soft_penalty_scale

            # Hard penalty zone
            hard_violations = deriv_vals > params.derivative_limit
            if isinstance(hard_violations, np.ndarray):
                penalty = (
                    penalty
                    + hard_violations.astype(np.float32) * params.distance_penalty_scale
                )
            elif hard_violations:
                penalty = penalty + params.distance_penalty_scale

    # Combine base distance with penalties
    return base_distance + penalty


def compute_distance_matrix(
    input_curves: np.ndarray,
    freqs: np.ndarray,
    distance_params: DistanceParams,
    band: tuple[float, float] = (5, 50),
) -> np.ndarray:
    """
    Precompute the full distance matrix once.
    :param input_curves:the catalogue of responses.
    :param freqs: the frequency array.
    :param distance_params: the distance parameters.
    :param band: the frequency band to analyze.
    :return: the matrix.
    """
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    full_masked_catalogue = input_curves[:, band_mask]

    logger.info("=" * 80)
    logger.info("PRECOMPUTING FULL DISTANCE MATRIX (will be reused across iterations)")
    logger.info("=" * 80)

    start_time = time.time()

    full_distance_matrix = compute_beq_distance_matrix(
        full_masked_catalogue,
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

    elapsed = time.time() - start_time
    logger.info(f"Full distance matrix computed in {elapsed:.3f}s")
    logger.info(
        f"Matrix shape: {full_distance_matrix.shape}, size: {full_distance_matrix.nbytes / 1024 / 1024:.2f} MB"
    )

    return full_distance_matrix


def compute_distance_chunk(args):
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
    logger.debug(f"Computed chunk {i}-{end_i} in {b - a:.3f}s")
    return i, end_i, rms_row, cos_row, max_row, deriv_row

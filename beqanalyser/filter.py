"""
Generate RBJ-format biquad IIR filters from BEQ composite frequency response curves.

This module fits composite curves (typically low-shelf-like responses) with
cascaded biquad filters using an analytical approach optimized for bass EQ profiles.
"""

import logging
import math
import time
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import optimize, signal

from beqanalyser import BEQComposite, BiquadCoefficients

logger = logging.getLogger(__name__)


def q_from_octave_bandwidth(bw: float = 1.0 / 3.0) -> float:
    """
    Convert octave bandwidth (measured between -3 dB points) to equivalent peaking filter Q.
    """
    return 1.0 / (2 ** (bw / 2) - 2 ** (-bw / 2))


@dataclass
class BiquadFilter:
    """Complete biquad filter with parameters and coefficients."""

    filter_type: str  # 'lowshelf', 'peaking', 'highshelf'
    fc: float = field(metadata={"fmt": "{:.1f} Hz"})  # Center/corner frequency in Hz
    gain_db: float = field(metadata={"fmt": "{:.2f} dB"})  # Gain in dB
    q: float = field(metadata={"fmt": "{:.3f}"})  # Q factor
    fs: float = 48000  # Sample rate in Hz
    coefficients: BiquadCoefficients | None = None

    def __post_init__(self):
        if self.fs and not self.coefficients:
            if self.filter_type == "lowshelf":
                self.coefficients = lowshelf_rbj(self.fc, self.gain_db, self.q, self.fs)
            elif self.filter_type == "highshelf":
                self.coefficients = highshelf_rbj(
                    self.fc, self.gain_db, self.q, self.fs
                )
            elif self.filter_type == "peaking":
                self.coefficients = peaking_rbj(self.fc, self.gain_db, self.q, self.fs)

    def __repr__(self) -> str:
        return (
            f"BiquadFilter({self.filter_type}, fc={self.fc:.1f}Hz, "
            f"gain={self.gain_db:.2f}dB, Q={self.q:.3f})"
        )

    def as_row(self):
        row = []
        for f in fields(self):
            if "fmt" in f.metadata or f.name == "filter_type":
                value = getattr(self, f.name)
                fmt = f.metadata.get("fmt", "{}")
                row.append(fmt.format(value))
        return row

    @classmethod
    def column_labels(cls):
        return [
            f.name
            for f in fields(cls)
            if "fmt" in f.metadata or f.name == "filter_type"
        ]


@dataclass
class GraphicEQBand:
    """A single band in a graphic equaliser."""

    fc: float = field(metadata={"fmt": "{:.1f} Hz"})  # Centre frequency in Hz
    gain_db: float = field(metadata={"fmt": "{:.2f} dB"})  # Gain in dB
    width: float = 1.0 / 3.0

    def __repr__(self) -> str:
        return f"GraphicEQ(fc={self.fc:.1f}Hz, gain={self.gain_db:.2f}dB"

    def as_biquad(self, fs: float) -> BiquadFilter:
        return BiquadFilter(
            filter_type="peaking",
            fc=self.fc,
            gain_db=self.gain_db,
            q=q_from_octave_bandwidth(self.width),
            fs=fs,
        )


def lowshelf_rbj(fc: float, gain_db: float, q: float, fs: float) -> BiquadCoefficients:
    """
    Generate RBJ low-shelf biquad coefficients.

    Args:
        fc: Corner frequency in Hz
        gain_db: Gain in dB (positive for boost, negative for cut)
        q: Q factor (typically 0.5 to 1.0 for shelf filters)
        fs: Sample rate in Hz

    Returns:
        BiquadCoefficients in RBJ format
    """
    A = 10 ** (gain_db / 40)  # sqrt of linear gain
    w0 = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    # RBJ low-shelf formulas
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    return BiquadCoefficients(b0, b1, b2, a0, a1, a2)


def peaking_rbj(fc: float, gain_db: float, q: float, fs: float) -> BiquadCoefficients:
    """
    Generate RBJ peaking EQ biquad coefficients.

    Args:
        fc: Center frequency in Hz
        gain_db: Gain in dB
        q: Q factor (bandwidth)
        fs: Sample rate in Hz

    Returns:
        BiquadCoefficients in RBJ format
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    # RBJ peaking formulas
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    return BiquadCoefficients(b0, b1, b2, a0, a1, a2)


def highshelf_rbj(fc: float, gain_db: float, q: float, fs: float) -> BiquadCoefficients:
    """
    Generate RBJ high-shelf biquad coefficients.

    Args:
        fc: Corner frequency in Hz
        gain_db: Gain in dB
        q: Q factor
        fs: Sample rate in Hz

    Returns:
        BiquadCoefficients in RBJ format
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    # RBJ high-shelf formulas
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    return BiquadCoefficients(b0, b1, b2, a0, a1, a2)


def compute_filter_response(
    filters: list[BiquadFilter], freqs: np.ndarray, fs: float
) -> np.ndarray:
    """
    Compute the magnitude response of cascaded biquad filters.

    Args:
        filters: List of BiquadFilter objects
        freqs: Frequency points to evaluate (Hz)
        fs: Sample rate in Hz

    Returns:
        Magnitude response in dB
    """
    if not filters:
        return np.zeros_like(freqs)

    # Stack all filter sections
    sos = np.vstack([f.coefficients.to_sos() for f in filters])

    # Compute frequency response
    w = 2 * np.pi * freqs / fs
    _, h = signal.sosfreqz(sos, worN=w)

    return 20 * np.log10(np.abs(h) + 1e-12)


def analyze_curve_characteristics(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    low_freq: float = 10.0,
    high_freq: float = 120.0,
) -> dict:
    """
    Analyze characteristics of a composite curve to guide filter design.

    Args:
        mag_db: Magnitude response in dB
        freqs: Frequency array in Hz
        low_freq: Low frequency reference point
        high_freq: High frequency reference point

    Returns:
        Dictionary with curve characteristics
    """
    # Find indices for analysis points
    low_idx = np.argmin(np.abs(freqs - low_freq))
    high_idx = np.argmin(np.abs(freqs - high_freq))

    # Estimate total gain change (shelf-like behavior)
    total_gain = mag_db[high_idx] - mag_db[low_idx]

    # Find approximate transition frequency (where gain is halfway)
    target_gain = mag_db[low_idx] + total_gain / 2
    transition_idx = np.argmin(np.abs(mag_db[low_idx:high_idx] - target_gain))
    transition_freq = freqs[low_idx + transition_idx]

    # Estimate slope steepness at transition
    slope = np.gradient(mag_db, np.log10(freqs))
    max_slope_idx = np.argmax(np.abs(slope[low_idx:high_idx])) + low_idx
    max_slope = slope[max_slope_idx]

    return {
        "total_gain": total_gain,
        "transition_freq": transition_freq,
        "max_slope": max_slope,
        "low_level": mag_db[low_idx],
        "high_level": mag_db[high_idx],
        "is_boost": total_gain > 0,
    }


def fit_single_lowshelf(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    fs: float,
    initial_params: tuple[float, float, float] | None = None,
) -> BiquadFilter:
    """
    Fit a single low-shelf filter using robust optimization.

    Args:
        mag_db: Target magnitude response in dB
        freqs: Frequency array in Hz
        fs: Sample rate in Hz
        initial_params: Optional (fc, gain, q) initial guess

    Returns:
        BiquadFilter with optimal parameters
    """
    # Analyze curve if no initial params provided
    if initial_params is None:
        char = analyze_curve_characteristics(mag_db, freqs)
        fc_init = char["transition_freq"]
        gain_init = char["total_gain"]
        # Start with neutral Q, let optimizer find best value
        q_init = 0.707
    else:
        fc_init, gain_init, q_init = initial_params

    def error_function(params):
        fc, gain, q = params

        # Bounds checking with penalties
        if fc < 5 or fc > 300:
            return 1e10
        if q < 0.2 or q > 5.0:
            return 1e10
        if abs(gain) > 30:
            return 1e10

        try:
            # Generate filter response
            filt = BiquadFilter("lowshelf", fc, gain, q, fs)
            response = compute_filter_response([filt], freqs, fs)

            # Weighted error - prioritize the transition region
            error = response - mag_db

            # Create frequency-dependent weights
            weights = np.ones_like(freqs)
            # Weight transition region (fc/2 to fc*2) more heavily
            transition_mask = (freqs >= fc / 2) & (freqs <= fc * 2)
            weights[transition_mask] = 3.0
            # Also weight low frequencies reasonably
            weights[freqs < fc / 2] = 1.5

            # RMS error with weights
            weighted_error = np.sqrt(np.mean(weights * error**2))

            # Add penalty for large parameter values (regularization)
            param_penalty = 0.0001 * (abs(fc - fc_init) + abs(q - q_init))

            return weighted_error + param_penalty
        except:
            return 1e10

    # Try multiple optimization strategies
    best_result = None
    best_error = float("inf")

    # Strategy 1: Nelder-Mead from initial guess
    try:
        result = optimize.minimize(
            error_function,
            x0=[fc_init, gain_init, q_init],
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 0.01, "fatol": 0.0001},
        )
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
    except:
        pass

    # Strategy 2: Powell from initial guess
    try:
        result = optimize.minimize(
            error_function,
            x0=[fc_init, gain_init, q_init],
            method="Powell",
            options={"maxiter": 2000},
        )
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
    except:
        pass

    # Strategy 3: Try with different Q values
    for q_try in [0.5, 0.707, 1.0, 1.5]:
        try:
            result = optimize.minimize(
                error_function,
                x0=[fc_init, gain_init, q_try],
                method="Nelder-Mead",
                options={"maxiter": 1000},
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        except:
            pass

    if best_result is None:
        # Fallback: use initial parameters
        logger.warning("Optimization failed, using initial parameters")
        fc_opt, gain_opt, q_opt = fc_init, gain_init, q_init
    else:
        fc_opt, gain_opt, q_opt = best_result.x

    # Clamp to reasonable ranges
    fc_opt = np.clip(fc_opt, 5, 300)
    gain_opt = np.clip(gain_opt, -30, 30)
    q_opt = np.clip(q_opt, 0.2, 5.0)

    return BiquadFilter("lowshelf", fc_opt, gain_opt, q_opt, fs)


def fit_peaking_to_residual(
    residual: np.ndarray, freqs: np.ndarray, fs: float
) -> BiquadFilter | None:
    """
    Fit a peaking filter to residual ripple/resonance.

    Args:
        residual: Residual magnitude response in dB
        freqs: Frequency array in Hz
        fs: Sample rate in Hz

    Returns:
        BiquadFilter or None if no significant peak found
    """
    # Find peaks in residual
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(np.abs(residual), prominence=0.3)

    if len(peaks) == 0:
        return None

    # Find most prominent peak
    peak_idx = peaks[np.argmax(properties["prominences"])]
    fc_init = freqs[peak_idx]
    gain_init = residual[peak_idx]

    # Only fit if peak is significant
    if abs(gain_init) < 0.5:
        return None

    def error_function(params):
        fc, gain, q = params

        if fc < 5 or fc > 300 or q < 0.5 or q > 20.0:
            return 1e10

        try:
            filt = BiquadFilter("peaking", fc, gain, q, fs)
            response = compute_filter_response([filt], freqs, fs)

            # Focus on region around peak
            mask = (freqs >= fc / 2) & (freqs <= fc * 2)
            error = np.sqrt(np.mean((response[mask] - residual[mask]) ** 2))
            return error
        except:
            return 1e10

    result = optimize.minimize(
        error_function,
        x0=[fc_init, gain_init, 2.0],
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )

    fc_opt, gain_opt, q_opt = result.x
    return BiquadFilter("peaking", fc_opt, gain_opt, q_opt, fs)


def fit_composite_to_peq(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    fs: float = 48000.0,
    max_filters: int = 4,
    residual_threshold: float = 0.3,
) -> list[BiquadFilter]:
    """
    Fit a composite curve with cascaded biquad filters using iterative residual fitting.

    Args:
        mag_db: Target magnitude response in dB
        freqs: Frequency array in Hz
        fs: Sample rate in Hz
        max_filters: Maximum number of biquad filters to use
        residual_threshold: RMS residual threshold to stop adding filters (dB)

    Returns:
        List of BiquadFilter objects
    """
    filters = []
    residual = mag_db.copy()

    logger.info(f"Fitting composite curve with up to {max_filters} filters")
    initial_rms = np.sqrt(np.mean(mag_db**2))
    logger.info(f"  Initial RMS: {initial_rms:.3f} dB")

    for i in range(max_filters):
        # Analyze current residual
        characteristics = analyze_curve_characteristics(residual, freqs)

        # Decide filter type based on residual characteristics
        if i == 0 or abs(characteristics["total_gain"]) > 1.0:
            # Primary shelf filter
            logger.info(
                f"  Filter {i + 1}: Fitting lowshelf, gain≈{characteristics['total_gain']:.2f}dB, "
                f"fc≈{characteristics['transition_freq']:.1f}Hz"
            )

            new_filter = fit_single_lowshelf(residual, freqs, fs)
            filters.append(new_filter)

        else:
            # Try fitting peaking filter for ripple correction
            logger.info(
                f"  Filter {i + 1}: Attempting peaking filter for residual correction"
            )
            new_filter = fit_peaking_to_residual(residual, freqs, fs)

            if new_filter is None:
                logger.info(f"    No significant peak found, stopping")
                break

            filters.append(new_filter)

        # Compute new residual
        filter_response = compute_filter_response([new_filter], freqs, fs)
        residual = residual - filter_response

        # Check convergence
        rms_residual = np.sqrt(np.mean(residual**2))
        max_residual = np.max(np.abs(residual))
        logger.info(f"    Fitted: {new_filter}")
        logger.info(
            f"    RMS residual: {rms_residual:.3f} dB, Max: {max_residual:.3f} dB"
        )

        if rms_residual < residual_threshold:
            logger.info(f"  Converged after {i + 1} filters")
            break

    # Final check: optimize all filters together
    if len(filters) > 1:
        logger.info(f"  Running global optimization on all {len(filters)} filters...")
        filters = optimize_cascade_globally(filters, mag_db, freqs, fs)

        final_response = compute_filter_response(filters, freqs, fs)
        final_rms = np.sqrt(np.mean((final_response - mag_db) ** 2))
        logger.info(f"  Final RMS after global optimization: {final_rms:.3f} dB")

    return filters


def optimize_cascade_globally(
    filters: list[BiquadFilter], target: np.ndarray, freqs: np.ndarray, fs: float
) -> list[BiquadFilter]:
    """
    Optimize all filter parameters simultaneously for best overall fit.

    Args:
        filters: Initial filter cascade
        target: Target magnitude response in dB
        freqs: Frequency array in Hz
        fs: Sample rate in Hz

    Returns:
        Optimized list of BiquadFilter objects
    """
    # Pack all parameters into single vector
    x0 = []
    filter_types = []
    for f in filters:
        x0.extend([f.fc, f.gain_db, f.q])
        filter_types.append(f.filter_type)

    def error_function(params):
        # Unpack parameters
        filters_temp = []
        for i in range(len(filter_types)):
            fc, gain, q = params[i * 3 : (i + 1) * 3]

            # Bounds checking
            if fc < 5 or fc > 300 or q < 0.2 or q > 5.0 or abs(gain) > 30:
                return 1e10

            try:
                filters_temp.append(BiquadFilter(filter_types[i], fc, gain, q, fs))
            except:
                return 1e10

        # Compute cascade response
        response = compute_filter_response(filters_temp, freqs, fs)

        # RMS error
        return np.sqrt(np.mean((response - target) ** 2))

    # Optimize
    result = optimize.minimize(
        error_function,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 0.01},
    )

    # Rebuild filters from optimised parameters
    optimized_filters = []
    for i in range(len(filter_types)):
        fc, gain, q = result.x[i * 3 : (i + 1) * 3]
        fc = np.round(np.clip(fc, 5, 300), 1)
        gain = np.round(np.clip(gain, -30, 30), 2)
        q = np.round(np.clip(q, 0.2, 8.0), 3)

        if math.fabs(gain) < 0.1:
            continue

        optimized_filters.append(BiquadFilter(filter_types[i], fc, gain, q, fs))

    return optimized_filters


def fit_all_composites_to_peq(
    composites: list[BEQComposite],
    freqs: np.ndarray,
    fs: float = 48000.0,
    max_filters: int = 4,
    residual_threshold: float = 0.3,
) -> dict:
    """
    Fit biquad filters to all composite curves from BEQ analysis.

    Args:
        composites: List of BEQComposite objects from analyser.py
        freqs: Frequency array in Hz (should match the band used in analysis)
        fs: Sample rate in Hz
        max_filters: Maximum filters per composite
        residual_threshold: RMS residual threshold (dB)

    Returns:
        Dictionary mapping composite_id to list of BiquadFilter objects
    """
    results = {}

    for comp in composites:
        logger.info(
            f"Fitting composite {comp.id} ({len(comp.assigned_entry_ids)} entries)"
        )

        # Fit filters to this composite
        filters: list[BiquadFilter] = fit_composite_to_peq(
            comp.mag_response, freqs, fs, max_filters, residual_threshold
        )

        # Compute final fit quality
        fitted_response: np.ndarray = compute_filter_response(filters, freqs, fs)
        rms_error = float(np.sqrt(np.mean((fitted_response - comp.mag_response) ** 2)))
        max_error = float(np.max(np.abs(fitted_response - comp.mag_response)))

        logger.info(
            f"  Final fit: {len(filters)} filters, "
            f"RMS error={rms_error:.3f}dB, max error={max_error:.3f}dB"
        )

        results[comp.id] = {
            "freqs": freqs,
            "filters": filters,
            "rms_error": rms_error,
            "max_error": max_error,
            "target_response": comp.mag_response,
            "fitted_response": fitted_response,
        }

    return results


def fit_all_composites_to_geq(
    composites: list[BEQComposite], freqs: np.ndarray, fs: float = 48000.0, smoothing: float = 1.0
) -> dict:
    """
    Fit graphic eq filters to all composite curves from BEQ analysis.

    Args:
        composites: List of BEQComposite objects from analyser.py
        freqs: Frequency array in Hz (should match the band used in analysis)
        fs: Sample rate in Hz

    Returns:
        Dictionary mapping composite_id to list of BiquadFilter objects
    """
    results = {}

    for comp in composites:
        logger.info(
            f"Fitting composite {comp.id} ({len(comp.assigned_entry_ids)} entries)"
        )

        # Fit filters to this composite
        filters: list[BiquadFilter] = fit_composite_to_geq(comp.mag_response, freqs, fs, smoothing=smoothing)

        # Compute final fit quality
        fitted_response: np.ndarray = compute_filter_response(filters, freqs, fs)
        rms_error = float(np.sqrt(np.mean((fitted_response - comp.mag_response) ** 2)))
        max_error = float(np.max(np.abs(fitted_response - comp.mag_response)))

        logger.info(
            f"  Final fit: {len(filters)} filters, "
            f"RMS error={rms_error:.3f}dB, max error={max_error:.3f}dB"
        )

        results[comp.id] = {
            "freqs": freqs,
            "filters": filters,
            "rms_error": rms_error,
            "max_error": max_error,
            "target_response": comp.mag_response,
            "fitted_response": fitted_response,
        }

    return results


def export_filters_to_text(results: dict, freqs: np.ndarray) -> str:
    """
    Export fitted filters to human-readable text format.

    Args:
        results: Dictionary from fit_all_composites
        freqs: Frequency array used for fitting

    Returns:
        Formatted string with filter parameters
    """
    output = []
    output.append("=" * 80)
    output.append("BEQ COMPOSITE BIQUAD FILTERS (RBJ Format)")
    output.append("=" * 80)
    output.append("")

    for comp_id, data in sorted(results.items()):
        output.append(f"Composite {comp_id}:")
        output.append(
            f"  Fit Quality: RMS error = {data['rms_error']:.3f} dB, "
            f"Max error = {data['max_error']:.3f} dB"
        )
        output.append(f"  Filters ({len(data['filters'])}):")

        for i, filt in enumerate(data["filters"], 1):
            output.append(f"    {i}. {filt}")
            coef = filt.coefficients.normalize()
            output.append(f"       b: [{coef.b0:.8f}, {coef.b1:.8f}, {coef.b2:.8f}]")
            output.append(f"       a: [1.0, {coef.a1:.8f}, {coef.a2:.8f}]")

        output.append("")

    return "\n".join(output)


def generate_third_octave_frequencies(
    start_freq: float = 5.0, end_freq: float = 200.0
) -> list[float]:
    """
    Generate standard 1/3 octave band centre frequencies.

    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz

    Returns:
        List of standard 1/3 octave frequencies
    """
    # Standard ISO 266 / ANSI S1.11 1/3 octave center frequencies
    standard_freqs = [
        5,
        6.3,
        8,
        10,
        12.5,
        16,
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
        16000,
        20000,
    ]
    return [f for f in standard_freqs if start_freq <= f <= end_freq]


def fit_composite_to_geq(
    target_db: np.ndarray,
    target_freqs: np.ndarray,
    fs: float = 48000.0,
    smoothing: float = 1.0,
) -> list[BiquadFilter]:
    """
    Fit a target curve using graphic EQ bands.

    Args:
        target_freqs: Frequency points of target curve (Hz)
        target_db: Magnitude response in dB
        fs: sample rate (Hz).
        smoothing: Regularization to encourage smooth gain transitions (0-1)

    Returns:
        list of BiquadFilter
    """
    gain_range = [np.min(target_db) - 5, np.max(target_db) + 5]
    log_freqs = np.logspace(
        np.log10(np.min(target_freqs) * 0.8), np.log10(np.max(target_freqs) * 1.2), 500
    )
    q = q_from_octave_bandwidth()
    band_freqs = generate_third_octave_frequencies(log_freqs[0], log_freqs[-1])

    # Interpolate target to our frequency grid
    target_interp = np.interp(log_freqs, target_freqs, target_db)

    # Simple initial guess: sample target curve at band frequencies
    x0 = np.interp(band_freqs, target_freqs, target_db)
    # Clip to gain range
    x0 = np.clip(x0, gain_range[0], gain_range[1])
    # Set up optimisation bounds
    bounds = [gain_range] * len(band_freqs)
    # Pre-compute values for faster optimisation
    w = 2 * np.pi * log_freqs / fs
    z_inv = np.exp(-1j * w)
    z_inv2 = z_inv**2

    # Pre-compute filter parameters for each band
    band_params = []
    for fc in band_freqs:
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)
        band_params.append((alpha, cos_w0))

    def objective(gains):
        """Optimization objective function."""
        response = np.ones(len(w), dtype=np.complex128)

        for i, gain in enumerate(gains):
            if abs(gain) < 0.01:  # Skip near-zero gains
                continue

            alpha, cos_w0 = band_params[i]
            A = 10 ** (gain / 40)

            # Compute coefficients
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

            # Normalize and compute response (vectorized)
            b0_n, b1_n, b2_n = b0 / a0, b1 / a0, b2 / a0
            a1_n, a2_n = a1 / a0, a2 / a0

            h = (b0_n + b1_n * z_inv + b2_n * z_inv2) / (
                1.0 + a1_n * z_inv + a2_n * z_inv2
            )

            response *= h

        response_db = 20 * np.log10(np.abs(response))

        # Primary error term
        fit_error = np.sum((response_db - target_interp) ** 2)

        # Optional smoothing penalty (penalise adjacent band differences)
        if smoothing > 0:
            gain_diffs = np.diff(gains)
            smoothness_term = np.sum(gain_diffs**2)
            return fit_error + smoothing * smoothness_term

        return fit_error

    # Optimize
    logger.info(f"Starting optimization with {len(band_freqs)} bands...")
    start_time = time.time()

    result = optimize.minimize(
        objective, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 1000}
    )

    elapsed = time.time() - start_time
    logger.info(f"Optimization completed in {elapsed:.3f}s")

    return [
        GraphicEQBand(freq, gain).as_biquad(fs)
        for freq, gain in zip(band_freqs, result.x)
    ]

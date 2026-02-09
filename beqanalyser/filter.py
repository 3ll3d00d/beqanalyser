"""
Generate RBJ-format biquad IIR filters from BEQ composite frequency response curves.

This module fits composite curves (typically low-shelf-like responses) with
cascaded biquad filters using an analytical approach optimized for bass EQ profiles.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal

from beqanalyser import BEQComposite, BiquadCoefficients

logger = logging.getLogger(__name__)


@dataclass
class BiquadFilter:
    """Complete biquad filter with parameters and coefficients."""

    filter_type: str  # 'lowshelf', 'peaking', 'highshelf'
    fc: float  # Center/corner frequency in Hz
    gain_db: float  # Gain in dB
    q: float  # Q factor
    fs: float  # Sample rate in Hz
    coefficients: BiquadCoefficients

    def __repr__(self) -> str:
        return (
            f"BiquadFilter({self.filter_type}, fc={self.fc:.1f}Hz, "
            f"gain={self.gain_db:.2f}dB, Q={self.q:.3f})"
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
            coef = lowshelf_rbj(fc, gain, q, fs)
            filt = BiquadFilter("lowshelf", fc, gain, q, fs, coef)
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

    coef = lowshelf_rbj(fc_opt, gain_opt, q_opt, fs)
    return BiquadFilter("lowshelf", fc_opt, gain_opt, q_opt, fs, coef)


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
            coef = peaking_rbj(fc, gain, q, fs)
            filt = BiquadFilter("peaking", fc, gain, q, fs, coef)
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
    coef = peaking_rbj(fc_opt, gain_opt, q_opt, fs)

    return BiquadFilter("peaking", fc_opt, gain_opt, q_opt, fs, coef)


def fit_composite_curve(
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
                if filter_types[i] == "lowshelf":
                    coef = lowshelf_rbj(fc, gain, q, fs)
                elif filter_types[i] == "peaking":
                    coef = peaking_rbj(fc, gain, q, fs)
                else:
                    coef = highshelf_rbj(fc, gain, q, fs)

                filters_temp.append(
                    BiquadFilter(filter_types[i], fc, gain, q, fs, coef)
                )
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

    # Rebuild filters from optimized parameters
    optimized_filters = []
    for i in range(len(filter_types)):
        fc, gain, q = result.x[i * 3 : (i + 1) * 3]
        fc = np.clip(fc, 5, 300)
        gain = np.clip(gain, -30, 30)
        q = np.clip(q, 0.2, 5.0)

        if filter_types[i] == "lowshelf":
            coef = lowshelf_rbj(fc, gain, q, fs)
        elif filter_types[i] == "peaking":
            coef = peaking_rbj(fc, gain, q, fs)
        else:
            coef = highshelf_rbj(fc, gain, q, fs)

        optimized_filters.append(BiquadFilter(filter_types[i], fc, gain, q, fs, coef))

    return optimized_filters


def fit_all_composites(
    composites: list[BEQComposite],  # List of BEQComposite objects
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
        filters: list[BiquadFilter] = fit_composite_curve(
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


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Example: Create synthetic composite curve for testing
    freqs = np.logspace(np.log10(5), np.log10(200), 300)

    # Simulate a typical BEQ composite (low-shelf boost around 25-30 Hz)
    # Using a more realistic bass boost curve
    synthetic_composite = np.zeros_like(freqs)

    # Main shelf component
    for f, i in zip(freqs, range(len(freqs))):
        if f < 25:
            synthetic_composite[i] = 8.0  # Boost below 25 Hz
        elif f < 50:
            # Smooth transition
            t = (f - 25) / 25
            synthetic_composite[i] = 8.0 * (1 - t)
        else:
            synthetic_composite[i] = 0.0

    # Add some ripple
    synthetic_composite += 0.3 * np.sin(2 * np.pi * np.log10(freqs / 15))

    print("Fitting synthetic composite curve...")
    print("=" * 80)
    filters = fit_composite_curve(synthetic_composite, freqs, fs=48000.0, max_filters=4)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("\nFitted filters:")
    for i, filt in enumerate(filters, 1):
        print(f"{i}. {filt}")
        coef = filt.coefficients.normalize()
        print(f"   b: [{coef.b0:.8f}, {coef.b1:.8f}, {coef.b2:.8f}]")
        print(f"   a: [1.0, {coef.a1:.8f}, {coef.a2:.8f}]")

    # Show results
    fitted = compute_filter_response(filters, freqs, 48000.0)
    rms_error = np.sqrt(np.mean((fitted - synthetic_composite) ** 2))
    max_error = np.max(np.abs(fitted - synthetic_composite))
    print(f"\nFinal RMS error: {rms_error:.3f} dB")
    print(f"Final max error: {max_error:.3f} dB")

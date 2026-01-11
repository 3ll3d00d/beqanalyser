"""
Generate RBJ format biquad IIR filters to match BEQ composite curves.

This module takes the composite curves produced by beqanalyser and fits
them with standard biquad IIR filters in Robert Bristow-Johnson format.
"""

import logging
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import optimize

from beqanalyser import BiquadCoefficients, FitMetrics

logger = logging.getLogger()


@dataclass
class FilterLimits:
    """Limits for filter parameters by type."""

    peaking_q: tuple[float, float] = (0.5, 2.0)
    shelf_q: tuple[float, float] = (0.5, 1.0)  # Shelves should have conservative Q
    gain_range: tuple[float, float] = (-40.0, 20.0)
    max_peaking_filters: int = 3  # Limit number of peaking filters for smooth curves


class RBJBiquad:
    """Robert Bristow-Johnson biquad filter generator."""

    @staticmethod
    def peaking_eq(
        fs: float, fc: float, gain_db: float, q: float
    ) -> BiquadCoefficients:
        """Generate peaking EQ biquad coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        return BiquadCoefficients(b0, b1, b2, a0, a1, a2, "peaking", fc, gain_db, q)

    @staticmethod
    def low_shelf(
        fs: float, fc: float, gain_db: float, q: float = 0.707
    ) -> BiquadCoefficients:
        """Generate low shelf biquad coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)

        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return BiquadCoefficients(b0, b1, b2, a0, a1, a2, "low_shelf", fc, gain_db, q)

    @staticmethod
    def high_shelf(
        fs: float, fc: float, gain_db: float, q: float = 0.707
    ) -> BiquadCoefficients:
        """Generate high shelf biquad coefficients."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return BiquadCoefficients(b0, b1, b2, a0, a1, a2, "high_shelf", fc, gain_db, q)


class CompositeCurveFitter:
    """Fit composite curves with cascaded biquad filters."""

    def __init__(
        self,
        fs: float = 48000,
        freq_range: tuple[float, float] = (5, 50),
        filter_limits: FilterLimits | None = None,
        fc_range_margin: float = 0.5,
    ):
        """
        Initialize the curve fitter.

        Args:
            fs: Sample rate in Hz
            freq_range: Frequency range for analysis and optimisation (min, max) in Hz
            filter_limits: Filter parameter limits (Q, gain). If None, uses defaults.
            fc_range_margin: Margin factor for filter centre frequencies beyond freq_range.
                           0.0 = filters must be within freq_range
                           0.1 = filters can extend 10% beyond freq_range on each side
                           Useful for shelf filters that affect the band edges.
        """
        self.fs = fs
        self.freq_range = freq_range
        self.freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 500)
        self.filter_limits = filter_limits or FilterLimits()
        # Derive fc_range from freq_range with optional margin
        margin = (freq_range[1] - freq_range[0]) * fc_range_margin
        self.fc_range = (
            max(1.0, freq_range[0] - margin),  # Don't go below 1 Hz
            min(fs / 2, freq_range[1] + margin),  # Don't exceed Nyquist
        )

    def compute_filter_response(self, biquads: list[BiquadCoefficients]) -> np.ndarray:
        """Compute frequency response of cascaded biquads."""
        w = 2 * np.pi * self.freqs / self.fs
        response = np.ones(len(w), dtype=complex)

        for bq in biquads:
            # Normalize coefficients
            b = np.array([bq.b0, bq.b1, bq.b2]) / bq.a0
            a = np.array([1.0, bq.a1 / bq.a0, bq.a2 / bq.a0])

            # Compute frequency response
            h = np.zeros(len(w), dtype=complex)
            for i, wi in enumerate(w):
                z = np.exp(1j * wi)
                h[i] = (b[0] + b[1] * z ** (-1) + b[2] * z ** (-2)) / (
                    a[0] + a[1] * z ** (-1) + a[2] * z ** (-2)
                )

            response *= h

        return 20 * np.log10(np.abs(response))

    def fit_curve(
        self,
        target_freqs: np.ndarray,
        target_db: np.ndarray,
        max_filters: int = 10,
        validate_range: bool = True,
        prefer_shelves: bool = True,
        smoothness_penalty: float = 0.1,
    ) -> tuple[list[BiquadCoefficients], FitMetrics | None]:
        """
        Fit a target curve with biquad filters using optimisation.

        Args:
            target_freqs: Frequency points of target curve (Hz)
            target_db: Magnitude response in dB
            max_filters: Maximum number of biquad filters to use
            validate_range: If True, warn if target frequencies extend beyond freq_range
            prefer_shelves: If True, strongly prefer shelf filters over peaking (recommended for smooth curves)
            smoothness_penalty: Regularization weight to prevent ripples (0.0-1.0, default 0.1)
                               Higher values = smoother response, slightly higher RMS error

        Returns:
            List of biquad coefficients
        """
        # Validate frequency range compatibility
        if validate_range:
            target_min, target_max = target_freqs[0], target_freqs[-1]
            fitter_min, fitter_max = self.freq_range

            if target_min < fitter_min * 0.5 or target_max > fitter_max * 2.0:
                import warnings

                warnings.warn(
                    f"Target frequency range [{target_min:.1f}, {target_max:.1f}] Hz "
                    f"differs significantly from fitter range [{fitter_min:.1f}, {fitter_max:.1f}] Hz. "
                    f"Consider adjusting freq_range in CompositeCurveFitter constructor for better results.",
                    UserWarning,
                )

        # Interpolate target to our frequency grid
        target_interp = np.interp(self.freqs, target_freqs, target_db)

        # Start with simple parametric fit
        a = time.time()
        biquads = self._parametric_fit(target_interp, max_filters, prefer_shelves)
        b = time.time()
        logging.info(f"Parametric fit took {b - a:.2f}s")

        # Refine with optimization
        biquads, fit_metrics = self._optimize_filters(
            biquads, target_interp, smoothness_penalty
        )
        c = time.time()
        logging.info(f"Optimise took {c - b:.2f}s")

        # Post-optimization validation: remove problematic peaking filters
        if prefer_shelves:
            biquads = self._validate_and_cleanup(biquads, target_interp)
            # Recompute metrics after cleanup
            final_response = self.compute_filter_response(biquads)
            errors = final_response - target_interp
            fit_metrics = FitMetrics(
                sse=float(np.sum(errors**2)),
                rms_error=float(np.sqrt(np.mean(errors**2))),
                max_error=float(np.max(np.abs(errors))),
                mean_abs_error=float(np.mean(np.abs(errors))),
                n_points=len(self.freqs),
                n_filters=len(biquads),
            )

        return biquads, fit_metrics

    def _parametric_fit(
        self, target_db: np.ndarray, max_filters: int, prefer_shelves: bool = True
    ) -> list[BiquadCoefficients]:
        """Initial parametric fit to identify filter candidates.

        For smooth, broad curves (like BEQ composites), we prefer shelf filters
        over peaking filters as they're more stable and appropriate for gentle slopes.
        """
        biquads = []

        # Find overall trend (low shelf)
        # Use first 10% of frequency range or first 50 points, whichever is smaller
        low_freq_cutoff = min(int(len(self.freqs) * 0.1), 50)
        low_freq_gain = np.mean(target_db[:low_freq_cutoff])

        if abs(low_freq_gain) > 0.5:
            # Use conservative Q for shelf filters
            shelf_q = np.mean(self.filter_limits.shelf_q)  # Use middle of range
            # Place shelf at ~20% into the low frequency region
            shelf_fc = self.freqs[low_freq_cutoff // 2]
            biquads.append(
                RBJBiquad.low_shelf(self.fs, shelf_fc, low_freq_gain, shelf_q)
            )
            logger.info(f"Initial LS: gain={low_freq_gain:.2f}, Q={shelf_q:.2f}")
        else:
            logger.info(f"No initial LS for low_freq_gain={low_freq_gain:.2f}")

        # Check if we should add a high shelf for rolloff
        high_freq_cutoff = max(int(len(self.freqs) * 0.9), len(self.freqs) - 50)
        high_freq_gain = np.mean(target_db[high_freq_cutoff:])

        # If high frequencies show significant different trend from low, add high shelf
        if len(biquads) > 0:
            gain_difference = high_freq_gain - low_freq_gain
            if abs(gain_difference) > 2.0:  # Significant slope across band
                shelf_q = np.mean(self.filter_limits.shelf_q)
                shelf_fc = self.freqs[(high_freq_cutoff + len(self.freqs)) // 2]
                biquads.append(
                    RBJBiquad.high_shelf(self.fs, shelf_fc, gain_difference, shelf_q)
                )
                logger.info(f"Initial HS: gain={gain_difference:.2f}, Q={shelf_q:.2f}")
            else:
                logger.info(f"No initial HS for gain_difference={gain_difference:.2f}")

        # Only add peaking filters if explicitly allowed and needed
        if not prefer_shelves or len(biquads) < max_filters:
            max_peaking = (
                self.filter_limits.max_peaking_filters
                if prefer_shelves
                else max_filters - len(biquads)
            )

            # Only add peaking filters for significant local features that shelves can't capture
            # For smooth curves, this should rarely trigger
            if len(biquads) < max_filters and max_peaking > 0:
                # Compute derivative to find inflection points
                derivative = np.gradient(target_db)
                second_derivative = np.gradient(derivative)

                # Find zero crossings in second derivative (peaks/dips)
                zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]

                # Only add peaking filters for features that are:
                # 1. Narrow (high curvature)
                # 2. Significant amplitude (> 3 dB from trend for smooth curves)
                threshold = 3.0 if prefer_shelves else 2.0

                added_peaking = 0
                for idx in zero_crossings:
                    if added_peaking >= max_peaking or len(biquads) >= max_filters:
                        break

                    fc = self.freqs[idx]

                    # Estimate local trend from existing shelves
                    current_trend = sum(
                        bq.gain for bq in biquads if "shelf" in bq.filter_type
                    )
                    local_gain = target_db[idx] - current_trend

                    # Only add if it's a significant local feature
                    if abs(local_gain) > threshold:
                        # Estimate Q from width of feature
                        half_gain = local_gain / 2
                        left_idx = idx
                        right_idx = idx

                        while left_idx > 0 and target_db[left_idx] > target_db[
                            idx
                        ] - abs(half_gain):
                            left_idx -= 1
                        while right_idx < len(target_db) - 1 and target_db[
                            right_idx
                        ] > target_db[idx] - abs(half_gain):
                            right_idx += 1

                        bw = self.freqs[right_idx] - self.freqs[left_idx]
                        q = fc / max(bw, fc / 10)  # Limit Q

                        # For smooth curves, prefer wider (lower Q) peaking filters
                        # Apply filter-specific Q limits with bias toward lower Q
                        q = np.clip(
                            q,
                            self.filter_limits.peaking_q[0],
                            self.filter_limits.peaking_q[1],
                        )

                        biquads.append(RBJBiquad.peaking_eq(self.fs, fc, local_gain, q))
                        added_peaking += 1
                        logger.info(
                            f"Added Peaking: fc={fc:.2f}, gain={local_gain:.2f}, Q={q:.2f}"
                        )

        return biquads

    def _optimize_filters(
        self,
        initial_biquads: list[BiquadCoefficients],
        target_db: np.ndarray,
        smoothness_penalty: float = 0.0,
    ) -> tuple[list[BiquadCoefficients], FitMetrics | None]:
        """Refine filter parameters using optimisation (fast version).

        Args:
            initial_biquads: Starting filter configuration
            target_db: Target magnitude response
            smoothness_penalty: Penalty for non-smooth responses (0.0 = none, 0.1 = moderate, 1.0 = strong)
                               Helps prevent ripple artifacts in smooth curves.

        Returns:
            Tuple of (optimized_biquads, fit_metrics)
        """
        if not initial_biquads:
            return [], FitMetrics(0, 0, 0, 0, len(self.freqs), 0)

        # Pack parameters: [fc1, gain1, q1, fc2, gain2, q2, ...]
        x0 = []
        bounds = []
        filter_types = []

        for bq in initial_biquads:
            x0.extend([bq.fc, bq.gain, bq.q])

            # Apply type-specific Q limits
            if bq.filter_type in ["low_shelf", "high_shelf"]:
                q_bounds = self.filter_limits.shelf_q
            else:  # peaking
                q_bounds = self.filter_limits.peaking_q

            bounds.extend(
                [
                    self.fc_range,  # fc
                    self.filter_limits.gain_range,  # gain
                    q_bounds,  # Q (type-specific)
                ]
            )
            filter_types.append(bq.filter_type)

        x0 = np.array(x0)

        # Pre-compute frequency-dependent values once
        w = 2 * np.pi * self.freqs / self.fs
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        z_inv = np.exp(-1j * w)
        z_inv2 = z_inv**2

        def objective(x):
            """Fast vectorized objective function with optional smoothness penalty."""
            response = np.ones(len(w), dtype=np.complex128)

            for i in range(0, len(x), 3):
                fc, gain, q = x[i : i + 3]
                ftype = filter_types[i // 3]

                # Safety clamp on Q
                if ftype in ["low_shelf", "high_shelf"]:
                    q = np.clip(
                        q, self.filter_limits.shelf_q[0], self.filter_limits.shelf_q[1]
                    )
                else:
                    q = np.clip(
                        q,
                        self.filter_limits.peaking_q[0],
                        self.filter_limits.peaking_q[1],
                    )

                # Compute filter coefficients
                A = 10 ** (gain / 40)
                w0 = 2 * np.pi * fc / self.fs
                cos_w0 = np.cos(w0)
                sin_w0 = np.sin(w0)

                if ftype == "peaking":
                    alpha = sin_w0 / (2 * q)
                    b0 = 1 + alpha * A
                    b1 = -2 * cos_w0
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * cos_w0
                    a2 = 1 - alpha / A
                elif ftype == "low_shelf":
                    alpha = sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)
                    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
                    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
                    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
                    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
                    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
                else:  # high_shelf
                    alpha = sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / q - 1) + 2)
                    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
                    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
                    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
                    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
                    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

                # Normalize and compute frequency response (vectorized)
                b0_norm, b1_norm, b2_norm = b0 / a0, b1 / a0, b2 / a0
                a1_norm, a2_norm = a1 / a0, a2 / a0

                h = (b0_norm + b1_norm * z_inv + b2_norm * z_inv2) / (
                    1.0 + a1_norm * z_inv + a2_norm * z_inv2
                )

                response *= h

            response_db = 20 * np.log10(np.abs(response))

            # Primary error term: fit to target
            fit_error = np.sum((response_db - target_db) ** 2)

            # Smoothness penalty: penalize high-frequency content (ripples)
            if smoothness_penalty > 0:
                # Compute second derivative of response (curvature)
                response_grad = np.gradient(response_db)
                response_curvature = np.gradient(response_grad)

                # Penalize high curvature (indicates ripples/peaks)
                smoothness_term = np.sum(response_curvature**2)

                # Combined objective with weighted smoothness penalty
                return fit_error + smoothness_penalty * smoothness_term

            logger.debug(
                f"Optimised to objective with len {len(x)} parameters, error is {fit_error:.2f}"
            )
            return fit_error

        result = optimize.minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        # Reconstruct biquads from optimized parameters
        optimized_biquads: list[BiquadCoefficients] = []
        for i in range(0, len(result.x), 3):
            fc, gain, q = result.x[i : i + 3]
            ftype = filter_types[i // 3]

            # Final safety clamp on Q
            if ftype in ["low_shelf", "high_shelf"]:
                q = np.clip(
                    q, self.filter_limits.shelf_q[0], self.filter_limits.shelf_q[1]
                )
            else:
                q = np.clip(
                    q, self.filter_limits.peaking_q[0], self.filter_limits.peaking_q[1]
                )

            if ftype == "low_shelf":
                optimized_biquads.append(RBJBiquad.low_shelf(self.fs, fc, gain, q))
            elif ftype == "high_shelf":
                optimized_biquads.append(RBJBiquad.high_shelf(self.fs, fc, gain, q))
            else:
                optimized_biquads.append(RBJBiquad.peaking_eq(self.fs, fc, gain, q))

        # Compute final fit metrics
        final_response = self.compute_filter_response(optimized_biquads)
        errors = final_response - target_db

        fit_metrics = FitMetrics(
            sse=result.fun,
            rms_error=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(np.abs(errors))),
            mean_abs_error=float(np.mean(np.abs(errors))),
            n_points=len(self.freqs),
            n_filters=len(optimized_biquads),
        )

        return optimized_biquads, fit_metrics

    def _validate_and_cleanup(
        self, biquads: list[BiquadCoefficients], target_db: np.ndarray
    ) -> list[BiquadCoefficients]:
        """
        Post-optimization validation: remove peaking filters that create artifacts.

        This catches cases where the optimizer found a local minimum with inappropriate
        peaking filters that create ripple in an otherwise smooth response.
        """
        if not biquads:
            return biquads

        # Separate shelves from peaking filters
        shelves = [bq for bq in biquads if "shelf" in bq.filter_type]
        peaking = [bq for bq in biquads if bq.filter_type == "peaking"]

        if not peaking:
            return biquads  # No peaking filters to validate

        # Compute response with just shelves
        shelf_response = (
            self.compute_filter_response(shelves)
            if shelves
            else np.zeros_like(target_db)
        )

        # Check each peaking filter: does it improve the fit or create artifacts?
        validated_peaking = []
        for pk in peaking:
            # Compute error with and without this peaking filter
            current_filters = shelves + validated_peaking
            current_response = (
                self.compute_filter_response(current_filters)
                if current_filters
                else np.zeros_like(target_db)
            )

            with_peak = current_filters + [pk]
            with_peak_response = self.compute_filter_response(with_peak)

            # Measure errors
            error_without = np.abs(current_response - target_db)
            error_with = np.abs(with_peak_response - target_db)

            # Check if peaking filter improves RMS error significantly
            rms_without = np.sqrt(np.mean(error_without**2))
            rms_with = np.sqrt(np.mean(error_with**2))

            # Also check if it creates localized artifacts (high max error)
            max_error_increase = np.max(error_with) - np.max(error_without)

            # Keep the peaking filter only if:
            # 1. It improves RMS by at least 0.2 dB
            # 2. It doesn't increase max error by more than 0.5 dB
            # 3. Its gain is reasonable (not trying to cancel out shelf with opposite peaking)
            improvement = rms_without - rms_with
            reasonable_gain = abs(pk.gain) < 10.0  # Don't allow extreme gains

            if improvement > 0.2 and max_error_increase < 0.5 and reasonable_gain:
                validated_peaking.append(pk)

        result = shelves + validated_peaking

        if len(validated_peaking) < len(peaking):
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Removed {len(peaking) - len(validated_peaking)} problematic peaking filters"
            )

        return result


def export_filters(biquads: list[BiquadCoefficients], name: str = "filter") -> dict:
    """Export biquad filters to various formats."""
    result = {"name": name, "sample_rate": 48000, "filters": []}

    for i, bq in enumerate(biquads):
        # Normalized coefficients
        b0_norm = bq.b0 / bq.a0
        b1_norm = bq.b1 / bq.a0
        b2_norm = bq.b2 / bq.a0
        a1_norm = bq.a1 / bq.a0
        a2_norm = bq.a2 / bq.a0

        filter_data = {
            "index": i,
            "type": bq.filter_type,
            "fc": round(bq.fc, 2),
            "gain": round(bq.gain, 2),
            "q": round(bq.q, 3),
            "coefficients": {
                "b0": b0_norm,
                "b1": b1_norm,
                "b2": b2_norm,
                "a0": 1.0,
                "a1": a1_norm,
                "a2": a2_norm,
            },
        }
        result["filters"].append(filter_data)

    return result


def plot_filter_comparison(
    composite_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    fitted_filters: dict[str, list[BiquadCoefficients]],
    fs: float = 48000,
    freq_range: tuple[float, float] = (10, 200),
    figsize: tuple[float, float] | None = None,
    save_path: str | str = None,
):
    """
    Plot composite curves and their fitted filters on a grid of subplots.

    Args:
        composite_curves: Dict mapping curve names to (freq, dB) tuples
        fitted_filters: Dict mapping curve names to lists of BiquadCoefficients
        fs: Sample rate in Hz
        freq_range: Frequency range for plotting (min, max) in Hz
        figsize: Figure size (width, height). Auto-calculated if None
        save_path: Path to save the figure. If None, displays instead
    """
    n_curves = len(composite_curves)

    if n_curves == 0:
        print("No curves to plot")
        return

    # Calculate grid dimensions
    n_cols = min(3, n_curves)  # Max 3 columns
    n_rows = (n_curves + n_cols - 1) // n_cols

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    # Generate frequency grid for fitted responses
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 500)

    # Create fitter for computing responses
    fitter = CompositeCurveFitter(fs=fs, freq_range=freq_range)

    for idx, (name, (target_freqs, target_db)) in enumerate(composite_curves.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Plot target curve
        ax.semilogx(
            target_freqs,
            target_db,
            "o-",
            linewidth=2,
            markersize=4,
            alpha=0.7,
            label="Target Composite",
            color="#2E86AB",
        )

        # Plot fitted filter response if available
        if name in fitted_filters and fitted_filters[name]:
            biquads = fitted_filters[name]
            fitted_response = fitter.compute_filter_response(biquads)
            ax.semilogx(
                fitter.freqs,
                fitted_response,
                "-",
                linewidth=2,
                label=f"Fitted ({len(biquads)} filters)",
                color="#A23B72",
            )

            # Calculate and display RMS error
            target_interp = np.interp(fitter.freqs, target_freqs, target_db)
            rms_error = np.sqrt(np.mean((fitted_response - target_interp) ** 2))

            # Plot individual filter contributions (optional, lighter lines)
            if len(biquads) <= 5:  # Only show for simple cases
                cumulative = np.zeros(len(fitter.freqs))
                for i, bq in enumerate(biquads):
                    response = fitter.compute_filter_response([bq])
                    cumulative += response
                    ax.semilogx(
                        fitter.freqs,
                        response,
                        "--",
                        linewidth=1,
                        alpha=0.3,
                        label=f"{bq.filter_type} @ {bq.fc:.0f}Hz",
                    )
        else:
            rms_error = None

        # Formatting
        ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Magnitude (dB)", fontsize=10)
        ax.set_xlim(freq_range)

        # Set y-axis limits with some padding
        all_db = list(target_db)
        if name in fitted_filters and fitted_filters[name]:
            all_db.extend(fitted_response)
        y_min, y_max = min(all_db), max(all_db)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Title with RMS error if available
        title = name.replace("_", " ").title()
        if rms_error is not None:
            title += f"\n(RMS Error: {rms_error:.2f} dB)"
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Legend
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # Remove empty subplots
    for idx in range(n_curves, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(fig.add_subplot(gs[row, col]))

    plt.suptitle(
        "BEQ Composite Curves vs Fitted Biquad Filters",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_single_filter_detail(
    name: str,
    target_freqs: np.ndarray,
    target_db: np.ndarray,
    biquads: list[BiquadCoefficients],
    fs: float = 48000,
    freq_range: tuple[float, float] = (10, 200),
    save_path: str | None = None,
):
    """
    Create a detailed plot for a single filter showing individual contributions.

    Args:
        name: Name of the curve
        target_freqs: Target frequency points
        target_db: Target magnitude in dB
        biquads: List of fitted biquad coefficients
        fs: Sample rate
        freq_range: Frequency range for plotting
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    fitter = CompositeCurveFitter(fs=fs, freq_range=freq_range)

    # Top plot: Overall comparison
    ax1.semilogx(
        target_freqs,
        target_db,
        "o-",
        linewidth=2,
        markersize=6,
        label="Target Composite",
        color="#2E86AB",
    )

    if biquads:
        fitted_response = fitter.compute_filter_response(biquads)
        ax1.semilogx(
            fitter.freqs,
            fitted_response,
            "-",
            linewidth=2.5,
            label="Fitted Response",
            color="#A23B72",
        )

        # Calculate error
        target_interp = np.interp(fitter.freqs, target_freqs, target_db)
        error = fitted_response - target_interp
        rms_error = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))

        ax1.text(
            0.02,
            0.98,
            f"RMS Error: {rms_error:.3f} dB\nMax Error: {max_error:.3f} dB",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(f"{name} - Overall Fit", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(freq_range)

    # Bottom plot: Individual filter contributions
    if biquads:
        colors = plt.cm.tab10(np.linspace(0, 1, len(biquads)))

        for i, (bq, color) in enumerate(zip(biquads, colors)):
            response = fitter.compute_filter_response([bq])
            label = f"{i + 1}. {bq.filter_type}: {bq.fc:.1f}Hz, {bq.gain:+.1f}dB, Q={bq.q:.2f}"
            ax2.semilogx(
                fitter.freqs, response, "-", linewidth=2, label=label, color=color
            )

        # Also plot cumulative sum
        cumulative = np.zeros(len(fitter.freqs))
        for bq in biquads:
            cumulative += fitter.compute_filter_response([bq])
        ax2.semilogx(
            fitter.freqs,
            cumulative,
            "k--",
            linewidth=2,
            alpha=0.5,
            label="Sum of Individual",
        )

    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Magnitude (dB)", fontsize=11)
    ax2.set_title("Individual Filter Contributions", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="best", ncol=1)
    ax2.set_xlim(freq_range)
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Detailed plot saved to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example: fit multiple composite curves with custom filter limits
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - [%(threadName)s] - %(message)s",
    )

    # Define conservative filter limits to ensure stability
    custom_limits = FilterLimits(
        peaking_q=(0.5, 2.0),  # Peaking filters: very conservative Q range
        shelf_q=(0.5, 0.9),  # Shelf filters: very conservative Q (prevents instability)
        gain_range=(-15.0, 15.0),  # Gain range
        max_peaking_filters=2,  # Limit peaking filters for smooth curves
    )

    # freq_range defines both the analysis band AND the filter placement range
    fitter = CompositeCurveFitter(
        fs=48000,
        freq_range=(5, 50),  # BEQ typical band - filters will be placed here
        filter_limits=custom_limits,
        fc_range_margin=0.2,  # Allow filters 20% beyond band (for shelf edges)
    )

    # Simulate multiple composite curves (replace with actual BEQ composite data)
    composite_curves = {
        "action_heavy": (
            np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]),
            np.array([3, 5, 7, 8, 7, 5, 3, 1, 0, -1, -2, -2]),
        ),
        "action_moderate": (
            np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]),
            np.array([2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -1.5, -1.5]),
        ),
        "drama": (
            np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]),
            np.array([1, 2, 2.5, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1, -1]),
        ),
        "sci_fi": (
            np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]),
            np.array([2, 4, 6, 6.5, 5, 3, 2, 1, 0, -0.5, -1, -1.5]),
        ),
    }

    # Fit all curves
    fitted_filters = {}
    for name, (freqs, db) in composite_curves.items():
        logging.info(f"Fitting {name}...")
        a = time.time()
        biquads, metrics = fitter.fit_curve(freqs, db, max_filters=8)
        b = time.time()
        logging.info(f"Fitted {name} using {len(biquads)} filters in {b - a:.2f}s")
        fitted_filters[name] = biquads

        # Print fit metrics
        print(f"  Generated {len(biquads)} filters")
        print(f"  {metrics}")
        print(f"  Quality: {'✓ GOOD' if metrics.is_good_fit() else '✗ POOR'}")

        # Export individual filter
        filter_export = export_filters(biquads, name)
        logging.info(f"  Generated {len(biquads)} filters")

    # Plot all curves in a grid
    logging.info("Generating comparison plot...")
    plot_filter_comparison(
        composite_curves, fitted_filters, save_path="beq_composite_comparison.png"
    )

    # Create detailed plot for first curve
    logging.info("Generating detailed plot for first curve...")
    first_name = list(composite_curves.keys())[0]
    first_freqs, first_db = composite_curves[first_name]
    first_filters = fitted_filters[first_name]

    plot_single_filter_detail(
        first_name,
        first_freqs,
        first_db,
        first_filters,
        save_path=f"beq_{first_name}_detail.png",
    )

    # Print summary of all filters with stability info
    logging.info("" + "=" * 70)
    logging.info("FILTER SUMMARY (with stability constraints)")
    logging.info("=" * 70)
    logging.info(f"Filter Limits Applied:")
    logging.info(
        f"  Peaking Q: {custom_limits.peaking_q[0]:.2f} - {custom_limits.peaking_q[1]:.2f}"
    )
    logging.info(
        f"  Shelf Q:   {custom_limits.shelf_q[0]:.2f} - {custom_limits.shelf_q[1]:.2f}"
    )
    logging.info(f"  Fc Range:  {fitter.fc_range[0]:.0f} - {fitter.fc_range[1]:.0f} Hz")
    logging.info(
        f"  Gain:      {custom_limits.gain_range[0]:+.0f} - {custom_limits.gain_range[1]:+.0f} dB"
    )
    logging.info("=" * 70)

    for name, biquads in fitted_filters.items():
        logging.info(f"{name.upper()} ({len(biquads)} filters):")
        for i, bq in enumerate(biquads):
            stability_note = ""
            if bq.filter_type in ["low_shelf", "high_shelf"] and bq.q > 1.0:
                stability_note = " ⚠️ HIGH Q"
            logging.info(
                f"  {i + 1}. {bq.filter_type:12s} | "
                f"Fc: {bq.fc:6.1f} Hz | "
                f"Gain: {bq.gain:+6.2f} dB | "
                f"Q: {bq.q:5.2f}{stability_note}"
            )

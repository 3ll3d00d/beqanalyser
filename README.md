# BEQ Analyser

NB: Code and requirements are LLM generated with human guidance/review. 

---

## 1. Analysis Pipeline Features

### 1.1 Input Data Assumptions

* Input is a catalogue of **N BEQ curves**
* Curves are:
    * Normalised
    * Sampled on a **common frequency axis**
    * Expressed in **dB vs frequency**
    * Band-limited
* Frequency axis is **linear spaced**
* Catalogue shape:

  ```
  catalogue: ndarray[N, F]
  freqs: ndarray[F]
  ```

#### Frequency Axis Consistency

All input curves are assumed to share a common frequency axis and sampling resolution.

This is treated as a hard precondition of the analysis pipeline and is not explicitly validated at runtime. Input data must therefore be pre-aligned prior to analysis.

### 1.2 Pipeline Overview

The BEQ analysis pipeline proceeds through the following stages:

1. Load and convert catalogue filters into frequency responses
2. Apply band-limiting and optional weighting
3. Perform prototype reduction if required
4. Cluster prototypes to generate initial composites
5. Assign all catalogue curves to composites with metric evaluation
6. Recompute composite curves from assigned members
7. Generate fan curves and visualizations
8. Produce summary statistics and assignment reports

Each stage builds on the results of the previous stage.

---

### 1.2 Distance & Similarity Metrics

Each curve-to-composite comparison computes:

| Metric               | Description                               |
| -------------------- | ----------------------------------------- |
| RMS                  | Root mean square deviation from composite |
| MAX                  | Maximum absolute deviation from composite |
| Cosine similarity    | Shape similarity independent of magnitude |
| Derivative deviation | RMS of first derivative difference        |

All metrics are **stored per attempted assignment**, regardless of acceptance.

Each curve–composite comparison computes multiple complementary metrics.
No single metric is sufficient to capture perceptual similarity in infra-bass BEQ filters; instead, the metrics collectively guard against distinct failure modes observed in real catalogue data.

#### Composite Selection

When assigning a curve to a composite, the “best” composite is selected by selecting the highest cosine similarity from the composites with an RMS deviation within epsilon of the minimum RMS deviation.

Other similarity metrics (maximum deviation, derivative RMS) are used exclusively as **acceptance or rejection thresholds** and do not influence the ranking of candidate composites.

As a result, RMS is the primary metric determining composite assignment with cosine similarity as a secondary tiebreaker, while the remaining metrics act as safeguards against perceptually dissimilar matches.

#### Metric Weighting

Optional frequency weighting may be applied when computing RMS deviation.

Weights affect **only the RMS metric** and are not applied to cosine similarity, derivative RMS, or maximum deviation calculations. All other metrics operate on unweighted response data.

---

### RMS Deviation (Perceptually Weighted)

**Definition**
Root mean square of the difference between the curve and the composite over the selected band, optionally weighted by a perceptual frequency weighting.

**Why it is used**

* Measures **overall energy deviation** across the band
* Closely correlates with **perceived loudness difference**
* Penalises distributed mismatches more than localised ones
* Stable under noise and small local variations

**What it catches**

* Filters that are broadly “too strong” or “too weak”
* Gradual shape drift across the band
* Cumulative differences that are perceptually obvious but locally small

**What it does *not* catch well**

* Narrow, sharp deviations
* Directional shape inversions (e.g. shelf vs peak)
* Localised filter topology changes

---

### Maximum Absolute Deviation

**Definition**
Maximum absolute per-frequency difference between the curve and the composite.

**Why it is used**

* Acts as a **hard safety constraint**
* Prevents visually or perceptually egregious mismatches
* Guards against narrow but extreme features being hidden by RMS averaging

**What it catches**

* Sharp peaking filters
* Unexpected notches or spikes
* Low-pass or high-pass rolloffs that diverge sharply at one end

**What it does *not* catch well**

* Broad but moderate deviations
* Distributed shape differences with no extreme point

---

### Cosine Similarity (Shape Similarity)

**Definition**
Cosine similarity between mean-removed curves, treating each curve as a vector in frequency space.

**Why it is used**

* Measures **directional similarity of shape**, independent of magnitude
* Captures whether two curves “move together”
* Robust to overall strength scaling

**What it catches**

* Shape inversions (e.g. shelf vs inverted shelf)
* Filters with different structural intent:

    * Shelf vs peak
    * Boost vs cut
* Low-pass–like rolloffs vs broadband responses

**What it does *not* catch well**

* Absolute strength differences
* Localised deviations if overall trend aligns

---

### First-Derivative Deviation (Slope / Curvature)

**Definition**
RMS deviation of the first derivative (slope) of the curve vs the composite.

**Why it is used**

* Captures **rate of change differences**
* Sensitive to filter topology rather than level
* Penalises curves that change direction or curvature unexpectedly

**What it catches**

* Additional poles/zeros
* Sharp transitions or knees
* Peaking filters applied on top of shelves
* Low-pass filters appended to otherwise similar shapes

**What it does *not* catch well**

* Parallel but offset curves
* Uniform strength differences

---

### Why Multiple Metrics Are Required

Each metric guards against a **different failure mode**:

| Failure Mode       | RMS | Max | Cosine | Derivative |
| ------------------ | --- | --- | ------ | ---------- |
| Too strong / weak  | ✓   | ✗   | ✗      | ✗          |
| Sharp spike        | ✗   | ✓   | ✗      | ✓          |
| Shape inversion    | ✗   | ✗   | ✓      | ✓          |
| Extra filter stage | ✗   | ✓   | ✗      | ✓          |
| Broad drift        | ✓   | ✗   | ✗      | ✗          |

Only by combining these metrics can the pipeline:

* Maintain **perceptual consistency**
* Avoid visually misleading composites
* Allow **looser RMS/MAX thresholds** without admitting structurally different filters

---

### Design Principle

> **RMS measures “how much”, cosine measures “which way”, derivative measures “how”.**

The rejection logic intentionally treats these metrics as **orthogonal constraints**, not interchangeable thresholds.

---

### 1.3 Clustering & Prototype Reduction

* Initial clustering may use:
    * Hierarchical (Ward)
    * k-means (optional)
* Prototype reduction strategy:
    * Reduce full catalogue to a smaller set of representative shapes
    * Final composites are **real curves or averaged curves**
* Clustering occurs **before** rejection logic

#### Prototype Reduction

For large catalogues, clustering is performed on a reduced subset of representative curves rather than the full dataset.

If the number of input responses exceeds `n_prototypes`, a **k-medoids** selection is performed to identify a subset of curves that best represent the overall catalogue. Hierarchical clustering is then applied only to this prototype set. All remaining curves are subsequently assigned to the resulting composites.

This step reduces computational cost while preserving the overall shape distribution of the catalogue. The choice of `n_prototypes` may influence the resulting composites.

---

### 1.4 Composite Construction & Definition

Each composite contains:

* `shape`: composite curve (mean or medoid)
* `assigned_indices`: indices of accepted catalogue entries
* `fan_envelopes`: multi-level envelope bands derived from assigned curves

Composite curves are defined as the **per-frequency median** of all curves assigned to the composite.

Median aggregation is used instead of a mean to reduce sensitivity to outliers and to produce a more robust representative shape. 

Composite curves are recomputed after assignment to reflect the current membership of each composite.

---

### 1.5 Fan Envelope Computation

* Fan envelopes are computed using **sorted RMS distance**
* Curves are partitioned into percentile bands
* Each envelope contains:

    * A **unique subset** of assigned curves
* No curve appears in more than one envelope band

---

### 1.6 Assignment & Rejection Logic

Each catalogue entry is evaluated against **each composite** and produces exactly one `BEQFilterMapping`.

#### Possible Outcomes

* Assigned to a composite
* Rejected from a composite (with reason)

#### Rejection Reasons (Authoritative)

| Reason                | Meaning                             |
| --------------------- | ----------------------------------- |
| `RMS_EXCEEDED`        | RMS deviation exceeds threshold     |
| `MAX_EXCEEDED`        | Maximum deviation exceeds threshold |
| `BOTH_EXCEEDED`       | RMS and MAX exceed thresholds       |
| `COSINE_TOO_LOW`      | Shape similarity below threshold    |
| `DERIVATIVE_TOO_HIGH` | Excessive shape roughness mismatch  |

---

### Assignment Records

Assignment results are exported as tabular data containing, for each catalogue entry:

- Assigned composite identifier
- RMS deviation
- Maximum absolute deviation
- Cosine similarity
- Derivative RMS
- Assignment status (accepted or rejected)
- Rejection reason (if applicable)
- Catalogue metadata (e.g. title, author)

These records provide a complete audit trail of the assignment decision process.

---

## 2. Plotting Subsystem Features

Plotting is divided into **assigned curves** and **rejected curves**, with **zero overlap**.

## 2.1. Common Requirements

### 2.1.1. Core Goals

* No curve is plotted twice in the same figure
* Assigned and rejected curves are never mixed
* Legend appears once per figure
* Consistent colour semantics across all plots
* Fully deterministic ordering

### 2.2. Non-Goals

* No interactive widgets
* No animated plots
* No implicit curve downsampling
* No curve re-normalisation during plotting
* No silent dropping of rejected curves

---

## 2.2. Assigned Fan Curve Plotting

* Grid of subplots
* One subplot per composite
* Maximum 3 composites per row
* Shared axes

### Fan Curves

Fan curves visualize the variability of responses assigned to a composite.

Assigned curves are first sorted by increasing RMS deviation from the composite. Fan envelopes are then constructed using progressively larger percentile subsets of this ordered set. Each fan therefore represents the range of curves within a given RMS tolerance rather than the absolute min/max bounds.

This approach emphasizes typical variation while avoiding domination by extreme outliers.

---

### 2.2.1 Fan Curve Rendering

* Fan envelopes plotted from **tightest to loosest**
* Alpha increases with RMS rank
* Colour:

    * Light blue (assigned curves)
* No duplicate curves plotted

---

### 2.2.2 Composite Overlay

* Composite curve plotted in:

    * Black
    * Increased linewidth
    * Above fan curves (z-order)

---

### 2.2.3 Assigned RMS Histogram (Inset)

* Each composite subplot includes an inset histogram
* Histogram shows:

    * RMS values of **assigned curves only**
* Histogram properties:

    * Fixed inset position
    * Light blue bars
    * Independent scaling per composite

---

### 2.2.4 Titles & Labels

* Subplot title includes:

    * Composite index
    * Number of assigned curves
* Axes:

    * Frequency (Hz)
    * Magnitude (dB)
* Grid enabled with low alpha

---

## 2.3. Rejected Curve Plotting (Authoritative)

### 2.3.1 Separation by Rejection Reason

* Rejected curves are plotted:

    * **Separately from assigned curves**
    * In a **distinct figure per rejection reason**
* No rejected curve appears in more than one figure

---

### 2.3.2 Layout

* Grid of subplots identical to assigned plot layout
* One subplot per composite
* One figure per rejection reason

---

### 2.3.3 Rejected Curve Rendering

* Same fan-style rendering as assigned curves
* Sorted by RMS distance
* Colour:

    * Red-based (e.g. light coral)
* Alpha ramps from faint to strong
* Composite overlay in black

---

### 2.4. Metric-Specific Histograms (Inset)

Each rejected-curve subplot includes an inset histogram showing **only the metric responsible for rejection**.

| Rejection Reason    | Histogram Metric      |
| ------------------- | --------------------- |
| RMS_EXCEEDED        | RMS                   |
| MAX_EXCEEDED        | Max deviation         |
| BOTH_EXCEEDED       | RMS + Max (overlaid)  |
| COSINE_TOO_LOW      | 1 − Cosine similarity |
| DERIVATIVE_TOO_HIGH | Derivative deviation  |

Histogram rules:

* Only rejected curves for that composite & reason
* Colour-coded by metric
* Small-font legends where needed

---

### 2.4.1 Titles

* Figure title:

  ```
  Rejected Curves — Reason: <REASON>
  ```
* Subplot title:

    * Composite index
    * Count of rejected curves

---

## 3. Intended Use

This document is the **single source of truth** for:

* Refactoring
* Bug fixing
* Performance optimisation
* Re-implementation in other languages
* Regression testing

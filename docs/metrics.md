# Deconvolution Metrics

CIDeconvolve reports simple, no-reference image metrics for each channel before
and after deconvolution. These metrics are not a ground-truth quality score.
They are meant to show how the image changed: where fine detail increased,
where signal became more localized, and whether the robust intensity range was
compressed or expanded.

Metric calculation is optional and disabled by default because the FFT-based
metrics can take noticeable time on large images. In the GUI, enable **Compute
image metrics** in the Log window before running deconvolution. In the wrapper,
use `--compute_metrics True`.

To keep the calculation responsive on large volumes, metrics are computed on a
deterministic strided sample of each channel. The sampled data keeps at most 32
planes along `Z` and at most 512 pixels along `Y` and `X`. Smaller images are
used unchanged. The GUI and console log show the sampled shape before the metric
table, for example:

```text
Metrics sample: full shape (96, 2048, 2048) -> sampled (32, 512, 512)
```

The formulas below use `S`, the sampled channel. All metrics are computed per
channel after normalizing `S` to the range `[0, 1]`:

```text
S_norm = (S - min(S)) / (max(S) - min(S))
```

If the sampled channel is constant, the normalized channel is treated as all
zeros. Because these metrics are sampled, they should be read as fast
diagnostic indicators rather than exact full-volume measurements.

## Detail Energy

Formula:

```text
F = FFT(S_norm - mean(S_norm))
P(f) = |F(f)|^2
Detail energy = sum(P(f) for |f| > 0.25 * max(|f|)) / sum(P(f))
```

What it says:

Detail energy is the fraction of Fourier power that lives in higher spatial
frequencies. Deconvolution often increases this value because blur is moved
back into finer structures.

How to read it:

A higher result/source ratio usually means more fine-scale structure is present.
Very large increases can also indicate ringing or noise amplification, so it
should be read together with the visual result and the foreground-focused metric
below.

## Bright Detail Energy

Formula:

```text
B = S_norm >= percentile(S_norm, 95)
F_bright = FFT((S_norm - mean(S_norm)) * B)
P_bright(f) = |F_bright(f)|^2
Bright detail energy = sum(P_bright(f) for |f| > 0.25 * max(|f|)) / sum(P_bright(f))
```

What it says:

Bright detail energy measures high-frequency content mainly in the brightest
signal regions. This is useful for fluorescence data because deconvolution
should sharpen real signal structures, not just add high-frequency texture in
the background.

How to read it:

An increase suggests that bright objects contain more localized detail after
deconvolution. This metric is often more relevant than whole-image detail energy
when the image has a large dark background.

## Edge Strength

Formula:

```text
grad(S) = gradient of S_norm along each image axis
Edge strength = mean(sqrt(sum(axis_gradient^2)))
```

What it says:

Edge strength measures the average gradient magnitude. It responds to sharper
boundaries and stronger local transitions.

How to read it:

An increase usually means sharper edges. It can decrease when deconvolution
removes haze, suppresses background texture, or concentrates signal into fewer
pixels. A decrease is not automatically bad.

## Signal Sparsity

Formula:

```text
x = sorted(S_norm flattened into one vector)
n = number of pixels
Signal sparsity = (2 * sum(i * x_i) / (n * sum(x))) - ((n + 1) / n)
                  for i = 1 ... n
```

What it says:

Signal sparsity is a Gini-style concentration metric. It is close to `0` when
intensity is spread evenly over many pixels and approaches `1` when intensity is
concentrated in a small number of bright pixels. Deconvolution often makes
fluorescence structures more localized, which can increase this value.

How to read it:

A higher value means the signal is more concentrated. Unlike a
signal/background ratio, this metric remains bounded and does not explode when
the background or median intensity is near zero.

## Robust Range

Formula:

```text
Robust range = percentile(S_norm, 99.5) - percentile(S_norm, 0.5)
```

What it says:

Robust range measures the usable intensity spread while ignoring extreme
outliers. It helps explain whether the deconvolved image became more compressed
or more expanded in intensity.

How to read it:

This value can go up or down. A lower robust range can be reasonable when
background haze is removed or intensity is redistributed into compact
structures. It should not be treated as a direct quality score.

## Interpreting The Table

The console and GUI log show:

```text
Metric                  Source      Result      Change
```

`Change` is the ratio:

```text
Change = Result / Source
```

For example, `3.0x` means the metric is three times higher after
deconvolution. For most datasets, the most useful pattern is:

- detail energy increases,
- bright detail energy increases,
- edge strength stays stable or increases,
- signal sparsity increases moderately,
- robust range gives context for intensity redistribution.

These metrics are diagnostic aids. Final assessment should still include visual
inspection, knowledge of the biological structure, and, when available,
ground-truth or bead-based validation.

## References

These references support the metric families used here. They do not define this
exact CIDeconvolve table as a standardized image-quality score; the table is a
practical diagnostic summary for this workflow.

- Detail energy and bright detail energy use the idea that restoration and
  resolution changes can be inspected in the spatial-frequency domain. A closely
  related microscopy reference is Fourier ring/shell correlation for restoration
  and deconvolution progress:
  Koho, S. et al. (2019). *Fourier ring correlation simplifies image restoration
  in fluorescence microscopy*. Nature Communications, 10, 3103.
  DOI: [10.1038/s41467-019-11024-z](https://doi.org/10.1038/s41467-019-11024-z)

- The general deconvolution interpretation, especially improved contrast and
  detection of small dim objects in fluorescence microscopy, follows:
  Swedlow, J. R. (2007). *Quantitative fluorescence microscopy and image
  deconvolution*. Methods in Cell Biology, 81, 447-465.
  DOI: [10.1016/S0091-679X(06)81021-6](https://doi.org/10.1016/S0091-679X(06)81021-6)

- Edge strength is a gradient-magnitude focus/sharpness family metric. A broad
  review and comparison of focus-measure operators, including gradient-based
  operators, is:
  Pertuz, S., Puig, D., & Garcia, M. A. (2013). *Analysis of focus measure
  operators for shape-from-focus*. Pattern Recognition, 46(5), 1415-1432.
  DOI: [10.1016/j.patcog.2012.11.011](https://doi.org/10.1016/j.patcog.2012.11.011)

- Signal sparsity is a Gini-style concentration measure. For the origin and
  formulations of the Gini index:
  Ceriani, L., & Verme, P. (2012). *The origins of the Gini index: extracts from
  Variabilita e Mutabilita (1912) by Corrado Gini*. Journal of Economic
  Inequality, 10, 421-443.
  DOI: [10.1007/s10888-011-9188-x](https://doi.org/10.1007/s10888-011-9188-x)

- Robust range is based on empirical sample quantiles. A standard reference for
  sample quantile definitions in statistical software is:
  Hyndman, R. J., & Fan, Y. (1996). *Sample quantiles in statistical packages*.
  The American Statistician, 50(4), 361-365.
  DOI: [10.1080/00031305.1996.10473566](https://doi.org/10.1080/00031305.1996.10473566)

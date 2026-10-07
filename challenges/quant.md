# MicroQuant Challenges

Test your understanding of post-training quantization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Absmax INT8 With an Outlier

**Setup:** `quantize_absmax_int8` (lines 318-333) computes `scale = max_abs / 127.0` and maps each value to `round(val / scale)`, clipped to `[-127, 127]`. One scale serves the whole tensor.

**Question:** Suppose a weight matrix has values uniformly distributed in `[-0.5, 0.5]` except for one outlier at `10.0`. What is the absmax scale? What is the quantized representation of a typical value like `0.3`? Compare to quantizing the same matrix without the outlier.

<details>
<summary>Reveal Answer</summary>

**Answer:** With outlier: `scale = 10.0 / 127.0 ≈ 0.0787`. The value `0.3` quantizes to `round(0.3 / 0.0787) = round(3.81) = 4`, dequantizing to `4 * 0.0787 = 0.315` — error of `0.015`. Without outlier: `scale = 0.5 / 127.0 ≈ 0.00394`. The value `0.3` quantizes to `round(0.3 / 0.00394) = round(76.1) = 76`, dequantizing to `76 * 0.003937 = 0.29921` — error of `0.00079`. The outlier made the error on `0.3` 19 times larger (0.01496 versus 0.00079, computed with the script's own `quantize_absmax_int8` and `dequantize_absmax`).

**Why:** Absmax forces all 255 integer levels to cover the full range `[-max_abs, +max_abs]`. An outlier at `10.0` stretches this range to `[-10, 10]`, spreading those levels across a span 20x wider than necessary for 99.9% of the weights. The typical weight at `0.3` now only has access to the handful of integer levels near `4 * scale ≈ 0.3`, rather than the 76 levels it would use in the outlier-free case. This is the core failure mode of per-tensor absmax quantization on transformer weight matrices, which often have heavy-tailed distributions with occasional large outliers. The per-channel docstring (lines 398-399) makes the same point: "a single outlier weight forces the entire grid to be coarse."

**Script reference:** `03-systems/microquant.py`, lines 318-333 (quantize_absmax_int8), lines 398-399 (outlier sensitivity note), lines 462-479 (compute_roundtrip_error, the worst-case single-weight error), lines 680-705 (results table)

</details>

---

### Challenge 2: INT4 Asymmetric Range

**Setup:** `quantize_absmax_int4` (lines 336-353) sets `scale = max_abs / 7.0` (line 351) and clips to `[-8, 7]` (line 352), the 4-bit two's-complement range. Compare to INT8, which uses `scale = max_abs / 127.0` and clips to `[-127, 127]` (lines 331-332).

**Question:** INT4 has 16 possible values: why does the range `[-8, 7]` have one more negative value than positive? Given the scale on line 351, does this script ever produce the code `-8`?

<details>
<summary>Reveal Answer</summary>

**Answer:** Two's complement 4-bit signed integers represent `[-8, 7]`: the bit pattern `1000` is `-8` and `0111` is `+7`, because zero takes one of the non-negative patterns. This script never produces `-8`. Every weight satisfies `|w| <= max_abs`, so `w / scale` lies in `[-7, 7]` and rounds to at most 7 in magnitude; the clip at `-8` never fires. The script therefore uses 15 of the 16 codes, symmetrically.

**Why:** In N-bit two's complement the range is `[-(2^(N-1)), 2^(N-1) - 1]`; for N=4 that is `[-8, 7]`. Using the extra negative code would need an asymmetric scale such as `max_abs / 8` for negative values, which most symmetric quantizers skip so that `0` and `±max_abs` are represented exactly and the grid has no sign bias. Quantizing `[-1.0, 1.0, 0.5, -0.95]` with the script's function gives codes `[-7, 7, 4, -7]` with scale `1/7`: the grid has 15 levels, so the step is `2 * max_abs / 14`, not `2 * max_abs / 15`.

**Script reference:** `03-systems/microquant.py`, lines 336-353 (quantize_absmax_int4), lines 351-352 (scale and clip to [-8, 7]), lines 337-344 (INT4 docstring), lines 680-705 (results table, INT4 vs INT8)

</details>

---

### Challenge 3: Per-Channel vs Per-Tensor Accuracy

**Setup:** `quantize_per_channel_int8` (lines 393-413) computes a separate scale for each row: `max_abs = max(abs(w) for w in row)` and `scale = max_abs / 127.0`. `quantize_absmax_int8` uses a single global scale. The docstring explains why output channels benefit from independent scaling.

**Question:** Suppose a weight matrix has row 0 with values in `[-0.1, 0.1]` and row 1 with values in `[-5.0, 5.0]`. What is the per-tensor scale? What are the per-channel scales? Compute the roundtrip error for a value `0.05` in row 0 under each scheme.

<details>
<summary>Reveal Answer</summary>

**Answer:** Per-tensor scale: `5.0 / 127.0 ≈ 0.03937`. Value `0.05` → `round(0.05 / 0.03937) = round(1.27) = 1` → dequantized `0.03937` → error `|0.05 - 0.03937| = 0.0106`.

Per-channel scale for row 0: `0.1 / 127.0 ≈ 0.000787`. Value `0.05` → `0.05 / 0.000787` is 63.5 in exact arithmetic but 63.49999… in float64, so `round` gives `63` → dequantized `63 * 0.000787 = 0.04961` → error `0.00039`.

Per-channel reduces the error on this value 27x (0.01063 versus 0.00039, computed with the script's quantize and dequantize functions).

**Why:** Row 0's small values are compressed into just `round(0.1/0.03937) = ±2` integer levels under per-tensor scaling — only 5 distinct quantized values cover the entire row, causing massive quantization noise. Per-channel scaling allocates all 255 levels independently to each row's range. Since rows of weight matrices often correspond to different output neurons with very different activation scales (especially after layer normalization with learned scale parameters), per-channel quantization matches the scale to the actual data distribution. The tradeoff is that each row needs its own stored scale value — `N_rows` scale values instead of 1 — increasing metadata overhead by `N_rows * sizeof(float)` bytes.

**Script reference:** `03-systems/microquant.py`, lines 393-413 (quantize_per_channel_int8 with per-row scales), lines 318-333 (quantize_absmax_int8 for comparison), lines 462-479 (compute_roundtrip_error)

</details>

---

### Challenge 4: Zero-Point Quantization for Asymmetric Data

**Setup:** `quantize_zeropoint_int8` (lines 356-390) computes `scale = (w_max - w_min) / 255` and `zero_point = round(-w_min / scale)`, then `q = clamp(round(w / scale) + zero_point, 0, 255)`; `dequantize_zeropoint` returns `(q - zero_point) * scale`. The range covers `[0, 255]` (unsigned byte). The script applies it to weights; the question below applies the same function to activations.

**Question:** Suppose a ReLU activation layer produces values in `[0, 4.0]` (all non-negative). What zero-point does `quantize_zeropoint_int8` compute? Why would absmax INT8 (which covers `[-max_abs, max_abs]`) be particularly wasteful for this data?

<details>
<summary>Reveal Answer</summary>

**Answer:** `scale = (4.0 - 0.0) / 255.0 ≈ 0.01569`. `zero_point = round(-0.0 / 0.01569) = round(0) = 0`. The unsigned range `[0, 255]` maps exactly to `[0.0, 4.0]` with zero_point=0. Absmax INT8 would use `scale = 4.0 / 127.0 ≈ 0.03150` and cover `[-4.0, 4.0]`, but half of that range (negative values) is never used by ReLU activations — wasting half the representable range and giving 2x worse precision.

**Why:** Absmax INT8 is designed for symmetric distributions centered near zero (typical for weight matrices after training). Activations after ReLU are strictly non-negative, so the symmetric range `[-max, max]` wastes the 128 negative integer levels on values that never appear. Zero-point quantization (also called affine quantization) maps the data range `[min_val, max_val]` to the full unsigned range `[0, 255]`, using 256 levels instead of 128 to cover the same value range — effectively doubling precision. The `zero_point` parameter shifts the integer grid so that `q=0` corresponds to `val=min_val`, not to `val=0`. This is why asymmetric (zero-point) quantization is a common choice for non-negative activations such as ReLU outputs: it matches the actual data range rather than assuming symmetry. Running the script's function on `[0.0, 4.0, 1.0]` gives scale `4/255`, zero-point `0` and codes `[0, 255, 64]`.

**Script reference:** `03-systems/microquant.py`, lines 356-390 (quantize_zeropoint_int8), lines 425-429 (dequantize_zeropoint)

</details>

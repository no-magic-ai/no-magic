# MicroTurboQuant Challenges

Test your understanding of data-oblivious vector quantization by predicting what happens in these scenarios. Work out the answer before revealing it.

---

### Challenge 1: When Rotation Hurts Instead of Helps

**Setup:** `absmax_quantize` (lines 158-179) computes `scale = max(|x_i|) / levels` per vector. `turboquant_encode` (lines 188-200) rotates first, then absmax-quantizes the rotated coordinates. The comment on the `ANISOTROPY` constant (lines 42-49) notes that rotation helps at moderate anisotropy but loses once vectors become nearly 1-sparse.

**Question:** Suppose you quantize a perfectly 1-sparse unit vector `x = [1.0, 0.0, 0.0, ..., 0.0]` (1 non-zero coord, 31 zeros) at 4 bits. What does the baseline absmax quantization achieve? What does TurboQuant achieve? Why does one clearly win?

<details>
<summary>Reveal Answer</summary>

**Answer:** Baseline: `max(|x|) = 1.0`, `scale = 1.0 / 7`. The non-zero coordinate quantizes to `7` and dequantizes to exactly `1.0`; the zeros stay exactly `0`. The reconstruction error is exactly zero. TurboQuant: `y = R @ x` is the first column of `R`, so all 32 coordinates are non-zero with typical magnitude `1/sqrt(32) ≈ 0.18`. With a rotation drawn by `random_rotation(32)` right after `random.seed(42)`, the largest `|y_i|` is 0.449, the scale is 0.064, and rotating back gives a squared reconstruction error of 0.0106 (about what `32 * scale² / 12` predicts for rounding errors spread over 32 coordinates). Rotation loses on this vector, from an exact result to a small nonzero error.

**Why:** Absmax on a 1-sparse vector spends its whole grid on the single non-zero coordinate, and the 31 zeros cost nothing. The dense rotation turns "one exact non-zero, 31 exact zeros" into "32 values that each need rounding", and the inverse rotation carries all of that rounding error back. This does not contradict the paper. TurboQuant's guarantee is a worst-case bound: after a random rotation every unit vector, sparse or dense, has the same coordinate distribution, so the expected error is the same for all inputs. It does not promise to beat a quantizer that happens to be exact on a particular vector. The script also uses per-vector absmax after rotation, not the paper's Lloyd–Max codebooks for the rotated-coordinate distribution.

**Script reference:** `03-systems/microturboquant.py`, lines 42-49 (ANISOTROPY sweet-spot comment), lines 158-179 (absmax_quantize), lines 188-200 (turboquant_encode), lines 246-295 (sample_name_embeddings with its note on sparse vectors)

</details>

---

### Challenge 2: Why Gram-Schmidt on Gaussians Produces a Uniform Rotation

**Setup:** `random_rotation` (lines 109-142) fills a D-by-D matrix with i.i.d. standard-Gaussian entries, then orthonormalizes columns via Gram-Schmidt. The docstring claims the result is distributed as Haar measure on O(D) — uniform over all orthogonal matrices.

**Question:** What property of the Gaussian distribution makes this work? What would go wrong if you replaced `gaussian_sample()` with `random.uniform(-1, 1)` on line 124?

<details>
<summary>Reveal Answer</summary>

**Answer:** The multivariate standard Gaussian is **rotation-invariant**: if `g ~ N(0, I_D)`, then `R g ~ N(0, I_D)` for any orthogonal `R`. This means every direction on the unit sphere is equally likely to be the normalized first Gaussian column, so after Gram-Schmidt the first column is uniform on the sphere — and by induction, the whole orthonormal basis is Haar-uniform. Uniform(-1, 1) is NOT rotation-invariant: the distribution is a hypercube, not a ball. A uniform point in the cube is more likely to point toward a corner than along an axis, because the cube reaches out to `sqrt(D)` along a diagonal but only to `1` along an axis, so more of its volume lies in diagonal directions. Gram-Schmidt on such columns still produces an orthogonal matrix, but not a uniformly random one.

**Why:** The key property is that the Gaussian density `exp(-||x||^2/2)` depends only on `||x||`, not on direction. Any distribution with this radial symmetry works — Gaussian is the simplest and the one with the cleanest sampling primitive in `random`. The uniform hypercube's density is constant on `[-1, 1]^D` and zero outside, so the set of allowed lengths depends on direction: along an axis a point can have length at most 1, along the main diagonal up to `sqrt(D)`. Integrating over length, diagonal directions collect more probability, so the sampled direction is NOT uniform on the sphere. For TurboQuant, a non-uniform rotation breaks the data-oblivious guarantee: the MSE bound derived in the paper assumes Haar-uniform R, and any other distribution gives data-dependent worst-case performance.

**Script reference:** `03-systems/microturboquant.py`, lines 109-142 (random_rotation via Gram-Schmidt), lines 101-104 (gaussian_sample helper), lines 145-155 (orthogonality_error check), lines 414-419 (main asserts orthogonality < 1e-10)

</details>

---

### Challenge 3: QJL Signed-Bit Inner-Product Estimator

**Setup:** `qjl_signs` (lines 212-221) stores `K` sign bits per vector: `sign((S @ x)_k)` for each row of a fixed Gaussian `S`. `qjl_estimate_inner_product` (lines 224-241) returns `agreement * pi / 2.0` where `agreement` is the mean of `sign_a[k] * sign_b[k]` across the K sign bits. The comment cites the identity `E[sign(<g,a>) * sign(<g,b>)] = 1 - 2 * arccos(rho) / pi` for Gaussian `g`.

**Question:** Two vectors have cosine similarity `rho = 0.5`. Using the exact identity, what is `E[agreement]`? Using the linear approximation `rho ~ (pi/2) * agreement`, what does the estimator return? How much bias does the linear approximation introduce, and when does it matter?

<details>
<summary>Reveal Answer</summary>

**Answer:** Exact: `E[agreement] = 1 - 2 * arccos(0.5) / pi = 1 - 2 * (pi/3) / pi = 1 - 2/3 = 0.333`. Linear estimator: `rho_est = 0.333 * pi / 2 ≈ 0.523`. The true `rho` is `0.5`, so the linear estimate overshoots by `0.023` — a `+4.7%` relative bias. For near-orthogonal vectors (`rho` near 0) the bias vanishes: `arccos(0) = pi/2`, so `E[agreement] = 0` and the linear estimator returns exactly 0. The bias grows as `rho` moves toward `±1`: in general the estimator's expectation is `(pi/2) * (1 - 2 * arccos(rho) / pi) = arcsin(rho)`, so identical vectors give `pi/2 ≈ 1.571` instead of 1. With `rho = 0.5` and 40,000 projections, the script's own functions returned 0.526, against `arcsin(0.5) = 0.524`.

**Why:** Inverting the identity gives `rho = cos(pi/2 - pi/2 * agreement) = sin(pi/2 * agreement)`; the script drops the sine (`sin(x) ≈ x`), which is exact only at `x = 0`. Applying `sin` to the measured agreement would remove most of the bias, but the result would still not be exactly unbiased, because `sin` of a sample mean is not the mean of `sin`. This paired-sign estimator, in which both vectors are reduced to signs, is not the paper's method. The TurboQuant paper (Zandieh, Daliri, Hadian and Mirrokni, arXiv:2504.19874, Definition 1 and Lemma 4) keeps the query `y` at full precision and only quantizes the stored vector: it stores `sign(S · x)` for a Gaussian matrix `S` of shape `d × d` and estimates `⟨y, sqrt(pi/2)/d · Sᵀ · sign(S · x)⟩`, which is unbiased for unit `x` with variance at most `pi/(2d) · ||y||²`. The divisor is the number of rows of `S`, which the paper takes equal to the dimension `d`; with `m` rows it is `sqrt(pi/2)/m` (256, not 32, for the script's 256 × 32 projection matrix). Its inner-product quantizer applies this QJL step to the residual left by a `(b − 1)`-bit MSE quantizer and stores the residual's norm. The script's docstring calls the paired-sign form "unbiased" and says production QJL "inverts the arccos"; both statements are incorrect.

**Script reference:** `03-systems/microturboquant.py`, lines 212-221 (qjl_signs: sign bits of Gaussian projection), lines 224-241 (qjl_estimate_inner_product with pi/2 scaling), lines 369-392 (qjl_demo: empirical mean signed error on whole vectors), lines 225-237 (docstring with the identity and signpost)

</details>

---

### Challenge 4: The Orthogonality-Error Assertion

**Setup:** `main` (lines 414-419) calls `random_rotation(32)` then asserts `orthogonality_error(R) < 1e-10`. The error metric is `max |R^T R - I|` across all `D*D` entries (lines 145-155).

**Question:** Why is `1e-10` the threshold and not `0`? Why is it not `1e-15` (machine epsilon for f64)? And what would go wrong in `turboquant_decode` (lines 202-209) if the assertion were replaced with `< 1e-3` and silently passed a slightly non-orthogonal matrix?

<details>
<summary>Reveal Answer</summary>

**Answer:** `1e-10` leaves room for floating-point error in Gram-Schmidt. Each inner product and subtraction introduces relative error on the order of `eps = 2^-52 ≈ 2.2e-16`. For five random 32×32 draws (seeds 0-4) the script's `orthogonality_error` measured between `9.3e-16` and `2.7e-15`, and a full run of the script printed `8.80e-14`, so `1e-15` would fail on typical draws while `1e-10` leaves three to five orders of magnitude of headroom; the margin covers poorly conditioned Gaussian draws, where Gram-Schmidt loses orthogonality faster. The loop updates `col` before computing each next coefficient (`coef = inner(col, prev)`), which is the modified Gram-Schmidt variant; the classical variant, which projects the original column, loses orthogonality faster on ill-conditioned inputs. If the threshold were `1e-3`: `R^T R ≈ I + epsilon_matrix` where `||epsilon|| < 1e-3`. Then `turboquant_decode` returns `R^T y_hat ≈ R^T (R x + quant_error) = (I + epsilon) x + R^T quant_error`. The `epsilon * x` term is a systematic reconstruction bias independent of quantization — a "rotation-back-is-wrong" error that would appear at every bit-width including 32-bit, making the IP-MSE non-zero even before any bits are spent.

**Why:** The entire TurboQuant guarantee depends on `R^T R = I` exactly, because this is what ensures `||x_hat - x||_2 = ||y_hat - y||_2` (rotation preserves L2 norm of any vector, including the error vector). A non-orthogonal `R` breaks this isometry: the quantization error is no longer spread uniformly across coordinates, and the MSE bound from the paper's analysis no longer applies. The hard assertion enforces this invariant at construction time, so later code can trust it without re-checking. The threshold also catches degenerate Gaussian draws (a column nearly in the span of earlier columns gets amplified by Gram-Schmidt renormalization), which is why `random_rotation` raises an exception on line 138 rather than returning a questionable matrix.

**Script reference:** `03-systems/microturboquant.py`, lines 109-142 (random_rotation with degeneracy check at line 138), lines 145-155 (orthogonality_error: max entry-wise deviation from I), lines 202-209 (turboquant_decode applies R^T assuming exact orthogonality), lines 414-419 (main asserts < 1e-10 before using R)

</details>

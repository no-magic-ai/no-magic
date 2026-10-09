# Discretization Challenges

Test your understanding of SSM discretization methods by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Euler Stability Boundary

**Setup:** The stability analysis table at the end of the script compares Euler vs ZOH discretization. Euler computes `A_bar = 1 + delta * a_n`. The stability condition requires `|A_bar| < 1`. With `a_n = -0.5` (from the stability analysis), consider three step sizes: `delta = 3.0`, `delta = 4.0`, and `delta = 5.0`.

**Question:** What is `|A_bar|` for Euler at each delta? At what delta does Euler hit the stability boundary? What does ZOH compute for `delta = 5.0`?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- `delta = 3.0`: `|1 + 3.0 * (-0.5)| = |1 - 1.5| = |-0.5| = 0.5` -- stable.
- `delta = 4.0`: `|1 + 4.0 * (-0.5)| = |1 - 2.0| = |-1.0| = 1.0` -- marginal stability boundary. The state oscillates in sign each step but neither grows nor decays. The script's table tests `|A_bar| < 1` strictly, so it labels this row `NO — DIVERGES`.
- `delta = 5.0`: `|1 + 5.0 * (-0.5)| = |1 - 2.5| = |-1.5| = 1.5 > 1` -- diverges. Each step amplifies the state by 1.5x while flipping sign.

The stability boundary is at `delta = -2/a_n = -2/(-0.5) = 4.0`. Any `delta > 4.0` causes Euler to diverge.

ZOH at `delta = 5.0`: `exp(5.0 * (-0.5)) = exp(-2.5) ≈ 0.082` -- unconditionally stable regardless of step size.

**Why:** Euler's linear approximation `1 + delta * a_n` can overshoot into negative territory and exceed magnitude 1.0, creating oscillatory divergence. The ZOH exponential `exp(delta * a_n)` maps the continuous-time stable region (negative real axis) into the discrete-time stable region (unit disk) exactly, preserving stability for any step size. This is why production SSMs avoid forward Euler for the state: S4 uses the bilinear (Tustin) method, which also maps the stable half-plane into the unit disk (the comment on lines 264-267 of `microssm.py` notes the S4 choice), and Mamba keeps ZOH's state factor `exp(delta * A)`. The Mamba-1 paper describes its discretization as ZOH, but the Mamba-3 paper (arXiv:2603.15569, Section 3.1) points out that Mamba-1 and Mamba-2 implement exponential-Euler: the same `exp(delta * A)` state factor with the input term simplified to `delta * B`.

**Script reference:** `03-systems/microdiscretize.py`, lines 322-349 (`euler_discretize`), lines 744-783 (stability analysis table)

</details>

---

### Challenge 2: Trapezoidal Alpha and Method Reduction

**Setup:** The trapezoidal discretization splits the input coefficient between the current input `x_t` (weight `alpha`) and the previous input `x_{t-1}` (weight `1 - alpha`). With `alpha = 0.5`, the recurrence uses both: `h_t = A_bar * h_{t-1} + B_bar * (alpha * x_t + (1 - alpha) * x_{t-1})`.

**Question:** If you set `alpha = 1.0`, what does trapezoidal reduce to? If `alpha = 0.0`? Why is `alpha = 0.5` the choice that replaces Mamba-1/2's explicit short convolution?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- `alpha = 1.0`: All weight on `x_t`, zero weight on `x_{t-1}`. The `x_{t-1}` dependency vanishes, and the script's rule recovers its exact ZOH discretization -- a purely causal recurrence with no implicit look-back. (In Mamba-3's own rule the corresponding setting, `λ = 1`, recovers Mamba-2's exponential-Euler instead, because its input coefficient is `delta`, not the ZOH integral.)
- `alpha = 0.0`: All weight on `x_{t-1}`, zero weight on `x_t`. The method becomes fully implicit -- the current input has no direct effect, and the system responds with a one-step delay.

**Why:** The `alpha = 0.5` split creates an implicit dependence on `x_{t-1}` within the recurrence itself: the input enters through a 2-tap filter over adjacent inputs. The script's docstring (lines 395-415), citing Mamba-3 (arXiv:2603.15569), presents this as the reason Mamba-3 can drop the separate short convolution (`conv1d`) that Mamba-1 and Mamba-2 place before the SSM. The paper's rule differs from the script's: it uses `h_t = e^(Δ·A) h_{t-1} + (1 − λ)·Δ·e^(Δ·A)·B_{t-1} x_{t-1} + λ·Δ·B_t x_t` with a data-dependent `λ`, while the script splits the exact ZOH input integral with a fixed `alpha`. The paper also reports dropping the short convolution only in combination with added B and C biases (its ablation finds the two work together), so the size-2 filter alone does not show that the convolution is redundant.

**Script reference:** `03-systems/microdiscretize.py`, lines 395-447 (`trapezoidal_discretize`, the alpha parameter and the x\_{t-1} weighting)

</details>

---

### Challenge 3: Why ZOH Is Unconditionally Stable

**Setup:** ZOH computes `A_bar = exp(delta * a_n)`. For the system to be stable in discrete time, we need `|A_bar| < 1`. Assume the continuous-time system is stable: `a_n < 0` and `delta > 0`.

**Question:** Can `exp(delta * a_n)` ever exceed 1.0? Can it ever be negative? How does Euler's `1 + delta * a_n` compare on both counts?

<details>
<summary>Reveal Answer</summary>

**Answer:** No and no. `exp(delta * a_n)` with `delta * a_n < 0` is always in the open interval `(0, 1)`:

- It cannot exceed 1.0 because `exp(x) < 1` for all `x < 0`.
- It cannot be negative because `exp(x) > 0` for all real `x`.

Euler's `1 + delta * a_n` fails on both counts:

- It becomes negative when `delta > -1/a_n` (e.g., `delta > 2.0` for `a_n = -0.5`).
- Its magnitude exceeds 1.0 when `delta > -2/a_n` (e.g., `delta > 4.0` for `a_n = -0.5`).

**Why:** The matrix exponential is the mathematically exact solution to the continuous-time ODE, sampled at discrete points. It maps the stable continuous-time half-plane (`Re(s) < 0`) perfectly into the stable discrete-time unit disk (`|z| < 1`). Euler is a first-order Taylor approximation of this exponential (`exp(x) ≈ 1 + x`), and like all polynomial approximations of the exponential, it diverges for large arguments. This is not a minor numerical issue -- it is a fundamental limitation that makes Euler unusable for large step sizes in SSMs.

**Script reference:** `03-systems/microdiscretize.py`, lines 350-394 (`zoh_discretize`), lines 744-783 (stability comparison output)

</details>

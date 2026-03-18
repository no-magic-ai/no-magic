# MicroOptimizer Challenges

Test your understanding of optimization algorithms by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Adam with Beta1=0 and Beta2=0

**Setup:** The `step_adam` function (line 365) computes:

- `m = beta1 * m + (1 - beta1) * grad` (first moment, line 402)
- `v = beta2 * v + (1 - beta2) * grad^2` (second moment, line 404)
- `m_hat = m / (1 - beta1^t)` (bias correction, line 407)
- `v_hat = v / (1 - beta2^t)` (bias correction, line 408)
- `param -= lr * m_hat / (sqrt(v_hat) + eps)` (update, line 411)

**Question:** If you set `ADAM_BETA1 = 0` and `ADAM_BETA2 = 0`, what do `m`, `v`, `m_hat`, and `v_hat` simplify to? What optimizer does Adam reduce to?

<details>
<summary>Reveal Answer</summary>

**Answer:** `m = grad`, `v = grad^2`. The bias correction becomes `m_hat = grad / (1 - 0) = grad` and `v_hat = grad^2 / (1 - 0) = grad^2`. The update is `lr * grad / (|grad| + eps)`, which is sign-based gradient descent: every parameter moves by approximately `lr` in the direction of the gradient's sign (ignoring eps).

**Why:** With `beta1 = 0`, the first moment has no memory -- it equals the current gradient exactly. With `beta2 = 0`, the second moment equals the current squared gradient. The ratio `m_hat / sqrt(v_hat) = grad / sqrt(grad^2) = grad / |grad| = sign(grad)` (for `grad != 0`). This is SignSGD: every parameter gets the same magnitude update `lr`, differing only in direction. All adaptive scaling and momentum are gone. The epsilon prevents division by zero when the gradient is exactly 0, in which case the update is `lr * 0 / (0 + eps) = 0`.

**Script reference:** `01-foundations/microoptimizer.py`, lines 365-412 (`step_adam`), 53-55 (default ADAM_BETA1=0.9, ADAM_BETA2=0.999)

</details>

---

### Challenge 2: Why Bias Correction Matters Early

**Setup:** At step `t=1` with `ADAM_BETA1 = 0.9` and `ADAM_BETA2 = 0.999`, bias correction divides by `(1 - beta^t)`:

- `m_hat = m / (1 - 0.9^1) = m / 0.1 = 10 * m`
- `v_hat = v / (1 - 0.999^1) = v / 0.001 = 1000 * v`

**Question:** At step 1, `m = 0.1 * grad` and `v = 0.001 * grad^2`. After bias correction, what is the effective update? How does this compare to step 100, where `1 - 0.9^100 ~ 1.0` and `1 - 0.999^100 ~ 0.0952`?

<details>
<summary>Reveal Answer</summary>

**Answer:** At step 1: `m_hat = 0.1*grad / 0.1 = grad`, `v_hat = 0.001*grad^2 / 0.001 = grad^2`. The update is `lr * grad / (|grad| + eps) ~ lr * sign(grad)`. At step 100: `m_hat ~ m / 1.0 = m` (the running average itself), `v_hat ~ v / 0.0952 ~ 10.5 * v`. Bias correction at step 100 is minor for `m` but still significant for `v` (10.5x amplification) because `beta2 = 0.999` is very close to 1.

**Why:** The exponential moving averages `m` and `v` are initialized at 0. In early steps, they haven't accumulated enough gradient history and are biased toward zero. Bias correction on lines 407-408 compensates: at step 1, it exactly undoes the initialization bias, recovering the actual gradient. The correction factor `1/(1-beta^t)` is largest when `t` is small and converges to 1 as `t` grows. For `beta2 = 0.999`, convergence is slow (the effective window is ~1000 steps), so `v_hat` remains significantly amplified for hundreds of steps. Without bias correction (lines 380-384), the early updates would use `m ~ 0.1*grad` and `v ~ 0.001*grad^2`, producing updates far too small to make progress.

**Script reference:** `01-foundations/microoptimizer.py`, lines 379-384 (comment explaining why correction matters), 402-411 (the actual computation)

</details>

---

### Challenge 3: Constant Gradient Across All Steps

**Setup:** Suppose the gradient for a particular parameter is exactly `g = 0.5` at every single training step. The four optimizers are SGD (line 281), Momentum (line 303), RMSProp (line 333), and Adam (line 365).

**Question:** After many steps, what is the effective per-step parameter change for each optimizer? Which converges fastest to a steady-state update magnitude?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- **SGD:** Update = `lr * g = 0.05 * 0.5 = 0.025` per step. Constant from step 1.
- **Momentum:** Velocity converges to `g / (1 - beta) = 0.5 / 0.1 = 5.0`. Update = `lr * 5.0 = 0.25` per step. 10x larger than SGD. Converges in ~10 steps (`1/(1-0.9)`).
- **RMSProp:** `sq_avg` converges to `g^2 = 0.25`. Update = `lr * g / sqrt(g^2 + eps) = 0.01 * 0.5 / 0.5 = 0.01` per step. Converges in ~100 steps (`1/(1-0.99)`).
- **Adam:** `m` converges to `g = 0.5`, `v` converges to `g^2 = 0.25`. Update = `lr * g / (|g| + eps) = 0.01 * 0.5 / 0.5 = 0.01` per step. Converges in ~1000 steps (limited by `beta2 = 0.999`).

**Why:** Momentum amplifies consistent gradients by accumulating velocity -- the denominator `(1 - beta)` appears because the geometric series `g + beta*g + beta^2*g + ...` sums to `g/(1-beta)`. This is noted on lines 316-317: "At beta=0.9, the effective window is ~10 past gradients." RMSProp and Adam normalize the gradient by its own magnitude, so for a constant gradient, the adaptive scaling cancels out: `g / sqrt(g^2) = sign(g)`. The update magnitude becomes `lr` regardless of gradient scale. Adam converges to steady state slowest because `beta2 = 0.999` means the second moment averages over ~1000 steps.

**Script reference:** `01-foundations/microoptimizer.py`, lines 281-300 (`step_sgd`), 303-330 (`step_momentum`, esp. line 317), 333-362 (`step_rmsprop`), 365-412 (`step_adam`)

</details>

---

### Challenge 4: Learning Rate Warmup Purpose

**Setup:** The script includes an Adam variant with warmup + cosine decay (referenced on lines 60-62): `WARMUP_STEPS = 20`, linearly ramping lr from 0 to `COSINE_LR = 0.01`.

**Question:** Why start with a learning rate of zero and ramp up? Adam already has bias correction to handle early steps -- why isn't that sufficient?

<details>
<summary>Reveal Answer</summary>

**Answer:** Bias correction fixes the magnitude of the moment estimates, but it cannot fix the direction quality. In early steps, `m` and `v` are based on very few gradient samples, so the corrected estimates are noisy and unreliable. Warmup limits damage from these unreliable early updates by keeping the learning rate small while the moment estimates stabilize.

**Why:** Consider step 1: bias correction gives `m_hat = grad` and `v_hat = grad^2`, which are the raw gradient from a single mini-batch. This single-sample estimate of the gradient direction and curvature is high-variance. A full learning rate step in this noisy direction can push parameters into a bad region of the loss landscape, from which recovery is expensive. Warmup acts as a safety buffer: during the first 20 steps, the model takes tiny steps (linearly increasing from 0 to full lr), allowing `m` and `v` to accumulate over multiple batches before the optimizer commits to large updates. This is especially important for transformers, where early attention patterns are random and large updates can cause attention entropy collapse.

**Script reference:** `01-foundations/microoptimizer.py`, lines 60-62 (warmup/cosine parameters), 379-384 (bias correction rationale)

</details>

---

### Challenge 5: RMSProp vs Adam on Noisy Gradients

**Setup:** RMSProp (line 333) and Adam (line 365) both divide by `sqrt(v + eps)` where `v` tracks squared gradients. Adam adds momentum (`m`) on top. Consider a parameter where the gradient alternates between `+1` and `-1` every step.

**Question:** After many steps, what is the effective update for RMSProp? For Adam? Which one handles oscillating gradients better?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- **RMSProp:** `sq_avg` converges to `1.0`. Each step: `lr * grad / sqrt(1.0 + eps) ~ lr * grad = +/- 0.01`. The parameter oscillates by `+/-0.01` with zero net progress.
- **Adam:** `m` converges to approximately 0 (positive and negative gradients cancel). `v` converges to `1.0`. Update: `lr * ~0 / sqrt(1.0 + eps) ~ 0`. Adam effectively stops updating this parameter.

**Why:** Momentum in Adam averages the gradient direction over time. Alternating `+1/-1` produces a mean near zero, signaling that this parameter has no consistent gradient direction. Adam correctly suppresses updates for ambiguous parameters. RMSProp has no such mechanism -- it faithfully scales each gradient by the running RMS, so it oscillates forever. In practice, oscillating gradients often indicate saddle points or loss landscape ridges. Adam's momentum-based dampening lets it allocate its learning rate budget to parameters with consistent gradient signals. This is the core advantage of combining momentum (direction averaging) with adaptive scaling (magnitude normalization).

**Script reference:** `01-foundations/microoptimizer.py`, lines 333-362 (`step_rmsprop`), 365-412 (`step_adam`), 370-388 (Adam rationale: "combining the best of both worlds")

</details>

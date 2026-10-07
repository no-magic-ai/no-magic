# MicroDPO Challenges

Test your understanding of Direct Preference Optimization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Beta Equals Zero

**Setup:** The DPO loss (line 442) computes `delta = beta * (log_ratio_chosen - log_ratio_rejected)` (line 487). The loss is `-log(sigmoid(delta))`, implemented as `log(1 + exp(-delta))` for numerical stability (lines 494-500). The default `DPO_BETA = 0.1` (line 50).

**Question:** If you set `beta = 0`, what does `delta` become? What is the resulting loss value? What happens to the gradients flowing back through the policy model?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = 0` for every preference pair. The loss becomes `log(1 + exp(0)) = log(2) = 0.693` for every pair, regardless of model behavior. Gradients through the policy are zero because the loss is a constant with respect to model parameters.

**Why:** The `beta` multiplier on line 487 scales the entire preference signal. At `beta = 0`, the implicit reward difference is always zero: the model cannot distinguish preferred from rejected completions. The loss `log(2)` is the maximum-entropy baseline -- equivalent to a random coin flip between preferred and rejected. Since the loss doesn't depend on any policy log-probabilities (the `beta * ...` term zeroes them out), no gradient flows back through `log_ratio_chosen` or `log_ratio_rejected` to update the model weights. The policy stays frozen at the reference model. Only `beta = 0` exactly has this effect: for any small positive `beta` the gradient is nonzero, and because Adam divides each update by the running gradient scale, the first steps are about as large as with the default. The comment on lines 50-55 says a low beta "barely moves the policy"; in the DPO paper beta weights the KL penalty, so a small beta permits more drift from the reference, not less (see Challenge 4).

**Script reference:** `02-alignment/microdpo.py`, lines 50-55 (beta definition and comment), 487 (delta computation), 494-500 (loss computation)

</details>

---

### Challenge 2: Identical Preferred and Rejected Completions

**Setup:** The DPO loss computes log-probability ratios for both the chosen and rejected sequences (lines 472-482). The preference signal depends on `log_ratio_chosen - log_ratio_rejected` (line 487).

**Question:** If the chosen and rejected completions are the exact same token sequence, what happens to `delta`? What does the loss become?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = 0`, and the loss is `log(2) = 0.693` -- identical to the `beta = 0` case. No learning occurs.

**Why:** If `chosen_tokens == rejected_tokens`, then `log_pi_chosen == log_pi_rejected` (same sequence through the same model) and `log_ref_chosen == log_ref_rejected` (same sequence through the same reference). Therefore `log_ratio_chosen == log_ratio_rejected`, their difference is zero, `delta = beta * 0 = 0`, and the loss is `log(1 + exp(0)) = log(2)`. The model receives no signal about which direction to move. This makes intuitive sense: if the "preferred" and "rejected" examples are identical, there is no preference to learn from. In production DPO, this is a data quality issue -- identical pairs are filtered out during preprocessing.

**Script reference:** `02-alignment/microdpo.py`, lines 471-487 (log-probability computation and delta), 395-414 (`sequence_log_prob_policy` showing both sequences go through the same computation)

</details>

---

### Challenge 3: Reference Model Already Prefers the Chosen Response

**Setup:** The log-ratio `log(pi/pi_ref)` measures divergence from the reference (line 480-482). The DPO loss pushes this ratio up for chosen sequences and down for rejected ones.

**Question:** If the reference model already assigns 10x higher probability to the chosen completion than the rejected one (i.e., `log_ref_chosen - log_ref_rejected = log(10) = 2.3`), does DPO still learn anything? How does the initial loss compare to a case where the reference assigns equal probability to both?

<details>
<summary>Reveal Answer</summary>

**Answer:** DPO learns in the same way in both cases. The initial loss is `log(2) = 0.693` in both, and the first gradient is the same function of the policy's log-probabilities, because the reference's preference cancels out of the margin.

**Why:** At initialization, the policy equals the reference, so `log_ratio_chosen = log_ratio_rejected = 0`, `delta = 0` and the loss is `log(2)` regardless of the reference's preferences. More generally, `delta = beta * [(log π(y_w) − log π(y_l)) − (log π_ref(y_w) − log π_ref(y_l))]`: DPO rewards the policy only for preferring the chosen completion by more than the reference already does. A reference that prefers the chosen completion 10:1 therefore gives no head start, and a reference that prefers the rejected one is no obstacle; the term `log π_ref(y_w) − log π_ref(y_l)` is a constant offset that the policy's own gap must exceed. This is how the reference anchors the policy: preferences are measured as changes relative to it.

**Script reference:** `02-alignment/microdpo.py`, lines 460-466 (log-ratio interpretation), 480-487 (delta computation), 199-211 (`snapshot_weights` creating the frozen reference)

</details>

---

### Challenge 4: Very Large Beta

**Setup:** `DPO_BETA` is the "inverse temperature of the implicit reward model" (line 55). The default is 0.1. The loss is `log(1 + exp(-beta * (log_ratio_chosen - log_ratio_rejected)))`.

**Question:** If you set `DPO_BETA = 100`, what happens to the loss landscape? What behavior would you expect from the trained model?

<details>
<summary>Reveal Answer</summary>

**Answer:** The loss becomes very sensitive to small log-ratio differences and saturates almost immediately: once the policy's margin over the reference passes about 0.05 nats, `delta = 100 * 0.05 = 5` and the loss is already `log(1 + e⁻⁵) ≈ 0.0067`, so its gradient nearly vanishes. Expect the policy to stop close to the reference, not to collapse. That is the DPO paper's reading of beta as the weight on the KL penalty; the comment on lines 50-55, which says a high beta "aggressively reshapes the distribution" and risks mode collapse, has the direction reversed.

**Why:** With `beta = 100`, the `delta` term amplifies log-ratio differences by 100x: a gap of 0.01 gives `delta = 1.0` instead of `0.001`. The gradient of the loss is `beta * sigmoid(-delta)` times the difference of the policy's log-probability gradients, and `sigmoid(-delta)` falls toward 0 as soon as `delta` is a few units, so a large beta reaches that point after a tiny change in the policy. Adam's per-parameter normalization makes the first steps about the same size for any beta, but with a large beta the push fades much sooner. The paper derives the loss from maximizing reward minus `beta` times the KL divergence from the reference, whose optimum is `π_ref(y|x) * exp(r(x, y) / beta)` up to normalization: large beta keeps the policy near the reference and small beta lets it move further. Changing `DPO_BETA` and comparing the reference and aligned lengths the script prints is a direct way to check this on names.

**Script reference:** `02-alignment/microdpo.py`, lines 50-55 (beta definition and mode collapse warning), 487 (beta scaling in delta), 489-500 (loss computation and stability)

</details>

---

### Challenge 5: The Numerical Stability Guard

**Setup:** The DPO loss uses a stability check on line 496: `if neg_delta.data > 20.0`, it uses `neg_delta` directly instead of `log(1 + exp(neg_delta))`.

**Question:** Why is the threshold 20? What would happen without this guard when `neg_delta = 100`? Does this approximation affect gradients?

<details>
<summary>Reveal Answer</summary>

**Answer:** At `neg_delta = 100` nothing would go wrong: `exp(100) ≈ 2.69e43` is a finite float and `log(1 + 2.69e43) = 100.0`. The guard matters above about 709.78, where Python's `math.exp` (used by `Value.exp`, line 130) raises `OverflowError` and the run stops. The threshold 20 is where the approximation `log(1 + exp(z)) ≈ z` becomes harmless: the error is `log(1 + exp(-z)) ≈ exp(-20) ≈ 2.1e-9`. The fallback branch `loss = neg_delta` has gradient exactly 1 with respect to `neg_delta`, while the exact gradient is `sigmoid(20) ≈ 1 - 2.1e-9`.

**Why:** For `z > 20`, `exp(z) > 4.85e8`, so `log(1 + exp(z))` and `z` differ by less than `2.1e-9` — not below float64 resolution (about `3.6e-15` near 20), but far too small to matter for training. The autograd graph through `neg_delta` (which is `-delta`, which is `-beta * (log_ratio_chosen - log_ratio_rejected)`) propagates this unit gradient back through the policy log-probabilities. Without the guard, any pair whose `neg_delta` exceeded about 709.78 would make `math.exp` raise `OverflowError`; Python floats do not silently become `inf` here.

**Script reference:** `02-alignment/microdpo.py`, lines 489-500 (stability guard with comment explaining the logsigmoid identity)

</details>

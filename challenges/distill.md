# MicroDistill Challenges

Test your understanding of knowledge distillation by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: What Temperature Reveals

**Setup:** `softmax` (lines 153-155) computes `exp(z_i/T) / sum_j exp(z_j/T)` through the stabilized `log_softmax` (lines 139-150). The teacher's soft targets use `TEMPERATURE = 2.0` (line 57); inference uses T = 1 (lines 279-282). Suppose the teacher produces logits `z = [4, 1, 0]` for a point of class 0.

**Question:** What are the class probabilities at T = 1 and at T = 2? Which one tells the student more about the two wrong classes?

<details>
<summary>Reveal Answer</summary>

**Answer:** At T = 1 the probabilities are about `[0.936, 0.047, 0.017]`; at T = 2 they are about `[0.736, 0.164, 0.100]`. The T = 2 target carries more information about the wrong classes: class 1 is still about 1.6 times as likely as class 2, but both are now large enough to produce a real gradient.

**Why:** Dividing by T scales every logit gap down, here from 3 and 4 to 1.5 and 2, so the winner's share falls and the losers' shares rise while their order stays the same. At T = 1, a loss term weighted by 0.017 barely moves the student, so the ranking of wrong classes is mostly ignored. Hinton et al. call this ranking the "dark knowledge" a one-hot label throws away. The values above came from the script's own `softmax` with these logits. Temperature changes only the training targets. Both models are evaluated at T = 1.

**Script reference:** `02-alignment/microdistill.py`, lines 139-155 (`log_softmax`, `softmax`), line 57 (`TEMPERATURE`), lines 279-282 (`predict` at T = 1)

</details>

---

### Challenge 2: Removing the T²

**Setup:** `distillation_objective` (lines 235-273) adds `ALPHA * t * t * kl + (1 - ALPHA) * ce` per point (line 262). The matching logit gradient (lines 264-270) is `ALPHA * t * (q_T - p_T) + (1 - ALPHA) * (q_1 - onehot(y))`, divided by the batch size.

**Question:** If the objective dropped the `t * t` factor (and its gradient became consistent with that), by what factor would the soft-target gradient change at T = 2? What would that do to the balance between soft and hard targets set by `ALPHA = 0.9`?

<details>
<summary>Reveal Answer</summary>

**Answer:** The soft-target gradient would shrink by exactly T² = 4. The hard-label term would then carry much more relative weight than `ALPHA = 0.9` suggests.

**Why:** The derivative of `KL(p_T || softmax(z/T))` with respect to `z_i` is `(q_i - p_i) / T`. Multiplying by T² turns this into `T (q_i - p_i)`, the expression on lines 266-267. Without T², the soft term's gradient is `ALPHA (q_i - p_i) / T`, smaller by T² than with it. Hinton et al. (Section 2.1) note that soft-target gradients scale as 1/T², so the T² factor keeps their contribution roughly unchanged when T is varied. Checked on the script's helpers: finite differences of the per-point objective with and without T² gave a soft-gradient ratio of 4.000 for every logit at T = 2. The analytic gradient matched finite differences to 3e-10 at the logit level, and to 1e-9 over all 27 student parameters through the batch mean.

**Script reference:** `02-alignment/microdistill.py`, lines 235-273 (`distillation_objective`), line 262 (T²-scaled objective), lines 264-270 (logit gradient)

</details>

---

### Challenge 3: A Perfect Copy Is Not the Optimum

**Setup:** The mixed objective is `ALPHA * T² * KL(p_T || q_T) + (1 - ALPHA) * CE(y, q_1)` (line 262). The KL term uses the teacher's softened distribution, which is not hard-coded: `teacher_soft_targets` (lines 219-232) computes it once from the trained teacher. Imagine a student whose logits exactly equal the teacher's on every training point.

**Question:** What is the soft-target KL for that student? Is the mixed objective zero? Is its gradient zero?

<details>
<summary>Reveal Answer</summary>

**Answer:** The KL is zero. The objective is not zero: it equals `(1 - ALPHA)` times the teacher's own hard-label cross-entropy. The gradient is also not zero, so a perfect copy of the teacher is not where training stops.

**Why:** Equal logits give `q_T = p_T`, so every KL term vanishes. Equal logits also make `q_1` equal to the teacher's T = 1 distribution, so the hard term is `0.1 x CE(y, teacher)`, and the hard-term gradient `0.1 (q_1 - onehot(y))` is nonzero wherever the teacher is not perfectly confident in the true class. The mixed objective pulls the student slightly toward the labels and away from the teacher. Checked on the script's helpers with a 40-epoch (reduced) teacher plugged in as its own student: KL about 1.6e-18, objective 0.019866 = 0.1 x the teacher's CE, and a largest gradient entry of 5.2e-3. The real student, with 27 parameters against the teacher's 99, may not be able to reach an exact copy at all.

**Script reference:** `02-alignment/microdistill.py`, lines 219-232 (`teacher_soft_targets`), lines 235-273 (`distillation_objective`), line 259 (KL), line 262 (mixed objective)

</details>

---

### Challenge 4: Forgetting the 1/B

**Setup:** `hard_label_ce` (lines 198-216) adds the per-point logit gradient `(q_1 - onehot(y)) / len(data)` (line 212), so the accumulated gradient is that of the batch MEAN. `sgd_step` (lines 187-192) applies plain gradient descent with `LEARNING_RATE = 0.1`, and the teacher trains full-batch on 240 points.

**Question:** If line 212 dropped `/ len(data)`, how would the gradient and the first teacher update change? Would training still work?

<details>
<summary>Reveal Answer</summary>

**Answer:** The gradient would become exactly 240 times larger, the gradient of the summed loss. Plain SGD would then take a step 240 times larger, an effective learning rate of 24. Training would not work: the loss jumps up instead of falling.

**Why:** Gradients are linear, so dropping the 1/B factor multiplies every per-point contribution by B. Unlike Adam, plain SGD does not normalize the gradient's scale, so the step grows with it. Checked on the script's helpers from the seed-42 initialization: one correct full-batch step lowered the training CE from 1.1422 to 1.0189, while one step with the 240x gradient raised it to 6.0336. After 10 such steps it was still 5.66, with 36% training accuracy, about chance for three classes. The same 1/B appears in the student's gradient (line 269). That is the batch size, not the number of classes N in Hinton et al.'s notation.

**Script reference:** `02-alignment/microdistill.py`, lines 198-216 (`hard_label_ce`, 1/B on line 212), lines 187-192 (`sgd_step`), line 269 (student 1/B)

</details>

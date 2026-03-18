# MicroGAN Challenges

Test your understanding of generative adversarial networks by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Perfect Discriminator

**Setup:** The discriminator outputs `sigmoid(logit)` (line 298), interpreted as `P(input is real)`. The generator loss (line 336) is `generator_loss_nonsaturating`: it minimizes `-mean(log(D(G(z))))`, wanting `D(G(z))` close to 1.

**Question:** If the discriminator becomes perfect -- outputting exactly 0 for all fake samples and 1 for all real samples -- what happens to the generator's gradient? Does the non-saturating loss formulation help here?

<details>
<summary>Reveal Answer</summary>

**Answer:** With the original minimax loss `log(1 - D(G(z)))`, a perfect discriminator gives `log(1 - 0) = 0`, producing zero gradient for the generator. The non-saturating formulation `-log(D(G(z)))` computes `-log(0) = +inf`, which provides an extremely strong gradient signal pushing the generator to improve.

**Why:** This is exactly the motivation described on lines 336-352. The minimax formulation saturates when the discriminator wins: `log(1 - D(G(z)))` has gradient `1/(D(G(z)) - 1)`, which approaches 0 as `D(G(z))` approaches 0. The non-saturating variant flips to `-log(D(G(z)))` with gradient `-1/D(G(z))`, which approaches `-inf` as `D(G(z))` approaches 0 -- a strong signal that forces the generator to improve. In practice, `safe_log` (line 182) clamps to `1e-10`, so the gradient is large but finite (`-1/1e-10 = -10^10`). The non-saturating loss preserves the same Nash equilibrium (`D = 0.5`) but fixes the gradient dynamics.

**Script reference:** `01-foundations/microgan.py`, lines 336-352 (`generator_loss_nonsaturating`), 182-190 (`safe_log` clamping), 143-155 (sigmoid activation)

</details>

---

### Challenge 2: Detecting Mode Collapse

**Setup:** The `detect_mode_collapse` function (line 394) checks whether the standard deviation of generated points drops below `threshold_std = 0.05` in either coordinate. The generator maps 8D noise to 2D points, outputting through `tanh` (line 252).

**Question:** If the generator collapses to always outputting `(0.3, -0.2)` regardless of input noise, what would `std_x` and `std_y` be? Would the detection function catch it? What if the generator collapsed to two modes instead of one?

<details>
<summary>Reveal Answer</summary>

**Answer:** With perfect single-point collapse, `std_x = 0.0` and `std_y = 0.0`, well below the threshold -- detected immediately. With two-mode collapse (e.g., alternating between `(0.5, 0.5)` and `(-0.5, -0.5)`), `std_x` and `std_y` would both be approximately `0.5`, which exceeds the threshold. The function would miss this partial collapse.

**Why:** Standard deviation measures spread across the entire sample. Two distant clusters produce high variance even though the generator only learned two points out of the continuous spiral distribution. The detection function on lines 394-424 is a crude heuristic that catches the most extreme failure mode (total collapse to a point) but misses subtler failures like partial mode coverage. Production GAN evaluation uses distributional metrics like FID (Frechet Inception Distance) that compare the full shape of generated vs. real distributions, not just marginal variance.

**Script reference:** `01-foundations/microgan.py`, lines 394-424 (`detect_mode_collapse`), lines 410-423 (variance computation and threshold check)

</details>

---

### Challenge 3: Unbalanced Training Schedule

**Setup:** The training loop alternates between discriminator and generator updates each step. The discriminator uses `discriminator_loss` (line 313) and the generator uses `generator_loss_nonsaturating` (line 336). Both use Adam with `BETA1 = 0.5` (line 48), lower than the typical 0.9.

**Question:** If you trained the generator 10x more often than the discriminator (10 generator steps per 1 discriminator step), what would happen to training dynamics? Why is `BETA1 = 0.5` used instead of the standard 0.9?

<details>
<summary>Reveal Answer</summary>

**Answer:** The generator would overfit to the current discriminator's weaknesses, producing samples that exploit specific blind spots rather than matching the real distribution. The discriminator, updating 10x less often, can't keep up. This leads to oscillation: the generator finds an exploit, the discriminator eventually corrects it, the generator finds a new one. `BETA1 = 0.5` reduces momentum to prevent the optimizer from overshooting past the adversary's moving target.

**Why:** GAN training is a two-player game where each player's loss landscape shifts as the other updates. High momentum (`BETA1 = 0.9`) means the optimizer carries velocity from ~10 past steps, but those gradients were computed against a different adversary. Lower momentum (`BETA1 = 0.5`, effective window ~2 steps) makes the optimizer more responsive to the current game state. This is noted on lines 48-50. The standard 1:1 training ratio helps maintain approximate equilibrium between the two networks. Disrupting this ratio in either direction tends to destabilize training -- training the discriminator too much makes it unbeatable (see Challenge 1), training the generator too much makes it exploitative rather than generalizing.

**Script reference:** `01-foundations/microgan.py`, lines 48-50 (BETA1 rationale), 313-333 (`discriminator_loss`), 336-352 (`generator_loss_nonsaturating`)

</details>

---

### Challenge 4: Generator Output Range

**Setup:** The generator's final activation is `tanh` (line 252), bounding outputs to `[-1, 1]`. The spiral data is scaled by `1/2` via `scale_point` (line 443), putting most points in approximately `[-1, 1]`.

**Question:** If you replaced `tanh` with `relu` as the generator's output activation, what would go wrong? What if you removed the output activation entirely (raw linear output)?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- **ReLU:** The generator can only produce points with non-negative coordinates (first quadrant). Half the spiral lives in negative-coordinate regions, so the generator can never match the full distribution. The discriminator would trivially reject any point with a negative coordinate.
- **No activation:** Early in training, random weights produce arbitrary-magnitude outputs. Points far from `[-1, 1]` are trivially distinguishable from real data (which is bounded), so the discriminator wins easily and the generator gets no useful gradient signal. Even if training stabilizes, unbounded outputs make the loss landscape harder to navigate.

**Why:** The `tanh` on line 252 serves two purposes: it matches the output range to the data range, and it bounds the generator's output to prevent the discriminator's job from being trivially easy. Without range matching, the discriminator doesn't need to learn anything about the data distribution's shape -- it can classify based on magnitude alone. This is noted in the Generator class docstring on lines 224-226: "Without bounding, the generator can produce arbitrarily large values early in training, making the discriminator's job trivially easy."

**Script reference:** `01-foundations/microgan.py`, lines 218-252 (Generator class, especially lines 224-226 and 252), 443-448 (`scale_point`)

</details>

---

### Challenge 5: Sigmoid Saturation in the Discriminator

**Setup:** The discriminator outputs `sigmoid(logit)` (line 298). The sigmoid function (line 143) clamps input to `[-500, 500]` and computes `1 / (1 + exp(-x))`. Its derivative is `sigmoid(x) * (1 - sigmoid(x))`.

**Question:** If the discriminator's pre-sigmoid logit is +50 for a real sample, what is `sigmoid(50)`? What is the gradient `sigmoid(50) * (1 - sigmoid(50))`? What does this mean for learning?

<details>
<summary>Reveal Answer</summary>

**Answer:** `sigmoid(50)` is approximately `1.0` (specifically, `1 - 1.93e-22`). The gradient is approximately `1.93e-22` -- effectively zero. The discriminator has stopped learning from this sample because it is already maximally confident.

**Why:** Sigmoid saturates exponentially: for inputs above ~10, the output is indistinguishable from 1.0 in float64. The derivative `s * (1 - s)` peaks at 0.25 when `s = 0.5` (maximum uncertainty) and vanishes at both extremes. A logit of +50 means the discriminator is absurdly confident this is real, and no gradient flows back to adjust its weights for this example. This is fine when the discriminator is correct, but if it develops overconfident incorrect predictions (classifying a good fake as certainly real), the vanishing gradient prevents correction. This saturating behavior is one reason GAN training is unstable -- both the generator loss and discriminator confidence can create gradient dead zones.

**Script reference:** `01-foundations/microgan.py`, lines 143-155 (`sigmoid` method, derivative on line 155), 265-298 (Discriminator class using sigmoid output)

</details>

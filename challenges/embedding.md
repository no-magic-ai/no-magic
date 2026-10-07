# MicroEmbedding Challenges

Test your understanding of contrastive embedding learning by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: The Temperature Paradox

**Setup:** `TEMPERATURE = 0.1` (line 37). The InfoNCE loss divides cosine similarities by `temperature` before the softmax (lines 244, 251). The comment on line 232 states that low temperature "makes the loss focus on hard negatives."

**Question:** If you set `TEMPERATURE = 0.001` (100x smaller), would training become faster (sharper gradients toward correct pairs) or slower (collapse)? Why doesn't `TEMPERATURE = 0` work?

<details>
<summary>Reveal Answer</summary>

**Answer:** Training would become unstable rather than faster. `TEMPERATURE = 0` is undefined (division by zero).

**Why:** Dividing similarities by a very small temperature amplifies small differences between similarity scores to extreme values. Before training, embeddings are random, so the positive pair is unlikely to have the highest similarity in the batch. With tau=0.001, the softmax becomes a near-step-function: the single highest similarity gets all the probability mass, and `exp_pos / denom` approaches 0 if the positive isn't the maximum. The loss becomes `-log(~0) = large`, with near-infinite gradients that destabilize the weight matrix. The max-subtraction at lines 255-257 prevents numerical overflow, but the gradients carry a `1 / temperature` factor (lines 270-280), so at 0.001 they are 100 times larger than at 0.1 for the same learning rate. The script's comment calls 0.1 standard in SimCLR; it is a value that keeps that factor moderate.

**Script reference:** `01-foundations/microembedding.py`, lines 37 (TEMPERATURE), lines 244-261 (InfoNCE loss computation), lines 254-258 (log-sum-exp stability trick)

</details>

---

### Challenge 2: Same-Name Negatives

**Setup:** The InfoNCE loss treats other samples in the batch as negatives (line 248-251): `for j in range(bs): if j != i: sim_negs.append(cosine_similarity(anchor_embs[i], anchor_embs[j]) / temperature)`. `BATCH_SIZE = 64` (line 39).

**Question:** Suppose two different samples in the same batch happen to be the same name, e.g., "anna" appears at both index 3 and index 47. Index 3's anchor treats index 47's anchor as a negative. What happens to the gradient, and why is this a known issue in contrastive learning?

<details>
<summary>Reveal Answer</summary>

**Answer:** The gradient pushes the embeddings of the two "anna" samples apart — even though they should be close. This is the "false negative" problem in contrastive learning, and it degrades representation quality, especially when the dataset has many duplicates.

**Why:** The loss on line 261 (`-log(exp_pos / denom)`) is minimized by making `exp_pos` large and the denominator small. The denominator includes `exp(sim(anna_3, anna_47) / tau)`. To minimize the loss for sample 3, the gradient pushes `sim(anna_3, anna_47)` down — treating anna_47 as a distractor to avoid. Larger systems mitigate it by deduplicating batches or by detecting and down-weighting likely false negatives. This script uses random shuffling (line 306) and does neither; `names.txt` does contain repeated names (32,033 lines, 29,494 distinct), so collisions within a 64-name batch can occur.

**Script reference:** `01-foundations/microembedding.py`, lines 39 (BATCH_SIZE), lines 247-281 (negative computation in InfoNCE), lines 303-313 (batch construction via shuffle)

</details>

---

### Challenge 3: Augmentation Failure on Short Names

**Setup:** The `augment` function (line 112) returns the original name unchanged when `len(name) <= 2` (line 120-121). The positive pair is `(anchor, augmented_anchor)`.

**Question:** For two-character names like "Jo" or "Al", what does the InfoNCE loss compute? Will the model learn useful representations for short names, or does the skip cause a degenerate training signal?

<details>
<summary>Reveal Answer</summary>

**Answer:** For short names, `augment` returns the identical string. The positive pair is `("jo", "jo")`, so `cosine_similarity(anchor_emb, positive_emb) = 1.0` (identical vectors after identical encoding). The loss pushes this pair's similarity toward 1, which is correct — but the model receives no invariance training for short names.

**Why:** When anchor and positive are the same string, their n-gram encodings are identical, their raw embeddings are identical, and their normalized embeddings are identical. The cosine similarity is 1.0, so `exp(1.0 / tau)` dominates the numerator. The loss for that sample is low whenever the negatives are dissimilar, because the denominator is dominated by the positive term. The model effectively gets a "free" near-zero loss for short names without learning to be robust to perturbations. The guard is a deliberate simplification ("too short to augment safely", line 121): deleting a character from a two-letter name would leave a single letter.

**Script reference:** `01-foundations/microembedding.py`, lines 112-132 (`augment` function), lines 119-121 (short name guard), lines 326-337 (anchor and positive encoding in training loop)

</details>

---

### Challenge 4: Radial Gradient Collapse

**Setup:** The `grad_through_norm` function (line 180) projects out the radial component of the gradient: `d(L)/d(z_i) = (g_i - e_i * dot(g, e)) / ||z||` (line 186). The comment on lines 190-193 explains this prevents "representation collapse."

**Question:** What happens if you replace `grad_through_norm` with a simple pass-through (`return list(grad_normalized)`)? Would embeddings still train? Would they collapse?

<details>
<summary>Reveal Answer</summary>

**Answer:** Training still runs, but the updates no longer follow the loss. The loss depends only on the normalized embedding `e = z / ||z||`, so any change to `z` along its own direction `e` leaves the loss unchanged; the pass-through keeps exactly that radial part and also drops the `1 / ||z||` scale. Whether this ends in collapse is not settled by the script: its comment (lines 190-193) warns of collapse, but no run in the repository compares the two versions.

**Why:** Check the formula on a gradient that points straight along `e`: with `g = c * e`, `g - e * dot(g, e) = c * e - c * e = 0`, so the correct gradient with respect to `z` is zero — stretching `z` does not change `e`. The pass-through would instead push `z` along `e` with size `c`. Each embedding's radial part points along its own direction, and all of them flow into the shared projection matrix `W` (lines 352-363), so `W` receives updates that do not reduce the loss and the useful tangential part is mis-scaled by a factor of `||z||`. The projection on line 199 removes the radial part and applies the `1 / ||z||` factor, which is what the chain rule through `e = z / ||z||` requires.

**Script reference:** `01-foundations/microembedding.py`, lines 180-199 (`grad_through_norm`), lines 190-193 (why comment), lines 352-363 (gradient application in training loop)

</details>


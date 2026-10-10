# MicroWatermark Challenges

Test your understanding of green-list watermarking by predicting what happens in these scenarios. Try to work out the answer before revealing it. Numbers quoted in the answers come from the default seed-42 run of `03-systems/microwatermark.py` under CPython 3.12.8, or from probes that call the script's own functions on its own trained model. They describe this toy grammar and model only; none of them is a claim about real text.

---

### Challenge 1: The Right Text, the Wrong Key

**Setup:** `main` scores the 200 watermarked texts of the primary arm twice. With the key that generated each text they give the right-key rows. They are also placed in the control group `"watermarked text, WRONG key"` (line 1498) and scored with the 200 null keys of `null_tables` (line 1463), keys that never generated anything. Every key's green lists come from `green_masks` (lines 509-525): a keyed `blake2b` of the previous token seeds `random.Random`, which picks 12 of the 48 tokens (lines 519-520).

**Question:** Over these 40,000 (text, key) pairs at 200 generated tokens, what mean z and what rate of z > 4 do you expect from the deduplicated detector? Does the plain detector look the same?

<details>
<summary>Reveal Answer</summary>

**Answer:** The deduplicated detector sees nothing: mean z 0.044, standard deviation 0.926, and 0 of 40,000 pairs above 4 (gate N2). The plain detector also centers near zero (mean 0.024), but its standard deviation is 1.704 and 392 of the 40,000 pairs, a rate of 0.00980, cross z = 4.

**Why:** The generator pushed each token toward the green list of its own key. Under another key, the list after each previous token is an unrelated pseudorandom subset, so whether a given (previous, token) pair is green is close to a fresh coin with probability 12/48 = 0.25: the bias toward one key's lists carries no information about another key's lists. That is what makes the scheme keyed: the same text is strongly watermarked for one key and ordinary for all others. The plain detector's false positives do not come from the watermark at all. The watermarked texts repeat (previous, token) pairs as often as unwatermarked text does, and every repeat re-counts the same coin, so the plain statistic is too wide (Challenge 4). The unwatermarked model texts behave the same way under the null keys: 1 of 40,000 deduplicated z-scores and 435 of 40,000 plain ones exceed 4.

**Script reference:** `03-systems/microwatermark.py`, lines 509-525 (`green_masks`), line 1463 (`null_tables`), lines 1495-1501 (control groups), lines 1554-1564 (N1a/N1b/N2 gates and the report-only PLAIN line)

</details>

---

### Challenge 2: A Key Without a Bias

**Setup:** In `sample_token` (lines 597-628) the default soft rule adds `delta` to every green logit (line 616), then draws exactly one `rng.random()` per token (line 622). Every arm reuses the same random stream for each (prompt, seed) pair (line 1108). Suppose a twelfth arm were added: soft rule, γ = 0.25, the same five generation keys, but δ = 0.

**Question:** How would its 200 texts compare with the unwatermarked arm's texts, and what fraction would the right key flag at 200 tokens?

<details>
<summary>Reveal Answer</summary>

**Answer:** The texts are identical, token for token, to the unwatermarked arm (`none t=1.0`), and the right key flags none of them. A probe that runs the script's own `generate` with δ = 0 found all 200 texts identical. Scored with their own generation keys at 200 tokens, 0 of 200 exceed z = 4 under either detector: deduplicated mean 0.043, standard deviation 0.855, maximum 2.84; plain mean −0.242, standard deviation 1.443, maximum 3.15.

**Why:** `v + 0.0` equals `v` exactly for every finite logit, so the sampling weights, the single uniform draw and the chosen token are all unchanged. The key only decides *where* the bias goes; δ is what moves probability. With δ = 0 the green lists exist but nothing in the text depends on them, so the right key is just one more null key. Gate W1 would fail (no detection), and so would gate W5, which requires the watermarked text's NLL under the unmodified model to be strictly higher than the unwatermarked text's. The z-statistic itself never uses δ: the detector only needs the key and γ (Sec. 3.1 of the paper), which is why the same Eq. 3 can score every arm.

**Script reference:** `03-systems/microwatermark.py`, lines 597-628 (`sample_token`), line 616 (soft bias), line 622 (one uniform draw), line 1108 (shared stream), lines 550-552 (`z_statistic`, no δ)

</details>

---

### Challenge 3: The Step the Watermark Cannot Touch

**Setup:** About 30% of grammar sentences open with "once upon a time ," (line 274), so after the context (".", "once") the trained model is almost certain the next word is "upon". The generator's exact chance of emitting a green token is `boosted_green_probability` (lines 823-825): P(green) = α g / (α g + 1 − g), with α = e^δ = e² ≈ 7.39 and g the model's own probability mass on the green list. `entropy_bucket_gates` (lines 1315-1382) tallies this prediction against what was sampled.

**Question:** For the five demo keys, what is P(green) at the step after ". once"? Compare it with the γ = 0.25 a uniform-random token would have.

<details>
<summary>Reveal Answer</summary>

**Answer:** Between 0.0013 and 0.0030, far below 0.25. The model gives "upon" probability 0.9988 there (entropy 0.012 nats). Under all five keys "upon" happens to be red after "once" (each key makes it red with probability 36/48 = 0.75, so all five red has probability 0.75⁵ ≈ 0.24). The green mass g is therefore only 0.00018 to 0.00041, and the bias raises it to P(green) = 0.0013 to 0.0030. By contrast, after ". the" (entropy 3.01 nats, top word probability 0.13) the five keys give g = 0.089 to 0.218 and P(green) = 0.418 to 0.674.

**Why:** δ multiplies the odds of green by about 7.39, and 7.39 times a tiny green mass is still tiny. A near-deterministic step emits its top token whatever the key, so it adds a scored pair whose color the watermark did not choose. Over the whole primary arm the low-entropy bucket (< 0.3 nats) ends up 0.340 green and the high-entropy bucket (> 1.2 nats) 0.655 green (gate W6b), and the observed counts match the exact prediction in every bucket (gate W6a). The low bucket sits above 0.25 because some near-deterministic steps have a green top token, which is then emitted almost surely. This is the paper's low-entropy limitation (Sec. 1.2 and 4): text made of fixed phrases, memorized passages or code carries little watermark, and no choice of δ fixes that without breaking the text. Theorem 4.2 states the same limit as a lower bound that shrinks with the spike entropy; the script's check of that bound is loose except in near-deterministic contexts.

**Script reference:** `03-systems/microwatermark.py`, line 274 (the opener), lines 823-825 (`boosted_green_probability`), lines 1315-1382 (`entropy_bucket_gates`; prediction, buckets and tallies on lines 1347-1353)

</details>

---

### Challenge 4: Counting the Same Coin Twice

**Setup:** `detect` (lines 555-574) scores every (previous, token) pair of the text unless `skip_repeats` is set, in which case a pair that has already been seen is skipped (lines 566-569). Gate E8 (lines 974-979) scores the alternating text 7 8 7 8 7 8 ("green old green old green old") with key 0 and γ = 0.25. Under that key, ("green", "old") is red and ("old", "green") is green.

**Question:** (a) What do the plain and deduplicated detectors return for the alternating text? (b) On unwatermarked text sampled at temperature 0.3, the plain detector's null z has standard deviation 2.91 at 200 tokens (gate M1a). A correctly calibrated z-score has standard deviation 1. Where does the extra spread come from?

<details>
<summary>Reveal Answer</summary>

**Answer:** (a) Plain: 5 scored pairs, 2 green (the two "old green" pairs), z = (2 − 1.25) / √(5 × 0.25 × 0.75) = 0.775. Deduplicated: 2 scored pairs, 1 green, z = (1 − 0.5) / √(2 × 0.25 × 0.75) = 0.816. (b) Repetition. Sharpened sampling keeps choosing the model's favorite continuations, so a 200-token text has only 57.4 distinct pairs on average among its 199 scored pairs (between 41 and 72), against 114.6 at temperature 1. A pair that occurs r times adds r copies of one green/red coin, which contributes variance r² γ(1 − γ) instead of the r γ(1 − γ) that Eq. 3 assumes. The plain statistic inherits that inflation: 3,224 of 40,000 null (text, key) pairs, a rate of 0.0806, exceed z = 4, where a Gaussian tail would give about 1. The deduplicated statistic scores each distinct pair once and keeps a standard deviation of 0.952, with no z above 4.

**Why:** Eq. 3 is a one-proportion z-test, and its denominator assumes T independent coins. For a fixed key the green/red color of a pair is a fixed property of that pair, so repeats are not new evidence; Sec. 4.1 of the paper suggests not counting repeated n-grams. The cost is that the deduplicated test has fewer scored pairs, so it needs a stronger per-pair excess to reach the same z: at 200 tokens it flags 0.970 of the primary watermarked texts, against 1.000 for the plain test. At matched false-positive rates the script prints both detectors' numbers without declaring a winner, because the gaps are a few texts out of 200.

**Script reference:** `03-systems/microwatermark.py`, lines 555-574 (`detect`), lines 566-569 (`skip_repeats`), lines 974-979 (gate E8), lines 1568-1575 (gate M1a), lines 660-673 (`prepare_text`, first occurrences per prefix)

</details>

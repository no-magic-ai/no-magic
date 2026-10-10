# MicroSFT Challenges

Test your understanding of supervised fine-tuning by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Which Positions Carry the SFT Loss?

**Setup:** `sft_example` (lines 137-150) frames `copy:a> → a` as `[BOUNDARY, c, o, p, y, :, a, >, a, BOUNDARY]` and returns `first_target = 1 + len(prompt) = 8`. `mean_target_nll` (lines 373-401) runs the decoder on positions 0-8 and appends a loss term only `if pos + 1 >= first_target` (line 399), then divides by the number of terms (line 401). Pretraining calls the same function with `first_target = 1`.

**Question:** How many terms does the SFT loss of one pair average, which decoder positions produce them, and what would change if SFT also passed `first_target = 1`?

<details>
<summary>Reveal Answer</summary>

**Answer:** Two terms: decoder position 7 (input `>`) predicting the response `a` at index 8, and position 8 (input `a`) predicting the end boundary at index 9. The loss is their sum divided by 2. With `first_target = 1` it would average all 9 next-token terms and train the model to predict the prompt characters too, which is ordinary language modelling on the demonstration text rather than a response-only objective.

**Why:** The causal shift pairs decoder position `t-1` with target token `t`, so targets 8 and 9 come from positions 7 and 8. The condition on line 399 is the mask: prompt targets 1-7 are never appended. Calling the script's own `mean_target_nll` on this pair gave a value equal to the mean of the separately computed position-7 and position-8 negative log-likelihoods to within 1e-12, while the single-term, two-term-sum and three-term (positions 6-8) alternatives all differed. The paper behind this program does not prescribe a response-only mask; the script makes it a disclosed teaching choice.

**Script reference:** `02-alignment/microsft.py`, lines 137-150 (framing and `first_target`), lines 373-401 (`mean_target_nll`, mask on line 399, denominator on line 401)

</details>

---

### Challenge 2: Masked Targets, Live Gradients

**Setup:** The eight characters `n o p t x y : >` (`PROMPT_ONLY_CHARS`, line 51) appear only inside SFT prompts, and none of them is ever a supervised SFT target. `decoder_step` (lines 327-367) appends every position's key and value to the attention cache before the response positions read it.

**Question:** During SFT, do the embedding rows of these prompt-only characters receive a nonzero gradient? Does the position-9 embedding row `wpe[9]`?

<details>
<summary>Reveal Answer</summary>

**Answer:** The prompt-only rows do; `wpe[9]` does not.

**Why:** Masking removes prompt positions from the loss sum, not from the forward pass. Position 7's prediction attends over the keys and values of positions 0-7, so the response loss depends on the prompt embeddings through attention, and backpropagation reaches them. Position 9 is never an input: the last input is index 8, because index 9 is only a target. Its embedding row never enters the computation graph, so its gradient is exactly zero. On the script's own helpers, at initialization, the full-batch SFT gradient summed to |grad| ≈ 0.67 over the prompt-only rows and exactly 0.0 on `wpe[9]`. The main program prints the same kind of measurement for one pair after pretraining (lines 591-605).

**Script reference:** `02-alignment/microsft.py`, lines 327-367 (`decoder_step`, attention cache), lines 373-401 (masked loss), lines 591-605 (printed prompt-gradient check)

</details>

---

### Challenge 3: What Pretraining Never Touched

**Setup:** The base corpus (lines 94-105) contains only `a`-`h` and the boundary token. `adam_update` (lines 416-426) applies `param.data -= LEARNING_RATE * m_hat / (v_hat**0.5 + EPS_ADAM)` to every parameter, including those whose gradient is zero. After pretraining, the main program compares the prompt-only embedding rows with their initial values (lines 577-587).

**Question:** After 200 pretraining updates, are the prompt-only embedding rows exactly equal to their random initialization, approximately equal, or changed? What does that mean for the claim that SFT transfers pretrained knowledge?

<details>
<summary>Reveal Answer</summary>

**Answer:** Exactly equal. These rows are not pretrained knowledge. Their first learning signal arrives during SFT. What transfers from pretraining is the shared decoder weights, the symbol embeddings and the position embeddings.

**Why:** A token that never appears as an input has zero gradient on its embedding row at every pretraining step. With zero gradients, Adam's moments stay exactly 0, so `m_hat = 0` and the update is `0 / (0 + 1e-8) = 0`. The subtraction leaves the value bit-for-bit unchanged. A reduced run of the script's own `train_stage` (10 pretraining updates, not the default 200) left every prompt-only row identical to its initial value while other parameters changed. The same argument holds for any number of updates. Unlike these rows, the output-head rows for these characters do change during pretraining: the softmax normalizer gives every vocabulary logit a gradient.

**Script reference:** `02-alignment/microsft.py`, lines 94-105 (base corpus), lines 416-426 (`adam_update`), lines 577-587 (untouched-row check)

</details>

---

### Challenge 4: Dropping the 1/14

**Setup:** Each SFT update calls `accumulate_batch_gradient` (lines 429-447). It runs `backward()` on `mean_target_nll(...) * scale` for every training pair with `scale = 1.0 / len(examples)` (line 441), so `param.grad` accumulates the gradient of the batch mean. `train_stage` (lines 450-479) then takes one Adam step and zeroes the gradients.

**Question:** Suppose line 441 were changed to `scale = 1.0`. By what factor would the accumulated gradient change? How much would the first Adam update change? Which printed number would visibly change?

<details>
<summary>Reveal Answer</summary>

**Answer:** The accumulated gradient becomes exactly 14 times larger: the sum of the per-pair gradients instead of their mean. The first Adam update stays almost the same, except for coordinates whose gradient is comparable to `EPS_ADAM = 1e-8`. The printed SFT loss becomes 14 times larger, because it is accumulated from the same scaled terms.

**Why:** Gradients are linear, so the per-pair contributions add, and the 1/14 factor scales them all. Adam divides the first moment by the square root of the second moment. Multiplying every gradient by 14 multiplies `m_hat` by 14 and `sqrt(v_hat)` by 14, so the ratio is unchanged until `EPS_ADAM` matters. On the script's helpers at initialization, the unscaled gradient was 14.000 times the scaled one in all 1,096 nonzero coordinates. One Adam step differed by at most 6.9e-4 per parameter, against a largest step of 0.01, and the large differences sat where the gradient was tiny. The batch mean is still the right quantity to report and to reason about: with plain SGD, which does not normalize like this, dropping the 1/14 would make the step 14 times larger.

**Script reference:** `02-alignment/microsft.py`, lines 429-447 (`accumulate_batch_gradient`, scale on line 441), lines 450-479 (`train_stage`), lines 416-426 (`adam_update`)

</details>

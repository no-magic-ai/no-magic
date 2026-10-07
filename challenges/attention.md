# Attention Challenges

Test your understanding of attention mechanisms by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Removing the Scaling Factor

**Setup:** The `vanilla_attention` function (line 108) computes `scale = 1.0 / math.sqrt(len(q[0]))`. With `D_MODEL = 64` and `N_HEADS = 4`, each head operates on `HEAD_DIM = 16` dimensions. Suppose you remove the scaling factor entirely, setting `scale = 1.0`.

**Question:** What happens to the softmax distribution over keys? How does this change the attention output?

<details>
<summary>Reveal Answer</summary>

**Answer:** The softmax distribution becomes much sharper — closer to one-hot — so each output row moves from a broad mixture of value vectors toward copying the few values with the highest scores.

**Why:** Dot products between random vectors in 16 dimensions have variance proportional to `d_k`. Without dividing by `sqrt(16) = 4`, the raw scores have ~4x larger magnitude. When fed into softmax, larger inputs push the distribution toward saturation: `exp(large) / sum(exp(...))` concentrates more mass on the maximum score. Pushed far enough (larger `d_k` or larger activations), attention stops forming weighted mixtures and behaves like a hard argmax lookup, and the softmax gradients shrink. This is exactly why Vaswani et al. introduced the `1/sqrt(d_k)` scaling -- it keeps the variance of the dot products at ~1.0 regardless of dimension.

**Script reference:** `03-systems/microattention.py`, lines 108-120 (`vanilla_attention` function, especially the `scale` computation on line 116 and the comment on lines 113-115)

</details>

---

### Challenge 2: All Keys Identical

**Setup:** In `vanilla_attention` (line 108), suppose every row of the key matrix `k` is the same vector -- e.g., all 32 positions share the identical key `[0.1, 0.2, ..., 0.1]`.

**Question:** What does the attention output reduce to? What is every row of the attention weight matrix?

<details>
<summary>Reveal Answer</summary>

**Answer:** Every row of the attention weight matrix becomes a uniform distribution `[1/n, 1/n, ..., 1/n]`, and the attention output at each position is the mean of all value vectors.

**Why:** When all keys are identical, `Q @ K^T` produces a matrix where each row has the same value repeated across all columns (each query dotted with the same key gives the same score for every position). Softmax of a constant vector is the uniform distribution. The output `softmax(scores) @ V` then computes `(1/n) * sum(V)` for each row -- a simple average of all value vectors. Attention with identical keys is equivalent to mean-pooling the values, completely ignoring positional information.

**Script reference:** `03-systems/microattention.py`, lines 117-120 (score computation, softmax, and value weighting)

</details>

---

### Challenge 3: Causal Window Bounds

**Setup:** The `sliding_window_attention` function (line 207) implements causal masking implicitly: position `i` only computes scores for positions `start` through `i` (lines 222-226). Consider a sequence of length 32 with `WINDOW_SIZE = 8`.

**Question:** For position 20, which positions can it attend to? For position 3, which positions can it attend to? What happens at position 0?

<details>
<summary>Reveal Answer</summary>

**Answer:**
- Position 20 attends to positions 13-20 (8 positions total: `max(0, 20-8+1) = 13`).
- Position 3 attends to positions 0-3 (4 positions -- fewer than the window size, because there aren't enough predecessors).
- Position 0 attends only to itself (1 position).

**Why:** The window bounds are computed as `start = max(0, i - window_size + 1)` on line 222. For early positions where `i < window_size - 1`, the window is clipped at 0, so these positions see fewer than `window_size` keys. Position 0 always sees only itself. That part is true of any causal attention, windowed or not; what the window adds is that late positions can never look further back than 8 positions in one layer. Stacking windowed layers widens the reach (Mistral 7B relies on this), and Longformer adds global attention on selected tokens.

**Script reference:** `03-systems/microattention.py`, lines 207-233 (`sliding_window_attention`, especially `start = max(0, i - window_size + 1)` on line 222)

</details>

---

### Challenge 4: GQA with Mismatched Head Counts

**Setup:** In `grouped_query_attention` (line 149), with `N_HEADS = 4` query heads and `N_KV_HEADS_GQA = 2` KV heads, the group size is `gs = 4 // 2 = 2`. Query heads 0-1 share one KV pair, heads 2-3 share another.

**Question:** If you set `N_KV_HEADS_GQA = 4` (equal to query heads), what does GQA reduce to? If you set `N_KV_HEADS_GQA = 1`, what does it reduce to? What is the KV cache memory ratio in each case compared to standard MHA?

<details>
<summary>Reveal Answer</summary>

**Answer:**
- `N_KV_HEADS_GQA = 4` (matching query heads): GQA becomes standard MHA. Group size = 1, so each query head has its own KV projection. KV cache ratio = 1x (no saving).
- `N_KV_HEADS_GQA = 1`: GQA becomes MQA (multi-query attention). All 4 query heads share a single KV pair. KV cache ratio = 1/4 of MHA.

**Why:** GQA is a generalization that spans the spectrum from MHA (all heads independent) to MQA (all heads share one KV). The group index `g = h // gs` on line 171 determines which KV head each query head uses. When `gs = 1`, each query head maps to a unique KV head (MHA). When `gs = n_heads`, all query heads map to KV head 0 (MQA). The KV cache stores `2 * n_kv_heads * seq_len * head_dim` floats (keys and values), so its size relative to MHA is `n_kv_heads / n_heads`.

**Script reference:** `03-systems/microattention.py`, lines 149-178 (`grouped_query_attention`, especially `gs = n_heads // n_kv_heads` on line 163 and `g = h // gs` on line 171)

</details>

---

### Challenge 5: MQA Output Quality

**Setup:** The script computes cosine similarity between each variant's output and MHA's output (line 339). MQA shares a single KV head across all 4 query heads, while GQA (2 KV heads) shares between pairs.

**Question:** Which variant has higher cosine similarity to MHA: GQA or MQA? The KV weights for both are derived from MHA weights via `avg_head_weights` (line 88). Does this initialization strategy help or hurt the comparison?

<details>
<summary>Reveal Answer</summary>

**Answer:** GQA. With the default seed the script's functions give 0.684 for GQA and 0.578 for MQA. These are untrained forward outputs compared with an untrained MHA output; the numbers say how much of MHA's key/value structure survives averaging, not how a trained GQA or MQA model would perform.

**Why:** GQA with 2 KV heads averages pairs of MHA heads (groups of 2), preserving more of the original per-head specialization. MQA averages all 4 heads into one, collapsing all KV diversity into a single representation. The averaging initialization (from Ainslie et al. 2023, line 92) is specifically designed to make converted GQA/MQA models approximate the original MHA output. With random initialization instead, both variants would start far from MHA's output. The averaging strategy gives GQA a structural advantage: averaging 2 heads loses less information than averaging 4.

**Script reference:** `03-systems/microattention.py`, lines 88-103 (`avg_head_weights`), lines 294-300 (weight derivation), line 339 (cosine similarity computation)

</details>

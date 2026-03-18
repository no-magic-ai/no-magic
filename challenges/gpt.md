# MicroGPT Challenges

Test your understanding of autoregressive language modeling by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Context Window Overflow

**Setup:** The model has `BLOCK_SIZE = 16` (line 38). During training, sequences are truncated: `seq_len = min(BLOCK_SIZE, len(tokens) - 1)` (line 458). During inference, the generation loop runs `for pos in range(BLOCK_SIZE)` (line 539).

**Question:** If a name in the training data produces a token sequence of length 20 (after adding BOS tokens), how many tokens does the model actually train on? During inference, what limits the maximum generated name length?

<details>
<summary>Reveal Answer</summary>

**Answer:** The model trains on the first 16 tokens (positions 0-15), predicting tokens at positions 1-16. The last 3 tokens of the 20-token sequence are never seen. During inference, generation stops after at most 16 characters (BLOCK_SIZE positions), or earlier if the model produces a BOS token (end-of-name signal).

**Why:** The truncation on line 458 caps `seq_len` at `BLOCK_SIZE = 16`. The training loop (lines 466-480) iterates `for pos in range(seq_len)`, processing input tokens 0 through 15 and predicting targets 1 through 16. The position embedding table `wpe` only has 16 rows (line 212), so there is no embedding for position 16 or beyond -- the model physically cannot represent longer contexts. Production GPTs handle this with techniques like RoPE (rotary position embeddings) that generalize beyond the training context length, but this implementation uses learned absolute position embeddings that are fixed at initialization.

**Script reference:** `01-foundations/microgpt.py`, lines 38 (BLOCK_SIZE), 212 (wpe initialization), 457-458 (truncation), 539 (inference loop)

</details>

---

### Challenge 2: Head Dimension Arithmetic

**Setup:** The model uses `N_EMBD = 16`, `N_HEAD = 4`, and `HEAD_DIM = N_EMBD // N_HEAD = 4` (lines 35-39). The Q, K, V projection matrices are `[N_EMBD, N_EMBD]` (lines 218-220). In the attention computation, each head slices out its portion: `q_head = q[head_start : head_start + HEAD_DIM]` (line 363).

**Question:** What is the total dimension of the concatenated head outputs before the output projection? If you changed `N_HEAD = 8` without changing `N_EMBD = 16`, what would `HEAD_DIM` become and what would the model's representational capacity trade look like?

<details>
<summary>Reveal Answer</summary>

**Answer:** The concatenated output is 16 dimensions (4 heads x 4 dims each = N_EMBD). With `N_HEAD = 8`, `HEAD_DIM` drops to 2. Each head can only form 2-dimensional attention patterns, severely limiting per-head expressiveness, but you get 8 independent "views" of the data instead of 4.

**Why:** Multi-head attention partitions the embedding dimension across heads: `HEAD_DIM = N_EMBD // N_HEAD`. The concatenation on line 387 (`x_attn.extend(head_output)`) reassembles these slices back to the full `N_EMBD` dimensions. With `HEAD_DIM = 2`, each head's Q and K are 2D vectors, meaning attention scores are based on just 2 features. The dot product in a 2D space can only distinguish directions, not complex feature combinations. This is the fundamental tradeoff of multi-head attention: more heads = more diverse attention patterns, but each pattern is computed in a lower-dimensional subspace.

**Script reference:** `01-foundations/microgpt.py`, lines 35-39 (constants), 359-387 (per-head attention and concatenation)

</details>

---

### Challenge 3: Learning Rate Extremes

**Setup:** The training loop uses Adam with linear learning rate decay: `lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)` (line 493). The default `LEARNING_RATE = 0.01` (line 42).

**Question:** If you set `LEARNING_RATE = 0.0`, what happens to the model? If you set `LEARNING_RATE = 10.0`, what happens? In neither case does the program crash -- why?

<details>
<summary>Reveal Answer</summary>

**Answer:**

- `LEARNING_RATE = 0.0`: All parameter updates are zero. The model generates purely from its random initialization, producing gibberish character sequences. The loss remains at approximately `-log(1/VOCAB_SIZE)` (uniform prediction).
- `LEARNING_RATE = 10.0`: Parameters overshoot wildly on every step. Logits explode to extreme values, softmax saturates, and the loss oscillates or grows. The model generates repetitive or degenerate sequences.

**Why:** The program doesn't crash because `safe_log` (line 280) clamps probabilities to `1e-10` before taking the log, preventing `-inf`. The softmax (line 250) subtracts the max logit before `exp()`, preventing overflow. These numerical stability guards keep the computation finite even when parameters diverge. However, with `lr=10.0`, the Adam bias-corrected updates `lr * m_hat / (sqrt(v_hat) + eps)` on line 511 become enormous, pushing weights far from any useful configuration. The model still produces valid probability distributions (softmax always sums to 1), but the distributions are meaningless.

**Script reference:** `01-foundations/microgpt.py`, lines 42 (LEARNING_RATE), 280-295 (safe_log), 250-262 (softmax stability), 493-511 (optimizer update)

</details>

---

### Challenge 4: Single-Name Training Data

**Setup:** The model trains on names from `names.txt` (~32K names). Suppose the file contained only one name, "anna", repeated.

**Question:** After training to convergence, what would the model generate? Would the loss reach zero?

<details>
<summary>Reveal Answer</summary>

**Answer:** The model would reliably generate "anna" (or close variants), but the loss would not reach exactly zero. It would converge to a small positive value.

**Why:** With one training example, the model memorizes the sequence `[BOS, a, n, n, a, BOS]`. At convergence, it assigns high probability to each correct next token, but softmax can never output exactly 1.0 for any class -- it can only approach it asymptotically as logits go to infinity. The loss `-log(p(target))` approaches 0 as `p(target)` approaches 1, but never reaches it. Additionally, position 0 must predict 'a' given BOS (learnable), position 2 must predict 'n' given 'n' at position 1 (ambiguous: the model sees the same context at positions 1 and 2 but must predict 'n' then 'a'). The model resolves this via positional embeddings (`wpe`), which distinguish position 2 from position 3 even when the token embedding is the same.

**Script reference:** `01-foundations/microgpt.py`, lines 209-212 (wpe embedding), 328-330 (tok_emb + pos_emb addition), 466-480 (training loop)

</details>

---

### Challenge 5: KV Cache and Causal Masking

**Setup:** During the forward pass (line 300), keys and values are appended to per-layer caches: `keys[layer_idx].append(k)` (line 354). The attention loop then iterates over all cached keys: `for t in range(len(k_head))` (line 373).

**Question:** At position 5, how many keys are in the cache for each layer? Does the model ever attend to future tokens? What would happen if you pre-filled the cache with random vectors before starting the forward pass?

<details>
<summary>Reveal Answer</summary>

**Answer:** At position 5, the cache contains 6 keys (positions 0 through 5). The model never attends to future tokens. Pre-filling the cache with random vectors would inject noise that the model treats as legitimate past context.

**Why:** The cache starts empty (`keys = [[] for _ in range(N_LAYER)]`, line 461), and each call to `gpt_forward` appends exactly one key and value. At position `t`, the cache has `t+1` entries. The attention loop on line 373 computes `range(len(k_head))` which is `t+1`, covering positions 0 through t. Future positions haven't been appended yet, so they're invisible -- this is how the KV cache provides causal masking without an explicit triangular mask matrix (noted in the comment on lines 389-393). If you pre-filled the cache with garbage vectors, the model would attend to them as if they were real past tokens, corrupting the attention-weighted sum and producing degraded outputs.

**Script reference:** `01-foundations/microgpt.py`, lines 354-355 (cache append), 371-374 (attention over cache), 389-393 (causal masking comment), 461-462 (cache initialization)

</details>

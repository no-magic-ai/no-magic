# MicroKV Challenges

Test your understanding of KV cache by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Multiply Count Growth

**Setup:** `linear_f` (lines 208-211) adds `len(w) * len(x)` to a per-step `counter` for every matrix-vector multiply, and the attention loops add one count per scalar multiply. Both generators start from the BOS token and generate `GEN_LEN = 16` tokens greedily (line 51). `generate_no_cache` (lines 233-308) re-runs every position seen so far at each step; `generate_with_cache` (lines 318-396) processes only the newest token and reuses stored keys and values. The model has `N_EMBD = 16`, `N_HEAD = 2`, `N_LAYER = 1` and a 27-symbol vocabulary.

**Question:** At step `n` (1-based), how many positions does each generator push through the projections and MLP? What are the multiply counts at step 1 and step 16, and how do the totals over 16 steps compare?

<details>
<summary>Reveal Answer</summary>

**Answer:** Without the cache, step `n` processes all `n` positions; with the cache, it processes one. Step 1 costs 3,536 multiplies either way. Step 16 costs 53,936 without the cache and 4,016 with it (13.4x). Over 16 steps the totals are 450,816 and 60,416, a ratio of 7.5x, which is the `Ratio` the script prints. The gap keeps widening with length: the no-cache total grows roughly quadratically and the cached total roughly linearly.

**Why:** Per position, the Q, K and V projections cost 3 × 16 × 16 = 768, the output projection 256 and the MLP 16 × 64 + 64 × 16 = 2,048, so 3,072 multiplies. The output head costs 27 × 16 = 432 once per step. Attention over `t` cached positions costs 2 × (8 + 8) × `t` = 32`t` for two heads. With the cache, step `n` costs 3,072 + 32`n` + 432 = 3,504 + 32`n`. Without it, step `n` costs 3,072`n` + 16`n`(`n` + 1) + 432, because position `p` attends to `p` + 1 positions. Running both generators with random weights reproduces every per-step count; the counts do not depend on the weight values.

**Script reference:** `03-systems/microkv.py`, lines 208-211 (`linear_f` counting), lines 233-308 (no-cache generation recomputing every position), lines 318-396 (cached generation processing one token per step), lines 540-549 (per-step table and totals)

</details>

---

### Challenge 2: Cache Memory Formula

**Setup:** After each cached step the script records `total_cached_floats = 2 * N_LAYER * N_EMBD * len(kv_cache[0]['k'])` (line 393). `N_LAYER = 1` (line 39) and `N_EMBD = 16` (line 37). Each cached position stores one key vector and one value vector of size `N_EMBD` per layer.

**Question:** After the 16 generation steps, how many floats does the cache hold, and how many bytes is that at float32 (the script's table) and at FP16? How large would it be after 10,000 tokens?

<details>
<summary>Reveal Answer</summary>

**Answer:** The cache holds one entry per processed position, 16 in all: `2 * 1 * 16 * 16 = 512` floats, which the table prints as 2,048 bytes at float32 (1,024 bytes at FP16). After 10,000 tokens it would hold 320,000 floats, 1.28 MB at float32 or 640 KB at FP16. The cache grows linearly in sequence length, 32 floats per position here.

**Why:** The formula `2 * N_LAYER * N_EMBD * T` counts the K and V tensors, each layer's own copy and `N_EMBD` values per token. This is the trade KV caching makes: memory that grows linearly with length in exchange for not recomputing every prefix. For a model with 32 layers, width 4,096 and one key/value head per query head, FP16 storage costs `2 * 32 * 4096 * 2 = 524,288` bytes, about 0.5 MB, per token, so a 100,000-token context needs about 52 GB per sequence. That is why grouped-query attention, which stores fewer key/value heads, and paged cache management exist.

**Script reference:** `03-systems/microkv.py`, lines 37-39 (N_EMBD, N_LAYER), lines 353-354 (K/V append), lines 392-394 (cache size bookkeeping), lines 561-573 (memory growth table)

</details>

---

### Challenge 3: Paged Attention Block Boundaries

**Setup:** `simulate_paged_attention` (lines 406-447) maps each position of one sequence to a fixed-size block: `logical_block = pos // block_size` and `slot_in_block = pos % block_size`, allocating a new physical block whenever a position enters a logical block that has none yet. The script calls it with the 16 generated positions and `PAGE_BLOCK_SIZE = 4` (line 52).

**Question:** How many blocks does the script's 16-position trace allocate, and how many slots does it waste? If you called it with 5, 3 and 7 positions instead, how many blocks and wasted slots would each call report, and how does that compare with reserving room for 16 positions per sequence up front?

<details>
<summary>Reveal Answer</summary>

**Answer:** 16 positions fill exactly 4 blocks of 4, so nothing is wasted. For 5, 3 and 7 positions the calls report 2, 1 and 2 blocks with 3, 1 and 1 wasted slots: 5 blocks and 5 wasted slots in total. Reserving 16 slots per sequence would use 48 slots for 15 tokens and waste 33.

**Why:** Paged allocation only ever wastes the unfilled tail of each sequence's last block, at most `block_size - 1` slots. Up-front reservation wastes the gap between the reserved maximum and the actual length, which is unknown in advance during generation. The page table also lets blocks live anywhere in the pool, so a new sequence can take any free block. The simulation here traces a single sequence; as its signpost comment notes, sharing physical blocks between sequences is not simulated here (see `micropaged.py` for multiple requests and copy-on-write).

**Script reference:** `03-systems/microkv.py`, lines 406-447 (simulate_paged_attention), lines 418-430 (logical-to-physical mapping), lines 440-443 (sharing signpost), line 52 (PAGE_BLOCK_SIZE)

</details>

---

### Challenge 4: Why Cached Attention Still Needs All Previous K/V

**Setup:** In `generate_with_cache`, the new token's query is computed from the new token alone (line 348), its key and value are appended to the cache (lines 353-354), and attention then runs over every cached position: `for t in range(cached_len)` (lines 363 and 371).

**Question:** Could you save memory by keeping only the last `W` positions in the cache (a sliding window)? What would break if you did this for the model in this script?

<details>
<summary>Reveal Answer</summary>

**Answer:** The outputs would stop matching the no-cache generator once the sequence is longer than `W`: the model would lose access to tokens older than `W` steps that it was trained to attend to. The script's `assert toks_no_cache == toks_cached` (line 529) would then be able to fail.

**Why:** Causal attention lets the token at position `t` attend to all positions `0..t`. The exact output is `softmax(q_t @ K[0..t].T / sqrt(d)) @ V[0..t]`; dropping old rows of `K` and `V` changes the softmax normalization and removes their contribution, so the result is a different function, not a compressed copy. This model was trained with full causal attention over its 32-position context, so windowing at inference is an approximation it never saw. Sliding-window attention works in models trained with that window, such as Mistral 7B, because training teaches them not to rely on tokens beyond it.

**Script reference:** `03-systems/microkv.py`, lines 348-354 (new-token projections and cache append), lines 363-374 (attention over all cached positions), line 529 (equality assertion)

</details>

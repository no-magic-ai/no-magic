# MicroTokenizer Challenges

Test your understanding of Byte-Pair Encoding by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Overlapping Pair Merge Order

**Setup:** In `apply_merge` (line 66), the merge loop processes the sequence left-to-right with `i += 2` after consuming a matched pair (line 81). Consider the input sequence `[a, a, a]` and the merge rule `(a, a) -> new`.

**Question:** After applying the merge, what is the output sequence? Would the output differ if the loop processed from right-to-left instead?

<details>
<summary>Reveal Answer</summary>

**Answer:** The output is `[new, a]`. A right-to-left scan would produce `[a, new]`.

**Why:** The left-to-right scan consumes position 0 and 1 (producing `new`) and then increments `i` by 2, landing at position 2. Position 2 is the leftover `a`, which cannot form a pair with nothing. The docstring on lines 69-71 explicitly states "Overlapping pairs resolve left-to-right." This determinism is essential: if the merge direction were data-dependent, the same string could tokenize differently on different runs, breaking the invariant that tokenization is a pure function of the input string.

**Script reference:** `01-foundations/microtokenizer.py`, lines 66-85 (`apply_merge`, especially lines 79-84 and the comment on lines 70-75)

</details>

---

### Challenge 2: Encoding vs. Re-Counting

**Setup:** The `encode` function (line 145) iterates over merges in priority order: `for pair, new_id in merges: token_ids = apply_merge(token_ids, pair, new_id)` (lines 159-160). It does NOT re-count pair frequencies on the new text after each merge.

**Question:** Why does encode replay merges in learned priority order rather than re-counting frequencies on the input text? What would go wrong if encode re-counted and re-ranked after each merge?

<details>
<summary>Reveal Answer</summary>

**Answer:** Replay applies the merges exactly in the priority training learned, so the output is a fixed function of the full input string and the learned merge list. Re-ranking by counts in the input would apply the same merges in a different order whenever the input's local counts disagree with the training order, and that can change the segmentation. It would not make encoding nondeterministic — `encode` takes one string, and a re-ranking encoder would also return the same tokens every time for that string — and each token ID would still name the same learned byte sequence. Neither encoder makes a word's tokens independent of its neighbours: this script has no pre-tokenization, so learned merges can span spaces and newlines.

**Why:** Overlapping merges compete for the same byte. Train two merges with the script's `train_bpe` on `"ab1ab2ab3bc4bc5"`: it learns `(a, b)` first and `(b, c)` second. Encode `"abcbc"`. Replay applies `(a, b)` first and gives `ab | c | bc`. Re-ranking the same two merges by their counts in this input applies `(b, c)` first (it occurs twice) and gives `a | bc | bc`. Both encoders are deterministic, and the token IDs for `ab` and `bc` mean the same bytes in both; only the order of the learned merges changed, and with it the segmentation. Replay does not make a word's tokens context-free, though: train three merges on `"ab ab ab ab cd cd ab cd"` and the second merge is `(ab, ' ')`, so replay encodes `"ab"` as `ab` but `"ab cd"` as `ab␣ | c | d`. The docstring on lines 148-153 justifies the design by determinism ("the same string always produces the same token sequence") and by "input batch" dependence; that rationale is incorrect, because re-counting a single string is also deterministic and `encode(text, merges)` has no batch. The behaviour it documents, replaying merges in learned order, is unchanged and correct.

**Script reference:** `01-foundations/microtokenizer.py`, lines 145-161 (`encode` function), lines 146-153 (docstring with the incorrect rationale)

</details>

---

### Challenge 3: The Empty String Edge Case

**Setup:** The `encode` function (line 145) begins with `token_ids = list(text.encode("utf-8"))` (line 158). The round-trip test on line 198 includes the empty string `""`.

**Question:** What does `encode("")` return? What does `decode(encode(""), vocab)` return? Does the empty string cause any errors or special cases in the merge loop?

<details>
<summary>Reveal Answer</summary>

**Answer:** `encode("")` returns `[]` (an empty list). `decode([], vocab)` returns `""`. No errors occur.

**Why:** `"".encode("utf-8")` is `b""`, so `list(b"")` is `[]`. In the merge loop (line 159), `apply_merge([], pair, new_id)` enters `apply_merge` with an empty `merged = []` and immediately returns it — the `while i < len(token_ids)` loop (line 78) never executes because `len([]) == 0`. Decode (line 171) does `b"".join(...)` over an empty iterator, which is `b""`, and `b"".decode("utf-8")` is `""`. The round-trip identity `decode(encode(text)) == text` holds for the empty string as a degenerate case.

**Script reference:** `01-foundations/microtokenizer.py`, lines 158-161 (encode start), lines 76-85 (apply_merge loop), lines 170-172 (decode), line 198 (empty string test case)

</details>

---

### Challenge 4: Vocabulary Size and Compression

**Setup:** `NUM_MERGES = 256` (line 34), giving a final vocabulary of `256 + 256 = 512` tokens. The final vocabulary size and the corpus compression ratio are computed at lines 216-220.

**Question:** If you doubled `NUM_MERGES` to 512, would the compression ratio approximately double? Would the vocabulary size double?

<details>
<summary>Reveal Answer</summary>

**Answer:** The vocabulary size would increase from 512 to 768 (adding 256 more entries), so it does not double. The compression ratio would improve but would NOT double — it follows a diminishing returns curve.

**Why:** The vocabulary grows linearly with `NUM_MERGES` (always +1 per merge, line 113). But compression ratio follows diminishing returns: early merges eliminate the most frequent pairs (massive corpus length reduction), while later merges eliminate rarer pairs (small incremental gain). The comment on line 36 notes that production tokenizers use 50K+ merges on hundreds of gigabytes. At that scale, each additional merge above ~10K provides very little additional compression on a small corpus like names.txt because the most compressible structure has already been captured by the early merges.

**Script reference:** `01-foundations/microtokenizer.py`, lines 34-36 (NUM_MERGES constant and comment), lines 111-116 (merge loop with new_id computation), lines 211-220 (compression ratio calculation)

</details>

---

### Challenge 5: Corpus Collapse

**Setup:** Inside `train_bpe` (line 88), after each merge `ids = apply_merge(ids, pair, new_id)` (line 115). The loop also checks `if not counts: break` (line 106-109).

**Question:** Under what real condition would the corpus collapse check at line 106 trigger? On the names.txt corpus with `NUM_MERGES = 256`, will it trigger?

<details>
<summary>Reveal Answer</summary>

**Answer:** It triggers when the entire corpus is compressed into a single token — there are no adjacent pairs left. On names.txt with 256 merges, it will NOT trigger because 256 merges is far fewer than what's needed to fully collapse a 228,145-byte corpus.

**Why:** After each merge, the corpus shrinks because each pair occurrence is replaced by a single token. Full collapse requires enough merges to reduce the corpus to one token, and the final token must spell out the whole corpus. Each merge adds one token to the vocabulary, so for text without long exact repeats this takes on the order of n merges for a corpus of length n. names.txt has 228,145 bytes of 32,033 names, so 256 rounds of pair elimination leave it far from a single token. The collapse check is a defensive guard for cases like a corpus consisting of a single unique byte repeated many times (e.g., `[65, 65, 65, 65]` → 2 merges to fully collapse), not for realistic corpora.

**Script reference:** `01-foundations/microtokenizer.py`, lines 104-116 (training loop with collapse check), lines 55-63 (get_pair_counts, which returns empty Counter when only 1 token remains)

</details>


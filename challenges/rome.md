# MicroROME Challenges

Test your understanding of rank-one model editing by predicting what happens in these scenarios. Try to work out the answer before revealing it. Numbers quoted in the answers come from the default seed-42 run under CPython 3.12.8 or from probes that call the script's own functions; older interpreters can print different low-order digits.

---

### Challenge 1: How Much of the Network Does One Edit Touch?

**Setup:** `rome_edit` (lines 714-756) builds the edited matrix as `new_w[i][j] = W[i][j] + lam[i] * u[j]` (line 751), where `lam` is Λ = (v* − W k*) / (uᵀ k*) (lines 747-748) and `u` = C⁻¹ k* (line 746). `W_proj` is 16 × 64 (lines 86-87). In `main`, the edited network is a copy of the trained one in which only `"Wproj"` is replaced (lines 1012-1013).

**Question:** How many of the 3,936 parameters does one edit change, what rank does the change ΔW have, and does any other weight tensor move?

<details>
<summary>Reveal Answer</summary>

**Answer:** All 1,024 entries of `W_proj` change and nothing else does. ΔW has rank exactly 1, which every edit line confirms with `rank=1`.

**Why:** ΔW[i][j] = Λ_i u_j is an outer product of a 16-vector and a 64-vector. Every row is a multiple of uᵀ, so the rows span one direction. Because Λ and u have no zero entries here, every product is nonzero. A probe on the script's own trained network found 1,024 changed entries for all six edits. The largest 2 × 2 minor of ΔW was below 5e-16 times max|ΔW|², and only `Wproj` differed from the trained tensors. "One rank-one update" does not mean "one weight": the edit is dense but has a single direction. For any incoming key k it adds Λ times the scalar uᵀk. The runtime hard checks (lines 759-779 and lines 1015-1025) assert the rank, `W_hat k* = v*`, ΔW = Λuᵀ entrywise, and that no other tensor moved.

**Script reference:** `02-alignment/microrome.py`, lines 714-756 (`rome_edit`), line 751 (W_hat = W + Λuᵀ), lines 759-779 (`mechanical_checks`), lines 1012-1025 (edited copy and unchanged-tensor check)

</details>

---

### Challenge 2: Same Value, Different Address

**Setup:** The C = I control (`variant == "identity"`, lines 743-744) sets `u = k*` instead of solving `C u = k*`. Look at where `v*` is computed (line 740): `optimise_value` receives the trained network, the subject, the target and the edit contexts, but not `cov`. In the default output, each `identity` line follows the `rome` line for the same edit.

**Question:** Will the `identity` line print the same `steps` and loss `L` as the `rome` line? What happens to the two neighbors that share the edited subject's old city?

<details>
<summary>Reveal Answer</summary>

**Answer:** Yes. Every `identity` line repeats the `rome` line's `steps` and `L:...->...` exactly, because `v*` is the same vector; only the "address" u changes. Neighbors suffer. Mean neighborhood success falls from 1.0000 (ROME) to 0.6865 (C = I). For `thom` it is 0.1625, with 7.47% of the other subjects' prompts newly answering the edit target (`BLEED=0.0747`).

**Why:** For another subject's key k, the update adds Λ (uᵀk). A probe on the script's own network measured uᵀk / uᵀk* for each neighbor key with no prefix. With u = C⁻¹k* the six edits gave values between −0.04 and 0.11. With u = k* they gave 0.19 to 1.04, so a neighbor can receive almost the full edit. ReLU keys are nonnegative and overlap, so k*ᵀk is large. C⁻¹ weights the directions that ordinary keys share least, and that is what keeps the edit away from other facts. The constraint `W_hat k* = v*` holds equally well for both variants (|Wk*−v*| ≈ 1e-16 on every line), so exactness at k* alone says nothing about locality. In the default run the C = I edit of `bavo` even fails its own efficacy (ES=0.0000) while satisfying the constraint to 4.4e-16.

**Script reference:** `02-alignment/microrome.py`, line 740 (`optimise_value` call, no covariance), lines 743-748 (u, uᵀk*, Λ), lines 647-664 (`estimate_covariance`), lines 804-883 (`evaluate`; neighborhood on line 858)

</details>

---

### Challenge 3: Exact at k*, Partial Elsewhere

**Setup:** The key k* is the mean of the subject's keys over 20 edit prefixes (line 735). A prefix holds 0-3 filler tokens, and nothing mixes positions before the MLP, so the subject's key depends only on the subject token and its position (lines 292-298). The default run prints this for `pelm`: `steps=20 L:46.147->0.047 ... |Wk*-v*|=6.7e-16`. That is below the early-stop threshold 0.05 (line 597) and satisfies the constraint to round-off.

**Question:** Will `pelm`'s efficacy be 1.0000? If not, which evaluation prompts fail?

<details>
<summary>Reveal Answer</summary>

**Answer:** No. `pelm` prints `ES=0.8000`, and the four failures are exactly the four evaluation prompts with no prefix. The subject sits at position 0 there, and the old city Elsk still wins.

**Why:** For the key at position p, the edited matrix adds Λ (uᵀk_p), so the MLP output becomes `W k_p + (uᵀk_p / uᵀk*) (v* − W k*)`. That equals v* only if k_p = k*. A probe on the script's own network measured uᵀk_p / uᵀk* for `pelm` as 0.864, 0.982, 0.980 and 1.068 at positions 0-3. Position 0 receives only about 86% of the change, and there all 4 evaluation prompts fail. Every prompt at positions 1-3 succeeds (16 of 16), giving 16/20 = 0.80. A value can converge on its fitting contexts yet transfer only partly, and more optimisation steps do not fix this. It is one reason the efficacy threshold holds at seed 42 but not at most other seeds (see the script's deviation list, item 10).

**Script reference:** `02-alignment/microrome.py`, lines 724-736 (k* and z0 averaged over contexts), lines 575-613 (`optimise_value`, early stop on line 597), lines 292-299 (key and MLP output per position)

</details>

---

### Challenge 4: Editing the Wrong Token

**Setup:** The `wrong_site` control (line 722) takes both the key and the injected value at the last position, the T0 template token, instead of the subject (lines 730 and 553). With no attention before the MLP, the key at that position is computed from the template token and its position alone (lines 292-298).

**Question:** What paraphrase score (T1-T3 prompts) will the wrong-token edit get? Will it stay local to the edited subject?

<details>
<summary>Reveal Answer</summary>

**Answer:** Paraphrase is 0.0000 on all six edits. The edit is not local: mean locality drops to 0.9515 (ROME: 1.0000), and up to 5.43% of the other subjects' prompts flip to the target (`pelm`: `BLEED=0.0543`). Its efficacy is erratic: 0.0000 for `tavi`, 1.0000 for `doph` and `thom`, mean 0.5250. The value search also hit the 60-step cap on 4 of the 6 edits.

**Why:** The key at the template position carries no information about which subject is in the prompt, so the edit changes the T0 readout of every subject alike instead of one subject's fact. It was fitted only at the T0 key, and nothing ties the edited subject's T1-T3 prompts, whose last-token keys come from other template tokens, to the new city. This outcome is fixed by the architecture, not discovered: it illustrates why a key must carry the subject's identity. It is not evidence for the paper's finding that the subject site matters in GPT. The paper locates that site with causal tracing in a model where attention runs before the MLP. Here the trace (lines 886-929) only confirms a route the architecture imposes.

**Script reference:** `02-alignment/microrome.py`, line 722 (`site`), line 730 and lines 553/562 (position used for key and value), lines 264-329 (`forward`), lines 886-929 (causal trace)

</details>

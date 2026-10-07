# MicroLoRA Challenges

Test your understanding of Low-Rank Adaptation by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: The B-Matrix Zero Initialization

**Setup:** `make_lora_B` (line 172) initializes LoRA B to all zeros. `make_lora_A` (line 164) initializes A with small random noise `~ N(0, 0.02)`. The adapted output is `base_out + A @ (B @ x)` in `lora_linear` (line 237), where B (shape `[LORA_RANK, N_EMBD]` = 2×16) projects the 16-dimensional input down to rank 2 and A (shape `[N_EMBD, LORA_RANK]` = 16×2) projects it back up.

**Question:** At the start of LoRA adaptation, what does the adapter contribute to the output? On the first update, which of A and B receives a nonzero gradient? How does this compare with the initialization in the LoRA paper?

<details>
<summary>Reveal Answer</summary>

**Answer:** The adapter contributes exactly zero (`B @ x = 0`, so `A @ 0 = 0`). On the first update only B, the down-projection, gets a nonzero gradient; A, the up-projection, gets exactly zero and is unchanged by the first Adam step. The paper does the reverse: in its `W₀ + BA` it sets the up-projection B (d×r) to zero and draws the down-projection A (r×k) from a Gaussian, so its up-projection moves first. The script's zero factor and the paper's zero factor sit on opposite sides of the rank bottleneck; this is a different initialization, not the same one with the letters swapped.

**Why:** Write `h = B @ x` (the rank-2 middle vector) and `y = W @ x + A @ h`. Then `∂L/∂A = (∂L/∂y) hᵀ`, which is zero while `h = 0`, and `∂L/∂B = (Aᵀ ∂L/∂y) xᵀ`, which is generally nonzero because A is random (it vanishes only if `Aᵀ ∂L/∂y` or `x` is zero). The zero for A holds for every input; the size of B's gradient depends on the name, the base weights and the seed. Executing the script's own adaptation loop (lines 549-586) for one step, on an untrained base with the name `olivia`, gave exactly 0 for A and a nonzero gradient for B; after the update B had changed and A and every base weight had not. From the second step on, `h` is no longer zero and A starts to learn. Either placement of the zero keeps the adapted model identical to the base at the start, which is the property the comment on lines 173-179 describes; they differ in which factor learns first. If both factors were random, the adapted model would start from the base plus a random rank-2 offset rather than from the base itself.

**Script reference:** `02-alignment/microlora.py`, lines 164-179 (make_lora_A and make_lora_B initialization with comments), lines 237-264 (lora_linear showing the A @ (B @ x) computation), lines 549-586 (adaptation loop)

</details>

---

### Challenge 2: Gradient Zeroing for Frozen Weights

**Setup:** During LoRA adaptation (lines 549-586), `loss.backward()` (line 565) runs on the full computation graph, which includes both the frozen base parameters and the adapter parameters. Then base gradients are zeroed: `for p in base_param_list: p.grad = 0.0` (lines 572-573), and Adam updates `lora_param_list` (lines 577-583).

**Question:** Which line actually keeps the base weights frozen? What would happen if you deleted lines 572-573? Could you avoid computing base gradients at all?

<details>
<summary>Reveal Answer</summary>

**Answer:** The optimizer loop does the freezing: it iterates only over `lora_param_list`, so base weights are never updated whether or not their gradients are zeroed. Deleting lines 572-573 would leave the base `.grad` fields accumulating across steps, but nothing reads them, so the run would produce the same numbers. You could avoid computing them by passing the frozen weights in as plain floats instead of `Value` nodes.

**Why:** `loss.backward()` walks every `Value` that contributed to the loss and sets its `.grad`, including the frozen weights, because `base_out = linear(x, w_frozen)` makes them parents of the output. This engine has no per-node "requires grad" switch, so with `Value` weights those gradients are always computed. Zeroing them only throws them away; the comment on lines 567-571 calls it "the core LoRA mechanism", but the mechanism is that only `A` and `B` are passed to the update. Storing the frozen weights as floats (as `microdpo.py` does for its reference model) would remove them from the graph entirely and save the work.

**Script reference:** `02-alignment/microlora.py`, lines 549-586 (adaptation loop), lines 565-573 (backward and gradient zeroing), lines 577-583 (update of LoRA parameters only), lines 259-264 (lora_linear linking frozen weights into the graph)

</details>

---

### Challenge 3: Rank and Parameter Count

**Setup:** `LORA_RANK = 2` (line 42), `N_EMBD = 16` (line 35). `init_lora_adapters` creates Q and V adapters per layer. A-matrix is `[N_EMBD, LORA_RANK]` (line 219), B-matrix is `[LORA_RANK, N_EMBD]` (line 220). `N_LAYER = 1` (line 37).

**Question:** How many trainable LoRA parameters are there in total? The base model has 4,192 parameters (printed by line 498). What percentage does LoRA add? If you tripled `LORA_RANK` to 6, how many LoRA parameters would there be?

<details>
<summary>Reveal Answer</summary>

**Answer:** With rank=2, there are `2 * (16*2 + 2*16) = 2 * (32 + 32) = 128` LoRA parameters (Q adapter + V adapter, A+B each). That is 128 / 4,192 = 3.1% of the base model, the figure the results line prints. With rank=6: `2 * (16*6 + 6*16) = 2 * (96 + 96) = 384` parameters, 9.2%.

**Why:** Each LoRA adapter consists of matrix A with shape `[N_EMBD, LORA_RANK]` = 16×2 = 32 values, and matrix B with shape `[LORA_RANK, N_EMBD]` = 2×16 = 32 values. One adapter = 64 parameters. There are 2 adapters (Q and V) × 1 layer = 2 adapters × 64 = 128 parameters total. The script reports these at lines 541-543. With rank r, each adapter has `N_EMBD * r + r * N_EMBD = 2 * r * N_EMBD` parameters. The base model has 4,192 parameters, so rank=2 adds 128 / 4,192 ≈ 3.1%. Tripling rank multiplies LoRA parameters by 3x (linear relationship), reaching 384 / 4,192 ≈ 9.2%.

**Script reference:** `02-alignment/microlora.py`, lines 42 (LORA_RANK), lines 219-224 (adapter shapes), lines 541-543 (parameter count reporting), line 593 (pct calculation)

</details>

---

### Challenge 4: Why Q and V, Not K?

**Setup:** The comment at lines 207-211 explains: "Why Q and V, not K or O? The original LoRA paper found that adapting Q and V projections captures the most task-relevant information per parameter. Intuitively: Q controls 'what to look for' and V controls 'what to extract' — both are highly task-specific."

**Question:** The base model trains on A-M names and the adapters on N-Z names; afterwards the script reports the loss of the base and the adapted model on both splits (lines 616-626). What does the frozen base guarantee about A-M performance, and what does it not guarantee?

<details>
<summary>Reveal Answer</summary>

**Answer:** It guarantees that the base model is still there: removing the adapters restores the base outputs exactly, because `W_frozen` never changes. It does not guarantee that the adapted model keeps its A-M loss, because the adapters stay switched on when the A-M split is evaluated. Whether A-M loss barely moves or rises is what the run measures.

**Why:** Each adapted projection computes `W_frozen @ x + A @ (B @ x)`. The added term always lies in the column space of A, a 2-dimensional subspace of the 16-dimensional output, so the adapter can change the query or value vector only along 2 directions and leaves the component orthogonal to that subspace exactly as the base produces it. That bounds the kind of change, not its size: along those 2 directions the shift can be large enough to hurt A-M predictions. The LoRA paper reports that, for a fixed parameter budget, adapting the query and value projections together worked best among the attention-weight combinations it tried; it did not study the MLP, so this script offers no evidence either way about adapting MLP weights instead.

**Script reference:** `02-alignment/microlora.py`, lines 204-226 (Q and V adapter rationale), lines 247-264 (lora_linear showing additive structure), lines 612-626 (cross-evaluation on both splits)

</details>

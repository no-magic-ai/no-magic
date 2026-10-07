# MicroSSM Challenges

Test your understanding of State Space Models (Mamba-style) by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Fixed State Size vs Growing KV Cache

**Setup:** The SSM state holds `N_STATE` values for each of the `N_EMBD` channels, `N_STATE * N_EMBD` in all (`ssm_state` in `selective_scan`, lines 337-438). `N_STATE = 8` (line 38), `N_EMBD = 16` (line 37). Compare to the KV cache in `microkv.py`, which grows by `2 * N_EMBD` values (a key and a value) per layer per new token.

**Question:** After processing a 1,000-token sequence, how much memory does the SSM state use (in floats)? How does this compare to a KV cache for the same sequence with `N_LAYER = 2`, `N_EMBD = 16`? What is the fundamental tradeoff this size difference represents?

<details>
<summary>Reveal Answer</summary>

**Answer:** SSM state: `N_STATE * N_EMBD = 8 * 16 = 128 floats` — the same regardless of sequence length (1 token or 1,000 tokens). KV cache after 1,000 tokens: `2 * N_LAYER * N_EMBD * T = 2 * 2 * 16 * 1000 = 64,000 floats`. The SSM uses 500x less memory at T=1,000, and the ratio grows linearly with sequence length.

**Why:** The SSM compresses the entire history of processed tokens into a fixed-size "state" vector `h`. At each step, the state is updated via `x_new = a_bar * ssm_state[state_idx] + b_bar * u[d]` (line 429), discarding the raw token and keeping only what the learned `A`, `B`, `C` matrices say is worth remembering. This is fundamentally lossy — information from 1,000 tokens ago may be attenuated or lost depending on the `A` eigenvalues. The KV cache is lossless: every token's key and value vector is stored verbatim, allowing perfect recall of any past token (subject to attention weight). The SSM's O(1) memory per step enables processing arbitrarily long sequences without memory growth, at the cost of potentially forgetting long-range details. Transformers with KV cache guarantee exact recall but require O(T) memory.

**Script reference:** `03-systems/microssm.py`, lines 38 (N_STATE), lines 37 (N_EMBD), lines 337-438 (selective_scan), line 429 (state update formula), line 524 (state size printed)

</details>

---

### Challenge 2: The Delta Bias and Input-Dependent Timescales

**Setup:** `delta_bias` is initialized to `-2.0` for every channel (line 314). The discretization computes `delta = softplus(W_delta @ u + delta_bias)` (lines 385-393). `A_diag[n] = -exp(log_A[n])` (line 401), with `log_A` initialized from `N(-1.0, 0.3)` (line 306), and `a_bar = 1 + delta_d * A_diag[n]` (line 421) uses Euler discretization (not ZOH).

**Question:** When the projection `W_delta @ u` is 0 for a channel, what is `delta` after softplus with the bias of `-2.0`? What does a small `delta` mean for how much the state changes at this timestep? Why is the default small rather than large?

<details>
<summary>Reveal Answer</summary>

**Answer:** `delta = softplus(0.0 + (-2.0)) = softplus(-2.0) = log(1 + exp(-2.0)) ≈ log(1 + 0.135) ≈ log(1.135) ≈ 0.127`. This is a small positive number. At the centre of the initialization, `log_A = -1.0`, so `A_diag = -exp(-1) ≈ -0.368` and `a_bar = 1 + 0.127 * (-0.368) ≈ 0.953` — close to 1, meaning the state barely changes (high retention of previous state), and `b_bar = 0.127 * B_k[n]` keeps the new input's contribution small.

**Why:** The `delta` parameter controls the "step size" of the discretization — how aggressively the continuous-time SSM dynamics are applied to one discrete token step. A large `delta` means the state rapidly forgets the past (the continuous-time system is allowed to evolve far) and incorporates the new input strongly. A small `delta` means the state is nearly frozen (the system barely evolves), preserving the existing memory. The default bias of `-2.0` makes the network "conservative by default" — most tokens make small updates to the state. This mirrors how language works: most tokens are background context, only occasionally does a key token (a name, a number, a negation) require a large state update. The input-dependent projection `W_delta @ u` lets the network learn to open the gate wide for important tokens. This selectivity is what distinguishes Mamba from S4 (which has fixed `delta`).

**Script reference:** `03-systems/microssm.py`, line 306 (log_A initialization), lines 309-314 (delta projection and bias), lines 385-393 (delta computation via softplus), lines 399-402 (A_diag from log_A), lines 421-424 (Euler discretization with delta)

</details>

---

### Challenge 3: Selective B and C vs Fixed

**Setup:** In `selective_scan` (lines 337-438), `B` and `C` are computed per-token from the input: `B_k = linear(u, W_B)` (line 377) and `C_k = linear(u, W_C)` (line 378). The `init_ssm_params` docstring (lines 280-284) contrasts this with classical SSMs such as S4, where `B` and `C` are fixed learned matrices.

**Question:** If `B` and `C` were fixed (not input-dependent), what would the SSM lose compared to the selective version? Give a concrete example of why selectivity matters for language.

<details>
<summary>Reveal Answer</summary>

**Answer:** With fixed `B` and `C`, every token would write the same relative amounts to the state (fixed `B`) and read from the state in the same pattern (fixed `C`). The model could not learn to strongly memorize a specific token (like a name) when it first appears and then precisely retrieve it many tokens later.

**Why:** Fixed `B` means the "input gate" applies the same projection to every token regardless of content. A pronoun "she" and a proper name "Maria" would update the state with identically-scaled projections, making it impossible to give the name a stronger memory trace. Fixed `C` means the "output gate" retrieves from the state in the same pattern for every query — the model cannot learn to "look up" a specific slot in the state when generating a token that depends on a specific earlier word. With selective `B` and `C` computed from the current input `x_t` (via learned linear projections `W_B` and `W_C`), the network can learn: when processing "Maria" (content-dependent), use a large `B` component to write the name's representation strongly into a specific state dimension; when generating a pronoun that should agree with "Maria", use a large `C` component on that same dimension to retrieve it. This input-dependent memory access (together with an input-dependent `delta`) is the selection mechanism that the Mamba paper adds to prior SSMs such as S4.

**Script reference:** `03-systems/microssm.py`, lines 337-375 (selective_scan docstring), lines 377-378 (input-dependent B_k and C_k), lines 280-284 (fixed vs selective comparison), lines 323-324 (W_B and W_C initialization)

</details>

---

### Challenge 4: Euler vs ZOH Discretization

**Setup:** The discretization uses the Euler method: `a_bar = 1 + delta_d * A_diag[n]` (line 421), `b_bar = delta_d * B_k[n]` (line 424). The comment on lines 264-267 notes that the exact zero-order-hold (ZOH) discretization would use `a_bar = exp(delta * A)` and `b_bar = A^(-1) * (exp(delta * A) - I) * B`. `A_diag[n] = -exp(log_A[n])` (line 401), with each `log_A[n]` drawn from `N(-1.0, 0.3)` (line 306).

**Question:** Take three state dimensions with `log_A` at -1.6, -1.0 and -0.4 (the centre of the initialization and two standard deviations either side) and `delta = 0.1`. What are `A_diag` and the Euler `a_bar` for each? Can any dimension start as a pure integrator (`a_bar = 1`)? How large must `delta` get before Euler stops being stable for the middle dimension?

<details>
<summary>Reveal Answer</summary>

**Answer:** `A_diag = -exp(log_A)` gives about -0.202, -0.368 and -0.670, so `a_bar = 1 + 0.1 * A_diag` is about 0.980, 0.963 and 0.933: every dimension forgets a little each step, at a different rate. No dimension can be a pure integrator, because `-exp(log_A)` is strictly negative for every finite `log_A`. For the middle dimension, Euler keeps `|a_bar| < 1` only while `delta < 2 / 0.368 ≈ 5.4`; beyond that the state oscillates and grows.

**Why:** Parameterizing `A_diag` as `-exp(log_A)` keeps every continuous-time eigenvalue negative, so the continuous system always decays; the spread of `log_A` gives the dimensions a range of decay rates and therefore of timescales. Euler replaces `exp(delta * A)` with its first-order approximation `1 + delta * A`, which is accurate for small `delta * |A|` (`exp(-0.0368) = 0.9639` versus `0.9632` here) but is not bounded: `delta` comes from an unbounded softplus, and once `delta * |A| > 2` the Euler factor drops below -1. ZOH's `exp(delta * A)` stays in `(0, 1)` for any `delta > 0`. `microdiscretize.py` compares the two methods directly.

**Script reference:** `03-systems/microssm.py`, line 306 (log_A initialization), lines 399-402 (A_diag from log_A), lines 421-424 (Euler discretization in selective_scan), lines 258-267 (Euler and ZOH comment)

</details>

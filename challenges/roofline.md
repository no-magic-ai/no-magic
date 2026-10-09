# Roofline Challenges

Test your understanding of roofline analysis and hardware utilization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Arithmetic Intensity of Vector Addition

**Setup:** The script defines arithmetic intensity as `AI = FLOPs / Bytes`. Consider a vector addition of two 4096-element float64 vectors: read two input vectors and write one output vector. `FLOPs = 4096` (one add per element). `Bytes = 3 * 4096 * 8 = 98304` (two reads + one write, 8 bytes per float64). Assume an M-series CPU with ~100 GB/s bandwidth, ~50 GFLOPS peak compute, and a ridge point at ~0.5 FLOPs/byte.

**Question:** What is the arithmetic intensity? Is this operation memory-bound or compute-bound? How much of the peak compute capacity is actually utilized?

<details>
<summary>Reveal Answer</summary>

**Answer:** `AI = 4096 / 98304 ≈ 0.042 FLOPs/byte`. This is nearly 12x below the ridge point of 0.5 -- deep in memory-bound territory.

Achievable throughput is limited by bandwidth: `0.042 * 100 GB/s = 4.2 GFLOPS`. Peak compute is 50 GFLOPS, so only `4.2 / 50 = 8.4%` of the compute is utilized. The remaining 91.6% of compute capacity sits idle, waiting for data to arrive from memory.

**Why:** Vector addition is the canonical memory-bound operation. Each element requires one floating-point operation but three memory transactions (two loads, one store). No amount of hardware compute scaling helps -- the bottleneck is entirely in the memory subsystem. This is the sloped part of the roofline: below the ridge point, attainable performance is `AI × bandwidth`, a line that rises with arithmetic intensity and does not depend on peak compute. The flat roof is to the right of the ridge, where compute is the limit.

**Script reference:** `03-systems/microroofline.py`, lines 36-38 (assumed peaks and ridge point), lines 165-194 (arithmetic intensity and theoretical throughput), lines 216-224 (vector-add FLOP and byte counts)

</details>

---

### Challenge 2: MIMO Rank and Arithmetic Intensity

**Setup:** `run_ssm_comparison` (lines 416-469) counts, per step, `3 * N * E` FLOPs and `(N*E + N + E) * 8` bytes for the SISO update, and `N*E + 2*N*E*r` FLOPs and `(N*E + N*r + r*E) * 8` bytes for a rank-`r` MIMO update, with `N_STATE = 16` and `N_EMBD = 8`. A common rule of thumb says MIMO intensity grows like `2r`, and the cards quote a GPU ridge point of about 300 FLOPs/byte.

**Question:** Using the script's own byte accounting, what intensities do ranks 1, 16 and 64 reach? Can any rank reach a ridge of 300 here? What would?

<details>
<summary>Reveal Answer</summary>

**Answer:** Rank 1 gives 384 / 1,216 ≈ 0.32 (the same as SISO), rank 16 gives 4,224 / 4,096 ≈ 1.03 and rank 64 gives 16,512 / 13,312 ≈ 1.24. No rank reaches 300 under this accounting: as `r` grows, both FLOPs and bytes grow linearly in `r`, so the intensity levels off at `2NE / (8(N + E)) = 256 / 192 ≈ 1.33`. To approach a GPU ridge point the state and input dimensions themselves must be large, as in the operation table, where a 16×16 outer product sits at 2.00 and a 256×256 rank-16 matrix multiply at 32.00.

**Why:** In this accounting every extra rank brings new data (`N` more values of `B` and `E` more of `X`) along with its `2NE` FLOPs, so the ratio cannot grow without bound. The "`AI ≈ 2r`" rule of thumb assumes the traffic is dominated by reading and writing the state, which does not grow with `r`; that holds when the state is much larger than the per-rank inputs, which is not the case at `N = 16`, `E = 8`. For a matrix multiply `[n × r] @ [r × m]` the script's counts give `2nmr / (8r(n + m)) = nm / (4(n + m))`, which depends on `n` and `m`, not on `r`. The SISO-to-MIMO move raises intensity here, by about 3.3x at rank 16, but the large GPU-scale gains need large matrices.

**Script reference:** `03-systems/microroofline.py`, lines 226-258 (outer-product and matmul counts), lines 294-366 (operation table sizes), lines 416-469 (SISO and MIMO state-update counts)

</details>

---

### Challenge 3: Why More FLOPs Can Be Faster

**Setup:** SISO performs `3 * N * E` FLOPs per step. MIMO rank-16 performs `N * E + 2 * N * E * 16 = 33 * N * E` FLOPs per step -- 11x more. With the byte counts from Challenge 2, SISO moves 1,216 bytes per step and MIMO-16 moves 4,096.

**Question:** On hardware where both updates are memory-bound, how long does a MIMO step take relative to a SISO step, and how does FLOP throughput compare? Does that make MIMO finish the same sequence sooner?

<details>
<summary>Reveal Answer</summary>

**Answer:** Below the ridge point a step takes `Bytes / Bandwidth`, so a MIMO-16 step takes about 4,096 / 1,216 ≈ 3.4x as long as a SISO step while doing 11x the FLOPs: FLOP throughput rises about 3.3x. It does not make the same sequence finish sooner: on memory-bound hardware MIMO takes about 3.4x longer per step. The gain is in work done per byte moved, which pays off if the extra rank does useful work that SISO would need more steps or more model capacity to do — the argument the script cites from Mamba-3 (arXiv:2603.15569), not something it measures.

**Why:** The roofline model separates time from work. When an operation is memory-bound, adding arithmetic that reuses data already loaded costs almost nothing, so higher arithmetic intensity raises achieved FLOPs per second. The script's Phase 3 prints FLOPs per second for both updates in pure Python, where interpreter overhead, not memory bandwidth, dominates timing, so those numbers illustrate the idea rather than measure a GPU.

**Script reference:** `03-systems/microroofline.py`, lines 416-469 (SISO vs MIMO counts and timings), lines 908-948 (Phase 3 throughput table and notes)

</details>

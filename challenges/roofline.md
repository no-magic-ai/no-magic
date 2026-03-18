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

**Why:** Vector addition is the canonical memory-bound operation. Each element requires one floating-point operation but three memory transactions (two loads, one store). No amount of hardware compute scaling helps -- the bottleneck is entirely in the memory subsystem. This is why the roofline model has a flat "roof" on the left side: below the ridge point, performance is a horizontal line determined solely by bandwidth, not compute.

**Script reference:** `03-systems/microroofline.py`, arithmetic intensity computation and roofline model definition

</details>

---

### Challenge 2: MIMO Rank and GPU Utilization

**Setup:** For a `B[n,r] @ X[r,d]` matrix multiplication in the SSM scan, arithmetic intensity scales as `AI ≈ 2r` FLOPs/byte (where `r` is the MIMO rank). SISO (rank-1) has `AI ≈ 2`. Consider an H100 GPU with a ridge point of ~300 FLOPs/byte.

**Question:** What MIMO rank is needed to reach the H100 ridge point? What fraction of peak compute does SISO (`AI ≈ 2`) utilize on this hardware?

<details>
<summary>Reveal Answer</summary>

**Answer:** To reach the ridge point: `2r = 300`, so `r = 150`. MIMO rank-150 is needed to fully saturate H100 compute.

SISO utilization: `2 / 300 = 0.67%`. The GPU is 99.3% idle during SISO scan operations.

Even MIMO rank-16 (AI ≈ 32) achieves only `32 / 300 ≈ 10.7%` utilization. Rank-64 reaches `128 / 300 ≈ 42.7%`.

**Why:** This is the hardware motivation behind Mamba-3's MIMO formulation. SISO SSMs were designed for sequential (CPU/TPU) execution where memory bandwidth is the bottleneck anyway. On modern GPUs with massive parallel compute, SISO wastes nearly all available FLOPS. Increasing the MIMO rank converts a memory-bound operation into a compute-bound one by increasing the ratio of arithmetic to memory traffic. The matmul `B @ X` reuses data across the rank dimension, amortizing the cost of loading from memory.

**Script reference:** `03-systems/microroofline.py`, MIMO rank sweep and arithmetic intensity scaling analysis

</details>

---

### Challenge 3: Why More FLOPs Can Be Faster

**Setup:** SISO performs `3 * N * D` FLOPs per step. MIMO rank-16 performs `N * D + 2 * N * D * 16 = 33 * N * D` FLOPs per step -- 11x more total work. Both process the same sequence.

**Question:** Under what condition does MIMO finish faster despite doing 11x more FLOPs? On a GPU where SISO utilizes less than 1% of peak compute, how much can MIMO's throughput increase before it becomes compute-bound?

<details>
<summary>Reveal Answer</summary>

**Answer:** MIMO finishes faster when it is still memory-bound (or just reaching compute-bound) and the additional FLOPs are "free" -- absorbed by compute units that were previously idle.

On a GPU where SISO uses 0.67% of peak compute (AI = 2, ridge = 300): the hardware can absorb up to `300 / 2 = 150x` more FLOPs before becoming compute-limited. MIMO rank-16 at 11x more FLOPs is still well within this headroom. The achieved GFLOPS jumps from 0.67% to ~7.3% of peak, but the wall-clock time is determined by memory bandwidth (which is the same for both), not by the additional compute.

**Why:** The roofline model makes this paradox clear. Below the ridge point, execution time is `Bytes / Bandwidth` -- it depends only on data movement, not on FLOPs. If MIMO's memory traffic is similar to SISO's (same state vectors loaded/stored), the wall-clock time is nearly identical despite 11x more arithmetic. The extra FLOPs execute on hardware that was idle during SISO. This is the core insight: on parallel hardware with high compute-to-bandwidth ratios, trading more FLOPs for better arithmetic intensity is not just free -- it is the correct optimization strategy.

**Script reference:** `03-systems/microroofline.py`, SISO vs MIMO comparison and roofline throughput analysis

</details>

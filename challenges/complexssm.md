# Complex SSM Challenges

Test your understanding of complex-valued state space models by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Why Real-Only Fails at Parity

**Setup:** The real-only SSM computes `h_t = exp(log_A) * h_{t-1} + B * x_t` where `A = exp(log_A)` is always positive (exponential of a real number). The parity task requires the model to output 1 if the input contains an odd number of 1-bits, and 0 if even. This is equivalent to XOR over the sequence.

**Question:** Why can't a positive decay factor represent XOR? If `A = 0.9` and `B = 1.0`, what is the hidden state after processing the input sequence `[1, 1, 1]` starting from `h_0 = 0`?

<details>
<summary>Reveal Answer</summary>

**Answer:** A positive `A` means `h_t` is always a positive multiple of `h_{t-1}` plus the input term. The state can grow or shrink but can never negate.

After `[1, 1, 1]` with `A = 0.9`, `B = 1.0`:

- `h_1 = 0.9 * 0 + 1.0 * 1 = 1.0`
- `h_2 = 0.9 * 1.0 + 1.0 * 1 = 1.9`
- `h_3 = 0.9 * 1.9 + 1.0 * 1 = 2.71`

The state monotonically accumulates. It never flips sign. XOR requires a sign flip (180-degree rotation) on each 1-bit: the correct parity alternates odd/even/odd. With a positive real eigenvalue, each state dimension responds to a 1-bit with the same sign every time -- it can count, but not alternate. How close the trained real-only model's test accuracy lands to chance is what Phase 5 of the script reports for its run.

**Why:** The fundamental issue is that positive reals form a multiplicative group that cannot represent negation. XOR is isomorphic to addition modulo 2, which requires a group element of order 2 (something that squares to the identity but isn't the identity itself). The only real number satisfying `a^2 = 1, a != 1` is `a = -1`, but `exp(log_A)` is always positive.

**Script reference:** `03-systems/microcomplexssm.py`, lines 400-416 (`init_real_only_params`), lines 465-493 (`forward_real_only`, `a_n = exp(log_A_n)`), lines 157-176 (running-parity data)

</details>

---

### Challenge 2: Complex Rotation as a 2x2 Matrix

**Setup:** The complex SSM represents its state as `(h_re, h_im)` and multiplies by `r * e^{i*theta}` using the decomposition into real and imaginary parts: `new_re = r * (cos(theta) * h_re - sin(theta) * h_im)` and `new_im = r * (sin(theta) * h_re + cos(theta) * h_im)`. This is equivalent to a 2x2 rotation-scaling matrix.

**Question:** What is the 2x2 matrix `R(theta)` when `theta = pi` (180 degrees) and `r = 1.0`? What does it do to the state vector `[1, 0]`? What happens after applying `R(pi)` twice?

<details>
<summary>Reveal Answer</summary>

**Answer:**
`R(pi)` with `r = 1.0`:

```
[cos(pi)  -sin(pi)]   [-1   0]
[sin(pi)   cos(pi)] = [ 0  -1] = -I
```

Applied to `[1, 0]`: `[-1, 0]`. The state is negated -- a perfect sign flip.

Applied twice: `R(pi) * R(pi) = (-I)(-I) = I`. The state returns to `[1, 0]`.

**Why:** This is exactly XOR in geometric form. Each 1-bit triggers a pi-rotation that negates the state. Two consecutive 1-bits: negate then negate again, returning to the original -- which matches `1 XOR 1 = 0`. Three 1-bits: three negations yield a net negation -- matching `1 XOR 1 XOR 1 = 1`. The complex plane provides the algebraic structure (a rotation group containing elements of order 2) that the positive reals lack. This is why a rotating state can represent running parity exactly (rotate by π on every 1-bit) while a positive real decay cannot. The script initializes every rotation angle near π (`theta ~ N(π, 0.3)`, lines 432 and 453), so the rotation variants start close to that solution.

**Script reference:** `03-systems/microcomplexssm.py`, lines 236-276 (`complex_ssm_forward` with Python complex numbers), lines 278-314 (`rope_ssm_forward` with 2x2 rotations), lines 316-385 (equivalence check)

</details>

---

### Challenge 3: Data-Dependent vs Position-Dependent Rotation

**Setup:** A RoPE-style model uses rotation angle `= position * frequency` (position-dependent, always rotates). The complex SSM in this script uses `effective_theta = theta * x_t` (data-dependent, rotation magnitude scales with input).

**Question:** If `x_t = 0` for all timesteps, what happens to the complex SSM state? How does this compare to RoPE when `position = 0`? What is the key behavioral difference between position-dependent and data-dependent rotation?

<details>
<summary>Reveal Answer</summary>

**Answer:** When `x_t = 0` for all timesteps, `effective_theta = theta * 0 = 0`. Then `cos(0) = 1` and `sin(0) = 0`, so the rotation matrix becomes the identity. The recurrence reduces to `h_t = r * h_{t-1}` -- pure exponential decay with no rotation. The complex SSM falls back to real-only behavior.

In RoPE at `position = 0`, the rotation is also identity (`cos(0) = 1`, `sin(0) = 0`). But at every subsequent position, RoPE rotates regardless of the input content.

**Why:** The key difference is selectivity. RoPE rotates at every position unconditionally -- it encodes where a token is, not what it is. The data-dependent rotation fires only when the input triggers it -- it encodes what the input is, selectively. For the parity task, this means the state rotates by pi only when a 1-bit arrives and holds steady (just decaying) when a 0-bit arrives. This selective gating is what makes the complex SSM input-dependent in the same spirit as Mamba's selection mechanism, while RoPE provides a fixed positional encoding that cannot adapt to input content.

**Script reference:** `03-systems/microcomplexssm.py`, lines 494-540 (`forward_complex`, `eff_theta = theta * x_t`), lines 541-583 (`forward_rope`, the same update written as a rotation), lines 740-750 (comparison notes)

</details>

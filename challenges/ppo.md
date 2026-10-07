# MicroPPO Challenges

Test your understanding of Proximal Policy Optimization by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: Clipping With Positive vs Negative Advantage

**Setup:** The clipped surrogate objective at lines 884-894: `surr1 = ratio * adv`, `surr2 = ratio.clip(1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv`. `PPO_CLIP_EPS = 0.2`. The final objective takes `min(surr1, surr2)` (lines 891-894). The comment on lines 887-894 explains the intuition.

**Question:** Suppose `ratio = 1.5` (the new policy assigns 50% more probability to this action than the old policy did). With `adv = +2.0` (good action), what is `surr1`, `surr2`, and the PPO objective? Now with `adv = -2.0` (bad action), what are the three values?

<details>
<summary>Reveal Answer</summary>

**Answer:**

With `adv = +2.0`: `surr1 = 1.5 * 2.0 = 3.0`. `surr2 = clip(1.5, 0.8, 1.2) * 2.0 = 1.2 * 2.0 = 2.4`. PPO objective = `min(3.0, 2.4) = 2.4` (clipped — prevents over-reinforcing a good action).

With `adv = -2.0`: `surr1 = 1.5 * (-2.0) = -3.0`. `surr2 = clip(1.5, 0.8, 1.2) * (-2.0) = 1.2 * (-2.0) = -2.4`. PPO objective = `min(-3.0, -2.4) = -3.0` (NOT clipped — allows full penalty for bad actions when policy changed a lot).

**Why:** The asymmetry is the key PPO insight. For positive advantage, taking `min(surr1, surr2)` caps the reward at the clipped value — the policy cannot gain more credit for moving a ratio above 1+eps. For negative advantage, the min of a larger negative and smaller negative is the larger negative (surr1), so clipping does NOT limit how much the policy is penalized. Intuitively: PPO prevents aggressively reinforcing good actions (which could cause policy collapse) but allows aggressively penalizing bad ones. The gradient through `ratio.clip(...)` (Value.clip at line 146) is zero when the ratio is outside `[1-eps, 1+eps]`, cutting off policy gradients in the direction of over-reinforcement.

This challenge is a what-if: in `microppo.py` itself the ratio is always 1 at the moment of the update. Each step computes `old_logp` from the current weights (line 843) and then makes a single gradient update on that batch, so `ratio = exp(0) = 1`, inside `[0.8, 1.2]`, and the clip never binds. Clipping starts to matter only when the same batch is reused for several updates, as in the PPO paper.

**Script reference:** `02-alignment/microppo.py`, lines 60 (PPO_CLIP_EPS), lines 146-154 (Value.clip implementation), lines 883-894 (clipped surrogate objective), lines 887-894 (asymmetry comment)

</details>

---

### Challenge 2: Why Advantage, Not Raw Reward

**Setup:** The advantage is computed at lines 852-858: `advantage = batch_rewards[i] - val` where `val = value_forward(batch_features[i], value_params)`. The comment explains this "centers the reward signal" with lower variance.

**Question:** Suppose all generated sequences receive a reward of exactly `+1.0` (the reward model scores everything the same). With no advantage baseline, what gradient does REINFORCE produce? With the value function baseline, what gradient is produced?

<details>
<summary>Reveal Answer</summary>

**Answer:** Without a baseline, every sampled action gets the same positive weight (+1), so the update raises the probability of whatever happened to be sampled — the policy learns nothing about which actions were better. With a value function that predicts +1.0 for every sequence, `advantage = 1.0 - 1.0 = 0` for every sequence and the policy-gradient term vanishes; in this script the only remaining push is the KL penalty, which is also zero while the policy equals the reference.

**Why:** REINFORCE updates policy parameters by `gradient = E[∇log π(a|s) * R]`. If all rewards equal `R`, the expected gradient is `R * E[∇log π(a|s)] = 0`, because the score function has zero mean; but each sampled batch still pushes up the actions it happened to draw, so the update is pure sampling noise scaled by `R`. Subtracting the value baseline V(s) makes the effective reward `R - V(s)`. When all R = V(s), the advantages are zero and the gradient is zero — a stable fixed point. Only when some sequences actually receive higher reward than the baseline's prediction do those sequences' actions get reinforced. This is why advantage estimation is essential for stable RLHF training.

**Script reference:** `02-alignment/microppo.py`, lines 460-466 (value function comment), lines 851-858 (advantage computation), lines 862-870 (PPO objective intuition comment)

</details>

---

### Challenge 3: The KL Penalty and Reference Model

**Setup:** The KL penalty at lines 905-910: `kl_per_sample = current_logp.data - batch_ref_logps[i]`, followed by `kl_penalty = KL_COEFF * log_diff * log_diff * 0.5`. `KL_COEFF = 0.5`. The reference policy (line 653) is frozen at the end of pretraining.

**Question:** After many PPO steps, the policy diverges heavily from the reference. The KL term is a squared log-ratio penalty. What does this penalty do to the gradient when `current_logp - ref_logp = 5.0` (the new policy is `e^5 ≈ 148x` more likely to generate the sequence)?

<details>
<summary>Reveal Answer</summary>

**Answer:** The KL penalty contribution to the loss is `0.5 * 0.5 * 5.0^2 = 6.25`. The gradient of `0.5 * (log_pi - log_pi_ref)^2` with respect to `log_pi` is `(log_pi - log_pi_ref) = 5.0`, scaled by `KL_COEFF = 0.5`, giving a gradient of `2.5`. This is a strong pull back toward the reference policy.

**Why:** Without the KL penalty, the policy would "reward hack" — find sequences that score well according to the reward model but are linguistically degraded (e.g., repeating the same character). The squared form `(log_pi - log_pi_ref)^2 / 2` penalizes divergence quadratically: small deviations get small penalties, large deviations get very large penalties. The comment on lines 896-904 ties it to Schulman's 2020 note on approximating KL divergence, where `0.5 * (log ratio)^2` is a low-variance but biased estimate of the KL divergence computed from samples. With KL_COEFF=0.5 (much higher than the typical 0.01-0.1), the tiny model requires strong regularization to avoid mode collapse. The comment at lines 61-62 explicitly notes this was chosen to "prevent mode collapse" for the synthetic reward.

**Script reference:** `02-alignment/microppo.py`, lines 60-62 (KL_COEFF and comment), lines 896-910 (KL penalty computation), lines 648-656 (reference policy storage), lines 657-681 (reference log-prob computation)

</details>

---

### Challenge 4: Fresh Adam State for PPO

**Setup:** At line 806, `m_ppo = [0.0] * len(policy_param_list)` and `v_ppo = [0.0] * len(policy_param_list)` create fresh Adam optimizer state for PPO, explicitly separate from the pretraining Adam state (`m_pre`, `v_pre` from lines 614-615).

**Question:** Why is a fresh Adam state used for PPO rather than continuing with the pretraining Adam state? What would happen if you continued with the pretraining momentum?

<details>
<summary>Reveal Answer</summary>

**Answer:** Fresh Adam state prevents the pretraining momentum from overshooting during PPO. The pretraining optimizer built up momentum toward the language modeling objective; continuing that momentum would carry the policy in a direction that's no longer appropriate when the objective has switched to reward maximization.

**Why:** Adam's first moment `m` is an exponential moving average of past gradients. After 500 pretraining steps, `m` encodes the direction of the language modeling loss gradient. When the PPO phase starts, the loss is now `-(ratio * advantage) + KL_penalty` — a completely different objective with different gradient directions. Reusing the pretraining `m` would cause Adam to take an initial step in the wrong direction, potentially destroying the pretrained capability before the PPO signal can steer it correctly. The comment on lines 803-805 states this explicitly: "do not carry over pretraining momentum, since the objective has changed." Additionally, `PPO_LR = 0.0005` is 20x smaller than `PRETRAIN_LR = 0.01`. With fresh state, Adam's bias correction makes the first update about `PPO_LR` per parameter in size whatever the gradient scale (`m_hat / sqrt(v_hat) = ±1` on step 1), so it is the smaller learning rate, not the fresh state, that keeps the steps small.

**Script reference:** `02-alignment/microppo.py`, lines 803-807 (fresh Adam state creation), lines 803-805 (comment), lines 47 (PRETRAIN_LR), lines 67 (PPO_LR), lines 636-641 (pretraining Adam state usage)

</details>


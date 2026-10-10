# Challenges

"Predict the behavior" exercises that test your understanding of the algorithms implemented in this repository.

Each challenge presents a code snippet or scenario drawn directly from one of the scripts and asks you to predict what happens. Work through the answer yourself before expanding the collapsible solution. The goal is to build intuition for how these algorithms behave at the edge cases -- not just the happy path.

## Available Challenges

| File                           | Script                             | Topics                                                                     |
| ------------------------------ | ---------------------------------- | -------------------------------------------------------------------------- |
| [attention.md](attention.md)   | `03-systems/microattention.py`     | Scaling factor, identical keys, causal masking, sliding window             |
| [complexssm.md](complexssm.md) | `03-systems/microcomplexssm.py`    | Real-only parity failure, complex rotation matrix, data-dependent rotation |
| [discretize.md](discretize.md) | `03-systems/microdiscretize.py`    | Euler stability, trapezoidal alpha, ZOH unconditional stability            |
| [distill.md](distill.md)       | `02-alignment/microdistill.py`     | Temperature softening, T² gradient scaling, mixed-objective optimum, batch-mean 1/B under SGD |
| [dpo.md](dpo.md)               | `02-alignment/microdpo.py`         | Beta parameter, identical completions, reference divergence                |
| [embedding.md](embedding.md)   | `01-foundations/microembedding.py` | Temperature, false negatives, augmentation, representation collapse        |
| [gan.md](gan.md)               | `01-foundations/microgan.py`       | Gradient saturation, mode collapse, training balance                       |
| [gpt.md](gpt.md)               | `01-foundations/microgpt.py`       | Context window, head dimensions, learning rate, KV cache                   |
| [kv.md](kv.md)                 | `03-systems/microkv.py`            | Multiply count scaling, cache memory formula, paged attention              |
| [lora.md](lora.md)             | `02-alignment/microlora.py`        | Zero-B initialization, frozen gradient zeroing, rank and parameter count   |
| [mcts.md](mcts.md)             | `04-agents/micromcts.py`           | UCB1 value negation, visit count selection, exploration constant           |
| [moe.md](moe.md)               | `02-alignment/micromoe.py`         | Expert collapse, top-K renormalization, gradient bridge, aux loss product  |
| [optimizer.md](optimizer.md)   | `01-foundations/microoptimizer.py` | Adam degeneracy, bias correction, constant gradients                       |
| [ppo.md](ppo.md)               | `02-alignment/microppo.py`         | Clipping asymmetry, advantage baseline, KL penalty, fresh Adam state       |
| [quant.md](quant.md)           | `03-systems/microquant.py`         | Absmax outlier sensitivity, INT4 range, per-channel vs per-tensor          |
| [react.md](react.md)           | `04-agents/microreact.py`          | Action masking, reward shaping, EMA baseline, TAO loop budget              |
| [rnn.md](rnn.md)               | `01-foundations/micrornn.py`       | Vanishing gradients, GRU update gate, gradient norm ratio                  |
| [roofline.md](roofline.md)     | `03-systems/microroofline.py`      | Arithmetic intensity, MIMO rank utilization, FLOPs vs wall-clock time      |
| [sft.md](sft.md)               | `02-alignment/microsft.py`         | Response mask and causal shift, prompt-conditioning gradient, untrained prompt embeddings, batch-mean scaling under Adam |
| [ssm.md](ssm.md)               | `03-systems/microssm.py`           | Fixed state vs KV cache, delta bias, selective B/C, Euler discretization   |
| [tokenizer.md](tokenizer.md)   | `01-foundations/microtokenizer.py` | Merge order, priority replay, vocab size, corpus collapse                  |
| [turboquant.md](turboquant.md) | `03-systems/microturboquant.py`    | Rotation-hurts-on-sparse, Haar uniformity, QJL sign-bit bias, orthogonality threshold |
| [vae.md](vae.md)               | `01-foundations/microvae.py`       | Reparameterization trick, KL collapse, log_var clamping, latent structure  |

## How to Use

1. Read the referenced script first (or at least the relevant section).
2. Read the challenge setup and question.
3. Work out your prediction on paper or in your head.
4. Expand the answer to check your reasoning.

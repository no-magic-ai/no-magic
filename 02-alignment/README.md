# Alignment & Training Techniques

Methods for steering, fine-tuning, and aligning models after pretraining. These are the techniques that turn a base model into something useful.

## Scripts

Time and Status are historical values recorded when each script was added (Apple M-series, Python 3.12, wall-clock). They were not re-measured for this inventory and do not certify current runtime, correctness or media; rows marked _unmeasured_ have no recorded timing. Each script's teaching kind is recorded in [`docs/catalog.json`](../docs/catalog.json). The table lists all 11 programs in this tier: 10 `micro*` programs and the unprefixed comparison `adam_vs_sgd.py`.

| Script              | Algorithm                                                             | Time   | Status | Video                                             |
| ------------------- | --------------------------------------------------------------------- | ------ | ------ | ------------------------------------------------- |
| `adam_vs_sgd.py`    | Adam vs. SGD with momentum on the same character bigram model (trained comparison) | _unmeasured_ | not recorded | no preview |
| `microbatchnorm.py` | Batch Normalization — internal covariate shift and running statistics | 0m 34s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microbatchnorm.gif) |
| `microdpo.py`       | Direct Preference Optimization                                        | 2m 42s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdpo.gif)       |
| `microdropout.py`   | Dropout, weight decay, and early stopping as regularization           | 3m 21s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdropout.gif)   |
| `microgrpo.py`      | Group Relative Policy Optimization (DeepSeek's RLHF simplification)   | 0m 23s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microgrpo.gif)      |
| `microlora.py`      | Low-Rank Adaptation (LoRA) fine-tuning                                | 2m 32s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microlora.gif)      |
| `micromoe.py`       | Mixture of Experts with sparse routing (hybrid autograd)              | 0m 06s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/micromoe.gif)       |
| `microppo.py`       | Proximal Policy Optimization for RLHF (hybrid autograd)               | 0m 34s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microppo.gif)       |
| `microqlora.py`     | QLoRA — fine-tuning 4-bit quantized models with LoRA adapters         | 2m 27s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microqlora.gif)     |
| `microreinforce.py` | REINFORCE — vanilla policy gradient with baseline                     | 5m 39s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microreinforce.gif) |
| `microsft.py`       | Supervised fine-tuning: pretrain a tiny decoder, then train the same weights on response-masked demonstrations | 1m 08s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microsft.gif) |

The `microsft.py` row was measured during M6 on Apple M1 Pro with CPython 3.12.8: both default runs took about 68 seconds and printed byte-identical output. Its Pass means the program's frozen training-side checks passed; it does not certify generalization, and the held-out prompts are reported only.

### Hybrid Autograd Scripts

`microppo.py` and `micromoe.py` use a **hybrid autograd approach** to meet runtime constraints:

- **microppo:** Policy model uses scalar autograd (`Value` class). Reward model and value function use plain float arrays with manual gradients — they're trained separately before the PPO loop.
- **micromoe:** Router uses scalar autograd. Expert MLPs use plain float arrays — the routing decision is the novel mechanism, not the expert forward pass.

See `docs/autograd-interface.md` for the canonical interface and `docs/implementation.md` for per-script details.

## Future Candidates

| Algorithm                    | What It Would Teach                       | Notes                                   |
| ---------------------------- | ----------------------------------------- | --------------------------------------- |
| **Learning Rate Scheduling** | Warmup, cosine decay, step decay          | How schedule choice affects convergence |
| **Knowledge Distillation**   | Training small models to mimic large ones | Compression via soft targets            |

## Learning Path

These scripts build on the foundations tier. Recommended order:

```
microbatchnorm.py   → How normalizing activations stabilizes training
microdropout.py     → How regularization prevents overfitting
microlora.py        → How fine-tuning works efficiently (1% of parameters)
microqlora.py       → How quantization combines with LoRA for memory efficiency
microsft.py         → How supervised fine-tuning turns a base model into a command follower
microreinforce.py   → How policy gradients turn rewards into learning signals
microdpo.py         → How preference alignment works (without reward model)
microppo.py         → How RLHF works (the full reward → policy loop)
microgrpo.py        → How DeepSeek simplified RLHF with group-relative rewards
micromoe.py         → How sparse routing scales model capacity
```

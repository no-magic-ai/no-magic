# no-magic — Implementation Plan

This document details every script in the `no-magic` repository: what each one teaches, what it implements, architectural decisions, dataset strategy, and expected complexity. Use this as the engineering spec for building out the collection.

### Relationship to Karpathy's Work

This project is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), and `microgpt.py`. We reference and attribute his work but do not replicate it. Specifically: `microgpt.py` is included with full attribution; `micrornn.py` covers the RNN → GRU progression that makemore explores across notebooks, condensed into a single comparative file; and the autograd engine (micrograd) is already embedded within `microgpt.py`. For deeper dives into those specific topics, readers are directed to Karpathy's original repositories.

---

## Repository Structure

```text
no-magic/
├── README.md
├── CONTRIBUTING.md
├── docs/
│   ├── implementation.md       # This file — engineering spec
│   └── autograd-interface.md   # Canonical Value class interface
├── 01-foundations/
│   ├── README.md               # Algorithm list + roadmap
│   ├── microgpt.py
│   ├── micrornn.py
│   ├── microtokenizer.py
│   ├── microembedding.py
│   ├── microrag.py
│   ├── microdiffusion.py
│   └── microvae.py
├── 02-alignment/
│   ├── README.md               # Algorithm list + roadmap
│   ├── microlora.py
│   ├── microdpo.py
│   ├── microppo.py
│   └── micromoe.py
└── 03-systems/
    ├── README.md               # Algorithm list + roadmap
    ├── microattention.py
    ├── microkv.py
    ├── microquant.py
    ├── microflash.py
    └── microbeam.py
```

## Design Constraints (Enforced Across All Scripts)

| Constraint          | Rule                                                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| File count          | Exactly one `.py` file per algorithm                                                                                                                                                     |
| Dependencies        | Python standard library only (`os`, `math`, `random`, `json`, `struct`, `urllib`, `collections`, `itertools`, `functools`, `string`, `hashlib`, `time`)                                  |
| Execution           | `python script.py` with no arguments runs the whole program: the full train + inference loop for `train_infer` scripts, training and comparing arms for `comparison` scripts, and the recorded `forward_pass` or `algorithm_demo` demonstration otherwise (see `docs/catalog.json`) |
| Runtime             | Under **7 minutes on M-series Mac** or under **10 minutes on 2019-era Intel i5**. Target 7 min to leave headroom for slower hardware.                                                    |
| Dataset             | Generated or embedded in the script, or auto-downloaded on first run via `urllib` and cached locally (`names_download`: makemore `names.txt` in the working directory, needs network once); under 5MB |
| Output              | Prints to stdout what the teaching kind demonstrates: training progress and inference results for `train_infer`, per-arm training and comparison results for `comparison`, and mechanism or algorithm results for `forward_pass` and `algorithm_demo` |
| Seed                | `random.seed(42)` for reproducibility                                                                                                                                                    |
| Comments            | **Mandatory.** Every script must follow the commenting standard in `CONTRIBUTING.md`. Scripts will not be merged without adequate commentary.                                            |
| Autograd            | Scripts using scalar autograd must implement the canonical `Value` class interface defined in `docs/autograd-interface.md`                                                               |
| Numerical stability | All scripts must use stable softmax (`exp(x - max(x))`), clipped log-probabilities (`max(p, 1e-10)`), and Adam epsilon (`1e-8`). See `docs/autograd-interface.md` for required patterns. |

### Minimum Hardware Requirements

- Python 3.10+
- 8 GB RAM
- Any modern CPU (2019-era or newer)

Scripts are tested on M-series Mac (primary) and Intel i5 (secondary). If a script runs in 7 minutes on M-series, it should complete within 10 minutes on 2019-era Intel.

### Commenting Standard

See `CONTRIBUTING.md` for the full commenting standard (7 required comment types, density targets, examples). This is the single authoritative reference for comment quality.

**Summary:** File thesis, section headers, "why" comments, math-to-code mappings, intuition comments, signpost comments, no obvious comments. Target 30-40% comment density. The test: _could a motivated engineer read this file top-to-bottom in one sitting and understand the algorithm?_

### Autograd Callout Pattern

Scripts that reimplement the scalar autograd `Value` class (microgpt, micrornn, microlora, microdpo, microppo, micromoe) must include a callout block immediately after the Value class definition:

```python
# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following additions/modifications for [algorithm name]:
# - sigmoid(): Required for GRU gating (not in microgpt's base set)
# - clip(): Required for PPO ratio clipping
# See docs/autograd-interface.md for the full canonical interface.
```

This prevents readers from skipping the autograd section and missing per-script differences.

---

## 01 — Foundations

### `microgpt.py` — Autoregressive Language Model

> _"The most atomic way to train and inference a GPT in pure, dependency-free Python."_

**What it teaches:**

- Scalar autograd via reverse-mode automatic differentiation
- Token and positional embeddings
- Multi-head self-attention with causal masking (via incremental KV construction)
- RMSNorm, residual connections, MLP blocks
- Cross-entropy loss, Adam optimizer with bias correction
- Temperature-scaled autoregressive sampling

**Architecture:** GPT-2 variant — RMSNorm instead of LayerNorm, no biases, ReLU instead of GELU

**Dataset:** `names.txt` from Karpathy's makemore (~32K names, 18KB, auto-downloaded via urllib)

**Hyperparameters:** `n_embd=16, n_head=4, n_layer=1, block_size=16, lr=0.01, ~4,200 params`

**Success criteria:**

- Final loss: < 2.7 (character-level cross-entropy, single-document per step — stochastic, not averaged)
- Generated names: ≥50% pronounceable English-like sequences
- Runtime: < 7 minutes on M-series Mac

---

### `micrornn.py` — Recurrent Sequence Modeling

> _"Before attention conquered everything — how sequences were modeled with recurrence, and why gating was the breakthrough."_

**What it teaches:**

- The vanilla RNN: hidden state as a lossy compression of sequence history
- Backpropagation through time (BPTT): unrolling the recurrence for gradient computation
- The vanishing gradient problem: why vanilla RNNs fail on long sequences (demonstrated numerically, not just stated)
- GRU gating: reset gate (what to forget) and update gate (what to keep)
- Why gating solves the gradient problem: the update gate creates gradient highways
- The historical arc from RNNs → GRUs/LSTMs → Transformers, and what each transition gained

**Algorithm outline:**

```texttext
1. Implement scalar autograd (reuse the Value class pattern)
2. Implement a vanilla RNN:
   a. h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
   b. y_t = W_hy @ h_t + b_y
   c. Train on name generation, print loss curve
   d. Print gradient norms at each timestep — show exponential decay
3. Implement a GRU:
   a. z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})      # update gate
   b. r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})      # reset gate
   c. h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
   d. h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate
   e. Train on same data, print loss curve
   f. Print gradient norms — show stable gradients through gating
4. Compare: final loss, gradient health, generated name quality
5. Inference: generate names from both models side by side
```

**Dataset:** `names.txt` — same task as `microgpt.py` for direct comparison across architectures.

**Key implementation details:**

- Both models train on identical data with identical hyperparameters — the only variable is the architecture
- Gradient norm tracking is the core pedagogical tool: print `||dL/dh_t||` at each timestep for a sample sequence to make vanishing gradients visible, not just theoretical
- Sigmoid implemented in the autograd: `sigmoid(x) = 1 / (1 + exp(-x))`
- The GRU is chosen over LSTM because it has fewer gates (2 vs. 3) and is easier to follow in scalar autograd, while teaching the same gating principle
- Hidden state dimension kept small (`n_hidden=32`) for tractability
- Include a direct comparison table at the end: architecture, final loss, gradient norm ratio (first vs. last timestep), sample outputs

**Historical context note:** This script exists to provide the "before" picture. Karpathy's [makemore](https://github.com/karpathy/makemore) series walks through this progression across multiple notebooks. Here, both models live in one file so the comparison is immediate and inescapable.

**Hyperparameters:** `n_hidden=32, seq_len=16, lr=0.1 (SGD), steps=3000 per model, ~800 params (RNN), ~800 params (GRU)`

**Success criteria:**

- Vanilla RNN gradient norm ratio (last/first timestep): < 0.01 (demonstrates vanishing)
- GRU gradient norm ratio: 0.1–10.0 (demonstrates stability)
- GRU final loss < vanilla RNN final loss
- Generated names from GRU are higher quality than vanilla RNN
- Runtime: < 9 minutes on M-series Mac

**Expected complexity:** ~350-400 lines. Two model implementations + gradient analysis + comparative inference.

---

### `microtokenizer.py` — Byte-Pair Encoding

> _"How text becomes numbers — the compression algorithm hiding inside every LLM."_

**What it teaches:**

- Why tokenization matters (vocabulary efficiency, subword representation)
- The BPE merge algorithm: iterative pair frequency counting and merging
- Encoding: greedy left-to-right application of learned merges
- Decoding: simple lookup from token IDs back to byte sequences
- The relationship between vocabulary size and sequence length

**Algorithm outline:**

```texttext
1. Start with byte-level vocabulary (256 base tokens)
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat for N merges (N controls vocab size)
5. Encoding: apply merges greedily to new text
6. Decoding: map token IDs back to byte strings
```

**Dataset:** `names.txt` from Karpathy's makemore (~32K names, auto-downloaded via urllib)

**Key implementation details:**

- Maintain a merge priority table (ordered list of merges)
- Encoding applies merges in priority order, not frequency order on new text
- Handle UTF-8 properly: base vocabulary is bytes (0-255), not characters
- Show compression ratio before/after tokenization

**Success criteria:**

- Round-trip correctness: `decode(encode(text)) == text` for all test inputs
- Compression ratio: ≥1.5x reduction in token count vs. byte-level encoding
- Runtime: < 2 minutes on M-series Mac

**Expected complexity:** ~150-200 lines. Straightforward algorithm, main challenge is clean encode/decode symmetry.

---

### `microembedding.py` — Contrastive Embedding Learning

> _"How meaning becomes geometry — training vectors where distance equals similarity."_

**What it teaches:**

- Why learned embeddings outperform bag-of-words and TF-IDF for semantic tasks
- Contrastive learning with InfoNCE / NT-Xent loss
- Positive and negative pair construction
- Temperature scaling in contrastive objectives
- Cosine similarity as the learned distance metric
- How embedding spaces organize semantically

**Algorithm outline:**

```texttext
1. Define a simple encoder (bag-of-character-ngrams → linear projection)
2. Construct training pairs:
   - Positive: augmented versions of the same document (e.g., character dropout)
   - Negative: other documents in the batch
3. Compute embeddings for all pairs
4. Apply InfoNCE loss: maximize similarity of positives, minimize similarity of negatives
5. Train with SGD/Adam
6. Inference: embed new strings and find nearest neighbors
```

**Dataset:** `names.txt` — embed names into a space where similar-sounding names cluster together

**Key implementation details:**

- Character n-gram features as input representation (no learned tokenizer dependency)
- Simple linear encoder (matrix multiply + normalize), no deep network needed
- Augmentation via random character deletion/swap to create positive pairs
- Cosine similarity computed manually: `dot(a,b) / (||a|| * ||b||)`
- Demonstrate nearest-neighbor retrieval at inference time

**Success criteria:**

- Nearest neighbors have low edit distance (e.g., "Anna" → "Anne", not "Anna" → "Zachary")
- Cosine similarity between positive pairs > 0.8 after training
- Cosine similarity between random pairs < 0.3 after training
- Runtime: < 5 minutes on M-series Mac

**Expected complexity:** ~200-250 lines. The loss function is the core; the encoder can stay simple.

---

### `microrag.py` — Retrieval-Augmented Generation

> _"How retrieval augments generation — the simplest system that actually works."_

**What it teaches:**

- The RAG architecture: retrieve-then-generate
- TF-IDF or BM25 scoring for document retrieval
- How retrieved context is injected into a generative model's input
- The fundamental tradeoff between parametric knowledge (model weights) and non-parametric knowledge (retrieved documents)
- Why RAG reduces hallucination

**Algorithm outline:**

```texttext
1. Build a document index:
   - Tokenize documents into terms
   - Compute TF-IDF (or BM25) scores
   - Store as an inverted index
2. At query time:
   - Score all documents against the query
   - Retrieve top-k documents
3. Concatenate retrieved context with the query
4. Feed the augmented input into a small trained language model
5. Generate output conditioned on both the query and retrieved context
```

**Dataset:** 100 synthetic factual paragraphs (cities, countries, basic facts). Generated programmatically within the script — no download needed. Simple enough to verify retrieval quality by inspection.

**Key implementation details:**

- BM25 implemented from scratch: term frequency saturation, document length normalization, IDF weighting
- The language model is a **character-level MLP** with concatenated input (`embed(query) + embed(retrieved_context)`). A bigram model cannot condition on retrieved context and would fail to demonstrate RAG's core mechanism.
- Demonstrate: same query with and without retrieval, showing improved accuracy
- The retrieval and generation components must both be implemented in the same file

**Design decision:** The MLP accepts concatenated query + retrieved context as input, enabling the model to actually use retrieved information. This is the minimum architecture that meaningfully demonstrates RAG. The focus is on the retrieval mechanism and context injection, not on the generative model's sophistication.

**Success criteria:**

- Retrieval: BM25 returns relevant documents for ≥80% of test queries
- Generation with retrieval produces measurably better outputs than without retrieval
- Runtime: < 6 minutes on M-series Mac

**Expected complexity:** ~350-400 lines. Most complex foundational script due to having two subsystems (BM25 + MLP).

---

### `microdiffusion.py` — Denoising Diffusion

> _"How images emerge from noise — the algorithm behind Stable Diffusion, in 2D."_

**What it teaches:**

- The forward process: progressively adding Gaussian noise to data
- The reverse process: learning to predict and remove noise
- Noise schedule (linear beta schedule)
- The denoising objective: predict the noise that was added
- Sampling: iterative denoising from pure noise to data

**Algorithm outline:**

```texttext
1. Define a tiny 2D dataset (e.g., points sampled from a spiral or Swiss roll)
2. Forward process: add noise at T timesteps with a linear schedule
3. Train a small MLP to predict the noise given (noisy_data, timestep)
4. Sampling: start from random noise, iteratively denoise for T steps
5. Visualize: print the generated 2D points (or their statistics)
```

**Dataset:** Synthetic — 2D point clouds (spiral, concentric circles, Swiss roll). Generated programmatically, no download needed.

**Key implementation details:**

- 2D data keeps the model tiny (MLP with ~1000 params) while preserving all algorithmic structure
- Linear noise schedule: `beta_t` linearly interpolated from `beta_1` to `beta_T`
- Precompute `alpha_bar_t` for efficient noising at arbitrary timesteps
- The denoising network takes `[x_noisy, t_embedding]` as input
- Timestep embedding via sinusoidal encoding (implemented manually)
- Output: statistics of generated points (mean, variance, distribution shape) printed to stdout

**2D-to-image mapping:** The algorithm is identical to image diffusion (Stable Diffusion, DALL-E). 2D coordinates map to pixel values, the 1000-param MLP maps to a billion-param U-Net, and Gaussian noise on (x,y) maps to Gaussian noise on RGB. The core insight — learn to predict noise, then iteratively denoise — is the same at any dimensionality. Comments must make this mapping explicit.

**Success criteria:**

- Generated point cloud statistics (mean, variance) match training distribution within 20%
- Visual inspection: generated spiral/Swiss roll is recognizable as the target shape
- Runtime: < 5 minutes on M-series Mac

**Expected complexity:** ~250-300 lines. The 2D simplification makes this tractable without any image libraries.

---

### `microvae.py` — Variational Autoencoder

> _"How to learn a compressed, generative representation of data — the reparameterization trick demystified."_

**What it teaches:**

- Encoder-decoder architecture for unsupervised learning
- The reparameterization trick: backpropagating through sampling
- ELBO loss: reconstruction loss + KL divergence regularization
- Latent space interpolation and generation
- Why VAEs produce blurry but diverse outputs (vs. GANs)

**Algorithm outline:**

```texttext
1. Define a tiny dataset (2D points or small discrete sequences)
2. Encoder: maps input → (mean, log_variance) of latent distribution
3. Reparameterize: z = mean + exp(0.5 * log_var) * epsilon, where epsilon ~ N(0,1)
4. Decoder: maps z → reconstructed input
5. Loss = reconstruction_loss + beta * KL(q(z|x) || p(z))
6. Train with Adam
7. Inference: sample z ~ N(0,1), decode to generate new data
```

**Dataset:** Synthetic 2D distributions or the same names dataset (character-level VAE).

**Key implementation details:**

- The reparameterization trick must be explicit — this is the entire pedagogical point
- KL divergence for Gaussian has a closed-form solution: `0.5 * sum(1 + log_var - mean^2 - exp(log_var))`
- Demonstrate latent space interpolation between two data points
- Beta-VAE weighting to show the reconstruction/regularization tradeoff

**Success criteria:**

- Reconstruction loss decreases over training (ELBO improves)
- KL divergence is positive and bounded (not collapsed to 0 or exploded)
- Latent interpolation produces smooth transitions between data points
- Generated samples from z ~ N(0,1) resemble training distribution
- Runtime: < 4 minutes on M-series Mac

**Expected complexity:** ~200-250 lines. Conceptually elegant; the tricky part is making the reparameterization trick crystal clear in code.

---

## 02 — Alignment & Training Techniques

### `microlora.py` — Low-Rank Adaptation

> _"How to fine-tune a model by updating 1% of its parameters — the math behind efficient adaptation."_

**What it teaches:**

- Why full fine-tuning is expensive (all parameters, all gradients, all optimizer states)
- Low-rank decomposition: `W_new = W_frozen + A @ B` where A and B are small
- Why low rank works (weight updates during fine-tuning are empirically low-rank)
- Freezing base weights vs. training adapter weights
- Rank as a hyperparameter: capacity vs. efficiency tradeoff

**Algorithm outline:**

```texttext
1. Train a base model (reuse microgpt architecture) on dataset A
2. Freeze all base model parameters
3. For selected weight matrices, add low-rank adapters: A (d×r) and B (r×d), r << d
4. Forward pass: output = W_frozen @ x + A @ B @ x
5. Only A and B receive gradients
6. Train on dataset B (different distribution)
7. Show: adapted model performs on B without forgetting A
```

**Dataset:** Two splits of `names.txt` — e.g., names starting A-M as base, N-Z as adaptation target. Or: English names as base, a different name list as adaptation.

**Key implementation details:**

- Base model trains first (reusing the microgpt loop)
- Adapter matrices initialized: A ~ N(0, σ), B = 0 (so initial adaptation is zero)
- Explicit gradient freezing: base `Value` nodes have their `.grad` reset to 0 after backward
- Compare: trainable parameter count with vs. without LoRA
- Show generation quality on both the base and adapted distributions

**Success criteria:**

- Base model converges on dataset A (loss < 2.5)
- LoRA-adapted model improves on dataset B without catastrophic forgetting on A
- Trainable parameter count with LoRA < 10% of full model parameters
- Runtime: < 7 minutes on M-series Mac (base: 3 min to 50% convergence + LoRA: 2 min + inference: 1 min)

**Expected complexity:** ~350-400 lines. Includes both base training and LoRA adaptation phases.

---

### `microdpo.py` — Direct Preference Optimization

> _"How to align a model with human preferences without training a separate reward model."_

**What it teaches:**

- The preference learning problem: given (prompt, chosen, rejected), make the model prefer "chosen"
- Why DPO simplifies RLHF: the optimal policy has a closed-form relationship to the reward
- The DPO loss function: a contrastive objective over log-probability ratios
- The role of the reference model (KL anchor)
- Beta parameter: how strongly preferences override the base distribution

**Algorithm outline:**

```text
1. Train a base/reference model on a text corpus
2. Create preference pairs: (prompt, chosen_completion, rejected_completion)
3. Compute log-probabilities of chosen and rejected under both the policy and reference model
4. DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
5. Update only the policy model
6. Show: generation shifts toward preferred completions
```

**Dataset:** Synthetic preference pairs derived from `names.txt` — e.g., prefer names with certain phonetic properties, or prefer longer names over shorter ones. The preference signal should be simple enough to verify visually.

**Key implementation details:**

- Reference model is a frozen copy of the base model's parameters
- Log-probability computation requires full sequence scoring (sum of per-token log-probs)
- The sigmoid and log-ratio math must be numerically stable
- Beta controls alignment strength — demonstrate with different values
- Show: KL divergence between policy and reference increases with training

**Success criteria:**

- DPO loss decreases over training
- Policy model generates preferred completions more frequently than rejected ones
- KL divergence between policy and reference increases with training (controlled by beta)
- Runtime: < 7 minutes on M-series Mac (pretrain: 3 min + DPO: 3 min + inference: 1 min)

**Expected complexity:** ~350-400 lines. Requires two model copies (reference + policy) and preference pair construction.

---

### `microppo.py` — Proximal Policy Optimization for RLHF

> _"The full RLHF loop: reward model, policy gradient, KL penalty — all in one file."_

**What it teaches:**

- The RLHF pipeline: pretrain → reward model → policy optimization
- Reward model training from preference pairs
- Policy gradient with a clipped surrogate objective (PPO)
- KL penalty to prevent the policy from diverging too far from the reference
- Value function baseline for variance reduction
- Why this is harder than DPO (and when you'd still want it)

**Algorithm outline:**

```text
1. Train a base language model (pretrain phase)
2. Train a reward model on preference pairs:
   - Input: (prompt, completion) → scalar reward score
   - Trained with pairwise ranking loss
3. PPO loop:
   a. Generate completions from the current policy
   b. Score completions with the reward model
   c. Compute advantages (reward - value baseline)
   d. Update policy with clipped surrogate objective
   e. Update value function
   f. Apply KL penalty against the reference policy
4. Show: generation quality improves according to reward signal
```

**Dataset:** Same synthetic preference setup as `microdpo.py` for comparability.

**Key implementation details:**

- **Hybrid autograd approach:** The policy model uses scalar autograd (`Value` class) because PPO gradients must flow through the policy. The reward model and value function use plain float arrays with manual gradient computation — they are trained separately before the PPO loop, so autograd overhead is unnecessary. This preserves the full RLHF algorithm while meeting runtime constraints.
- Policy: scalar autograd, ~1,000 params (`n_embd=8, n_head=2, n_layer=1`)
- Reward model: plain float MLP, ~500 params, trained with pairwise ranking loss
- Value function: plain float linear, ~200 params
- PPO clipping: `min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)`
- KL penalty: explicit computation of `KL(policy || reference)` per sequence
- Advantage estimation: simple `reward - value_baseline` (no GAE to keep it tractable)
- Training: 100 PPO steps, batch_size=4, seq_len=8
- This is the most complex script in the collection — budget accordingly

```python
# IMPLEMENTATION NOTE: The reward model and value function use plain floats (not
# autograd Value objects) for runtime tractability. The policy model uses scalar
# autograd because PPO gradients must flow through the policy's generation process.
# Production RLHF (InstructGPT, ChatGPT) vectorizes all three models on GPUs;
# we split the approach to stay within pure-Python runtime constraints while
# preserving the complete PPO algorithm.
```

**Success criteria:**

- Reward model accuracy: > 70% on held-out preference pairs
- Policy generation shifts toward preferred completions over training
- PPO loss decreases; KL divergence increases (controlled by penalty coefficient)
- Runtime: < 7 minutes on M-series Mac

**Expected complexity:** ~550-600 lines. The most ambitious script; three interacting models plus the RL loop.

---

### `micromoe.py` — Mixture of Experts

> _"How to scale model capacity without scaling compute — sparse routing in action."_

**What it teaches:**

- The MoE concept: multiple expert networks, only some activated per input
- Router/gating network: how tokens get assigned to experts
- Top-k expert selection and soft combining of expert outputs
- Load balancing loss: why naive routing collapses to using one expert
- The capacity vs. compute tradeoff that makes MoE compelling at scale

**Algorithm outline:**

```text
1. Define N small expert MLPs (each identical architecture, different weights)
2. Define a router: linear layer mapping input → N expert scores
3. For each input token:
   a. Router produces scores over N experts
   b. Select top-k experts (k=2 typically)
   c. Compute weighted sum of selected expert outputs
4. Add load balancing auxiliary loss to prevent expert collapse
5. Train on language modeling objective + auxiliary loss
6. Show: expert utilization statistics and per-expert specialization
```

**Dataset:** `names.txt` or a slightly larger text corpus to give experts enough signal to specialize.

**Key implementation details:**

- **Hybrid autograd approach:** The router uses scalar autograd (`Value` class) because routing decisions are the core MoE mechanism — gradients must flow through the gating function. Expert MLPs use plain float arrays with manual gradient computation for runtime tractability.
- 4 experts, top-2 routing (preserves meaningful load balancing dynamics)
- Router is a simple linear layer with softmax (autograd `Value` objects)
- Experts are 2-layer MLPs with ~200 params each (plain floats)
- Load balancing loss: minimize variance of expert assignment frequencies
- Track and print expert utilization per training step
- Demonstrate expert specialization: which experts activate for which input patterns

```python
# IMPLEMENTATION NOTE: Experts use plain floats (not autograd Value objects) for
# runtime tractability. The router uses scalar autograd because routing decisions
# are the core MoE mechanism — gradients must flow through the gating function.
# Production MoE frameworks (Mixtral, Switch Transformer) vectorize everything;
# we split the approach to stay within pure-Python runtime constraints.
```

**Success criteria:**

- All 4 experts receive >10% of token assignments (no expert collapse)
- Load balancing loss decreases over training
- Different experts show measurable specialization on different input patterns
- Runtime: < 7 minutes on M-series Mac

**Expected complexity:** ~350-400 lines. The routing logic and auxiliary loss are the core; each expert is a simple MLP.

---

## 03 — Systems & Inference

### `microattention.py` — Attention Variants Compendium

> _"Every attention mechanism that matters, implemented side by side in one file."_

**What it teaches:**

- Vanilla scaled dot-product attention
- Multi-head attention (parallel heads, concatenate, project)
- Grouped-query attention (GQA): shared KV heads across query heads
- Multi-query attention (MQA): single KV head for all query heads
- Sliding window attention: local context window
- How each variant trades off memory, compute, and quality

**Algorithm outline:**

```text
For each attention variant:
1. Implement the forward pass
2. Count FLOPs and memory usage analytically
3. Run on the same input sequence
4. Print: output values, FLOP count, memory footprint
5. Compare all variants in a summary table
```

**Dataset:** No training. Uses random input tensors (lists of lists of `Value` or plain floats) to demonstrate the mechanics.

**Key implementation details:**

- This is primarily a **forward-pass comparison**, not a training script (exception to the train+infer rule, justified because the focus is architectural comparison)
- Each variant is a self-contained function
- Print a comparison table at the end: variant, FLOPs, memory, output similarity
- GQA and MQA are implemented as modifications of the MHA base, making the differences explicit

**Success criteria:**

- All variants produce valid attention outputs (no NaN, no overflow)
- MQA and GQA outputs are close to MHA (cosine similarity > 0.95)
- Printed comparison table shows correct FLOP/memory tradeoffs
- Runtime: < 1 minute on M-series Mac

**Expected complexity:** ~250-300 lines. Multiple small implementations rather than one large one.

---

### `microkv.py` — KV-Cache Mechanics

> _"Why LLM inference is memory-bound — and exactly how the KV cache works."_

**What it teaches:**

- Why naively running attention at each generation step is O(n²) redundant
- The KV cache: store and reuse key/value projections from previous positions
- Memory growth: how cache size scales with sequence length, layers, and heads
- Paged attention intuition: why memory fragmentation matters at scale
- Prefill vs. decode phases

**Algorithm outline:**

```text
1. Implement attention WITHOUT KV cache (recompute everything each step)
2. Implement attention WITH KV cache (incremental, append-only)
3. Run both on the same autoregressive generation task
4. Compare: computation count, memory usage at each step
5. Simulate paged allocation: fixed-size blocks, pointer table
```

**Dataset:** Uses a pretrained tiny model (train inline or hardcode small weights) for generation.

**Key implementation details:**

- Side-by-side implementations make the redundancy obvious
- Count and print multiply operations at each generation step
- Show memory growth curve: cache size as a function of sequence position
- Paged attention section is conceptual/simulated — demonstrate the allocation strategy without a full memory manager

**Success criteria:**

- With-cache and without-cache produce identical outputs
- Operation count: without-cache grows as O(n²), with-cache grows as O(n)
- Memory growth curve is printed and shows linear scaling
- Runtime: < 4 minutes on M-series Mac

**Expected complexity:** ~200-250 lines. The comparison structure is the teaching tool.

---

### `microquant.py` — Weight Quantization

> _"How to shrink a model by 4x with minimal quality loss — the math behind INT8 and INT4."_

**What it teaches:**

- Why quantization works: neural network weights are approximately normally distributed
- Absmax quantization: scale to fit the integer range
- Zero-point quantization: asymmetric ranges
- Per-channel vs. per-tensor quantization granularity
- Dequantization for inference
- Quality degradation measurement

**Algorithm outline:**

```text
1. Train a small model to convergence (reuse microgpt architecture)
2. Quantize weights to INT8:
   a. Compute scale factor: max(abs(weights)) / 127
   b. Quantize: round(weight / scale)
   c. Store as integers + scale factor
3. Quantize weights to INT4 (same process, range [-8, 7])
4. Dequantize and run inference with each version
5. Compare: model size, generation quality, per-token loss
```

**Dataset:** `names.txt` — train the base model, then quantize and compare.

**Key implementation details:**

- Represent quantized weights as Python integers, not floats — this is the point
- Show the actual memory savings: `float32 (4 bytes) → int8 (1 byte) → int4 (0.5 bytes)`
- Compute and print perplexity/loss for each quantization level
- Demonstrate per-channel vs. per-tensor: quantize each row of a weight matrix separately vs. the whole matrix
- Round-trip test: quantize → dequantize → compare to original

**Success criteria:**

- INT8 quantized model loss within 10% of float32 baseline
- INT4 quantized model loss within 25% of float32 baseline
- Per-channel quantization outperforms per-tensor quantization
- Printed table shows model size reduction: float32 → INT8 (4x) → INT4 (8x)
- Runtime: < 6 minutes on M-series Mac

**Expected complexity:** ~300-350 lines. Includes base training + quantization + comparative evaluation.

---

### `microturboquant.py` — Data-Oblivious Vector Quantization

> _"Why rotating a vector before quantizing it works — random rotation makes coordinates independent, turning one hard D-dim quantization problem into D easy 1-D ones."_

**What it teaches:**

- The data-oblivious regime: compress vectors with no access to the data distribution
- Why coordinate-wise scalar quantization fails on anisotropic data (elongated clusters waste bits)
- Random rotation as a universal preprocessor: any unit vector's rotated coordinates concentrate near a Beta distribution
- Optimal 1-D scalar quantizers for a known marginal (vs. absmax, which is loss-oblivious)
- Inner-product preservation as the right quality metric for quantized embeddings (not raw MSE)
- The 1-bit Quantized Johnson-Lindenstrauss trick: sign bits on residuals recover unbiased inner-product estimates

**Algorithm outline:**

```text
1. Generate the evaluation corpus:
   a. Real: per-name embeddings via random projection of bigram counts
      (Johnson-Lindenstrauss densification; near-isotropic, realistic magnitudes)
   b. Synthetic: anisotropic Gaussians with one dominant-axis std inflated by 6x
      before unit-normalization — the regime where absmax wastes bits and
      rotation should help
2. Baseline: plain absmax scalar quantization (no rotation)
   a. Per-vector scale = max(abs(x)) / (2^(b-1) - 1)
   b. Quantize each coordinate independently
3. TurboQuant: random-rotation + scalar quantization
   a. Sample a D-by-D rotation R via Gram-Schmidt on Gaussian columns (one rotation,
      shared across all vectors — data-oblivious)
   b. Rotate: y = R @ x
   c. Absmax-quantize y
   d. Dequantize and rotate back: x_hat = R^T @ y_hat
4. 1-bit QJL residual stage:
   a. Sample a second random Gaussian projection S of shape (K, D)
   b. For each vector, store sign bits of S @ x (K bits per vector)
   c. Show <a,b> ~ (pi/2) * mean(sign(S @ a) * sign(S @ b)) recovers IP unbiasedly
5. Evaluate on held-out vector pairs at bit-widths {1, 2, 4, 8}:
   a. Inner-product MSE (primary metric — the paper's headline)
   b. Mean signed error for the QJL estimator
6. Print a rate-distortion table: baseline vs. TurboQuant at each bit-width
```

**Dataset:** `names.txt` (reused) for real embeddings. Synthetic anisotropic Gaussians generated inline — no download needed.

**Key implementation details:**

- **No NumPy.** Matrix-vector products are nested Python loops. Keep D small (32) and N moderate (~300 vectors) so total FLOPs stay under ~10M.
- **No autograd.** The entire pipeline is forward math; skip the `Value` class. Real embeddings use a fixed random projection of sparse bigram counts, not a trained skip-gram — random projection runs in milliseconds and produces vectors in the regime where TurboQuant's rate-distortion story is clean.
- Rotation construction: Gram-Schmidt on D columns of standard Gaussian samples. Verify orthogonality by printing `max(abs(R^T R - I))` after construction.
- **Anisotropy sweet spot.** ANISOTROPY = 6x is the point where absmax starts wasting bits AND the post-normalization vector is still dense. Above ~10x the normalized vector is nearly 1-sparse and absmax trivially wins; below ~3x the gap is too mild to show at D=32.
- **Sparse-vector signpost.** Document explicitly that TurboQuant loses on very sparse unit vectors (raw bigram counts, one-hot) — rotation converts sparse-good into dense-average. Production uses structured rotations + sparse-aware codebooks.
- **Beta-optimal quantizer is out of scope.** The paper derives per-coord optimal bin boundaries from the Beta marginal of rotated coordinates; implementing this adds ~60 LOC and buys ~10% MSE improvement. Signpost-comment the omission.

**Success criteria:**

- At 4 bits/coord, TurboQuant inner-product MSE is at least 1.5x lower than absmax baseline on anisotropic synthetic data
- At 8 bits/coord, TurboQuant inner-product MSE is at least 1.3x lower than absmax baseline on anisotropic synthetic data
- On near-isotropic real embeddings, TurboQuant IP-MSE stays within 1.2x of absmax at every bit-width (no regression in the data-oblivious regime)
- QJL mean signed error below 2% of unit-vector IP range on held-out pairs
- Orthogonality check: `max(abs(R^T R - I)) < 1e-10`
- Runtime: < 1 minute on M-series Mac (no training, all forward math)

**Expected complexity:** ~400-460 lines. Breakdown: ~70 lines linear-algebra primitives + rotation, ~60 lines quantizers, ~50 lines QJL, ~90 lines embedding sources, ~80 lines evaluation + rate-distortion, rest is data loading, constants, and main.

**Pedagogical connection:** Sits between `microquant` (data-aware, per-tensor) and `microembedding` (vectors as the object of interest). First entry in the catalog at the intersection of random projections and quantization — no prior coverage of Johnson-Lindenstrauss or data-oblivious compression.

**Reference:** Aamand et al., "TurboQuant: Online Vector Quantization with Optimal Bit Budget" (2025). https://arxiv.org/abs/2504.19874

---

### `microflash.py` — Flash Attention (Algorithmic Simulation)

> _"Why Flash Attention is fast — the tiling and online softmax trick, simulated in pure Python."_

**What it teaches:**

- Standard attention's memory bottleneck: materializing the full N×N attention matrix
- Tiled computation: process attention in blocks that fit in "fast memory"
- Online softmax: compute softmax incrementally without storing all scores
- The IO complexity argument: why fewer memory reads matter more than fewer FLOPs
- Memory access patterns: the real reason Flash Attention is faster

**Algorithm outline:**

```text
1. Implement standard attention (materialize full N×N matrix)
2. Implement Flash Attention:
   a. Tile Q, K, V into blocks of size B
   b. For each Q block:
      - For each K,V block:
        - Compute partial attention scores
        - Update running softmax using online algorithm
        - Accumulate weighted values
   c. Rescale final output
3. Verify: outputs match standard attention (within floating point tolerance)
4. Compare: peak "memory" usage (simulated), number of memory reads/writes
```

**Dataset:** No training. Random matrices of configurable size to demonstrate the mechanics.

**Key implementation details:**

- **This is an algorithmic simulation, not a performance optimization.** Pure Python will be slower than standard attention. The point is to show _what_ Flash Attention does, not to be fast.
- Online softmax is the key insight: maintain running `max` and `sum` across blocks
- Track and print simulated memory usage: standard (O(N²)) vs. flash (O(N))
- Configurable block size B to show how tiling granularity affects memory
- Numerical verification: assert outputs match within 1e-6

**Success criteria:**

- Flash attention output matches standard attention within 1e-6 tolerance
- Simulated peak memory: standard O(N²) vs. flash O(N) clearly shown
- Runtime: < 2 minutes on M-series Mac

**Expected complexity:** ~300-350 lines. The online softmax is the crux; tiling bookkeeping and comparison output add more lines than expected.

---

### `microbeam.py` — Decoding Strategies

> _"Beyond greedy: beam search, top-k, top-p, and speculative decoding in one file."_

**What it teaches:**

- Greedy decoding: take the argmax at each step (and why it's suboptimal)
- Temperature sampling: controlling randomness
- Top-k sampling: truncate to k most likely tokens
- Top-p (nucleus) sampling: truncate to cumulative probability threshold
- Beam search: maintain top-B candidates, score complete sequences
- Speculative decoding: use a small model to draft, large model to verify

**Algorithm outline:**

```text
1. Train two language models inline (reuses microgpt's autograd pattern):
   - Large "target" model: n_embd=16, n_layer=1 (~4,200 params)
   - Small "draft" model: n_embd=8, n_layer=1 (~1,300 params)
2. Implement each decoding strategy as a separate function
3. Generate from the same prompt with each strategy
4. Print: generated text, total log-probability, generation speed (simulated)
5. Speculative decoding:
   a. Small "draft" model generates k tokens greedily
   b. Large "target" model scores all k tokens in parallel
   c. Accept matching tokens, reject and resample from the first mismatch
```

**Dataset:** `names.txt` — generate names using different strategies from the same trained model.

**Key implementation details:**

- All strategies operate on the same underlying model, making comparison fair
- Beam search requires maintaining B independent KV caches (or re-computing)
- Speculative decoding uses two model sizes (different `n_embd` / `n_layer` configs)
- Print a comparison table: strategy, output, log-prob, tokens/step

**Success criteria:**

- All decoding strategies produce valid token sequences
- Beam search produces higher log-probability sequences than greedy
- Top-p and top-k produce more diverse outputs than greedy (measured by unique name count)
- Speculative decoding accepts ≥50% of draft tokens on average
- Printed comparison table: strategy, output, log-prob, tokens/step
- Runtime: < 7 minutes on M-series Mac

**Expected complexity:** ~450-500 lines. Many small implementations + the speculative decoding two-model setup + inline training.

---

## Implementation Priority & Sequencing

Scripts were built in this order to manage dependencies and validate the shared autograd/model patterns. The canonical autograd interface (`docs/autograd-interface.md`) was finalized before Phase 2.

| Phase       | Scripts                                                              | Rationale                                                                                                                                                                                  |
| ----------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase 1** | `microtokenizer.py`, `microembedding.py`                             | No autograd dependency, standalone algorithms                                                                                                                                              |
| **Phase 2** | `microgpt.py`, `micrornn.py`, `microattention.py`                    | Establishes the canonical autograd `Value` class pattern. microgpt is the reference implementation; micrornn extends it with `sigmoid`. microattention is forward-pass only (no autograd). |
| **Phase 3** | `microrag.py`, `microlora.py`                                        | microrag uses a character-level MLP (lighter autograd dependency). microlora builds directly on microgpt's training pattern.                                                               |
| **Phase 4** | `microdiffusion.py`, `microvae.py`                                   | Independent algorithms, different model families. Can be parallelized with Phase 3.                                                                                                        |
| **Phase 5** | `microdpo.py`, `microppo.py`                                         | Requires stable autograd pattern from Phase 2. microppo uses hybrid autograd (policy: Value class, reward/value: plain floats).                                                            |
| **Phase 6** | `microquant.py`, `microturboquant.py`, `microkv.py`, `microflash.py` | Systems scripts, fully independent of other phases. `microturboquant` is pure forward math (no autograd, no training); all matrix operations are hand-rolled pure-Python loops at D=32.                             |
| **Phase 7** | `microbeam.py`, `micromoe.py`                                        | microbeam trains two models inline (depends on Phase 2 patterns). micromoe uses hybrid autograd (router: Value class, experts: plain floats).                                              |

### Dependency Notes

- **Phase 2 is the critical path.** It establishes the autograd `Value` class that 6 subsequent scripts reimplement. The canonical interface must be validated here.
- **Phases 3 and 4 can run in parallel** — they have no cross-dependencies.
- **Phase 5 scripts** (DPO, PPO) depend on the autograd pattern being stable from Phase 2. They reimplement the base model training loop internally.
- **Hybrid autograd scripts** (microppo, micromoe) use a mix of `Value` objects and plain floats. The canonical interface still applies to the autograd portions.

## Quality Checklist

See `CONTRIBUTING.md` for the complete quality checklist (execution, commenting, readability, logistics). That document is the single authoritative reference for PR review criteria.

---

_Each script is a proof. The algorithm is simpler than you think._

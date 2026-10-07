# Learning Path

A structured guide through the no-magic implementations. Pick a track based on your interest, check off scripts as you complete them, and build intuition for how modern AI/ML systems work under the hood. The tracks currently place 36 of the 48 scripts in [`docs/catalog.json`](docs/catalog.json); the other 12 are listed under [Scripts not yet in a track](#scripts-not-yet-in-a-track).

## How to Use This Guide

1. **Pick a track** that matches your interest or time budget. Tracks 1-2 are weekend-sized. Tracks 3-5 go deeper on specific topics. Track 6 is the longest sequence.
2. **Check off scripts** as you complete them using the `- [ ]` checkboxes.
3. **Each script runs without installation** — just `python <path>`. No virtual environment, no dependencies, no configuration. Scripts whose catalog `data_source` is `names_download` fetch makemore's `names.txt` on first run (network needed once; cached as `names.txt` in the directory you run from). Each script's catalog `teaching_kind` says whether it trains and infers, compares trained variants, runs untrained forward computations, or demonstrates a non-learning algorithm.
4. **Read each script top-to-bottom like a tutorial**, then run it. The comments explain the "why" at every step. After running, experiment: change hyperparameters, swap datasets, break things on purpose.
5. **Prerequisites matter.** Each step lists what it builds on. If you jump into a track mid-way, check the "Builds on" field (in Tracks 1–3, "Why this step here") and backfill gaps.
6. **Predict, then run.** Each step in Tracks 1–3 also gives the exact command, the data it needs, links to the source, paper card, primary paper, preview GIF and (for some steps) an optional lesson in `no-magic-papers`, a question to answer before running with a checked answer, and the limits of what the program shows.

## Time Estimate Summary

| Track | Focus | Time |
|-------|-------|------|
| 1. Weekend Sprint: Transformers | Tokenization through attention | ~4 hrs |
| 2. Weekend Sprint: Alignment | Steering model behavior post-training | ~3 hrs |
| 3. Deep Dive: Modern Inference | Making models fast and small | ~7 hrs |
| 4. Deep Dive: Generative Models | How models create new data | ~4 hrs |
| 5. Deep Dive: Retrieval & Search | Connecting models to external knowledge | ~3 hrs |
| 6. Full Curriculum | 36 of the 48 scripts, dependency-ordered | ~22 hrs |
| 7. Agent Algorithms | Search and reasoning in autonomous agents | ~3 hrs |

---

## Track 1: Weekend Sprint — Transformers (~4 hrs)

From raw text to self-attention. This track builds the core transformer pipeline piece by piece: how text becomes tokens, tokens become vectors, vectors flow through recurrent and attention-based architectures, and how BERT inverts the GPT paradigm. Steps 1–3 are conceptual background: every script is self-contained, and none of the later programs imports or runs code from an earlier one.

### Steps

**1. `01-foundations/microtokenizer.py`**
- **Outcome:** You train a byte-level Byte Pair Encoding tokenizer (256 merges over the raw bytes of `names.txt`), watch each merge shrink the corpus, and check that encoding replays the learned merges in order and decodes back to the original text.
- **Why this step here:** It is the entry point and introduces tokens and vocabularies. It is a conceptual prerequisite only: `microgpt.py` does not use this tokenizer; it builds its own 27-symbol character vocabulary.
- **Run:** `python 01-foundations/microtokenizer.py`
- **Data:** Downloads makemore's `names.txt` (32,033 lowercase names, 228,145 bytes) into the current directory on first run and reuses it afterwards. Run every step from the repository root so all steps share one cached copy.
- **Links:** [source](01-foundations/microtokenizer.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/bpe.md) · [primary paper](https://arxiv.org/abs/1508.07909) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microtokenizer.gif)
- **Predict before you run:** `apply_merge` scans left to right. What does `apply_merge([97, 97, 97], (97, 97), 256)` return, and how many tokens will the final vocabulary hold?
  <details><summary>Check your prediction</summary>

  `[256, 97]`. The first two `a` bytes (97) merge into token 256 and the scan jumps past both, so overlapping pairs resolve left to right. The final vocabulary is 256 byte tokens + 256 merges = 512 tokens; the script prints this before training starts.
  </details>
- **Limits:** 256 merges learned from 228 KB of names is a toy merge table. The linked card is Sennrich et al.'s subword BPE for translation; the script follows GPT-2's byte-level variant and cites Gage (1994) and Radford et al. (2019).
- **Time:** 30 min
- [ ] Completed

**2. `01-foundations/microembedding.py`**
- **Outcome:** You learn 32-dimensional name embeddings by projecting character bigram and trigram counts through one linear layer trained with an InfoNCE contrastive loss on (name, augmented name) pairs, then compare cosine similarity for similar and random name pairs and list nearest neighbours.
- **Why this step here:** It shows how discrete symbols become vectors whose geometry encodes similarity. `microgpt.py` does not reuse these vectors; it learns its own embedding table (`wte`) from scratch.
- **Run:** `python 01-foundations/microembedding.py`
- **Data:** `names.txt`, downloaded on first run as in step 1. Training uses the first 5,000 names; the neighbour search uses the first 10,000.
- **Links:** [source](01-foundations/microembedding.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/word2vec.md) · [primary paper](https://arxiv.org/abs/1301.3781) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microembedding.gif)
- **Predict before you run:** `extract_ngrams` pads a name as `^name$`. How many n-grams does `"anna"` produce, how many of them does `"anne"` share, and how many parameters will the printed model line report?
  <details><summary>Check your prediction</summary>

  `"anna"` gives 9 n-grams (5 bigrams and 4 trigrams). It shares 5 with `"anne"`: `^a`, `an`, `nn`, `^an`, `ann`. The n-gram vocabulary from the first 5,000 names fills its 500-entry cap, so the model line reads `32 x 500 = 16,000 params`.
  </details>
- **Limits:** This is not word2vec. There is no skip-gram or CBOW model, no words and no context window; the script cites SimCLR and sentence-transformers, and the word2vec card is linked only as the closest card. "Similar" here means overlapping spelling: positives are the same name with a deleted or swapped character.
- **Time:** 40 min
- [ ] Completed

**3. `01-foundations/micrornn.py`**
- **Outcome:** You train a vanilla RNN and a GRU for 3,000 steps each on next-character prediction over a 200-name subset, then compare their final loss, the ratio of gradient norms at the first and last time step of one long sequence, and sampled names.
- **Why this step here:** It models sequences with a recurrent hidden state before attention replaces recurrence. The GRU update `h_t = (1 - z_t) * h_{t-1} + z_t * candidate` shows how a gate can pass state (and gradient) through unchanged. `microgpt.py` uses none of this code; the step is a conceptual contrast.
- **Run:** `python 01-foundations/micrornn.py`
- **Data:** `names.txt`, downloaded on first run; training uses 200 names.
- **Links:** [source](01-foundations/micrornn.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/rnn-elman.md) · [primary paper](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/micrornn.gif)
- **Predict before you run:** The vocabulary is 27 symbols (26 letters plus a boundary token) and the hidden size is 32. How many parameters will each model print?
  <details><summary>Check your prediction</summary>

  Vanilla RNN: 2,811 (`W_xh` 32×27 + `W_hh` 32×32 + `b_h` 32 + `W_hy` 27×32 + `b_y` 27). GRU: 6,555 (three input and three recurrent matrices, 3 × (864 + 1,024), plus the same 27×32 output layer and `b_y`, and no hidden bias). That is 2.33 times the vanilla count, not the doubling the source comment mentions.
  </details>
- **Limits:** One run on 200 names is not evidence that GRUs always beat vanilla RNNs, and the gradient ratio comes from a single sequence. The linked card is Elman (1990); the source cites Rumelhart et al. for the vanilla RNN and Cho et al. (2014) for the GRU.
- **Time:** 45 min
- [ ] Completed

**4. `01-foundations/microgpt.py`**
- **Outcome:** You build a one-layer, four-head decoder-only transformer on a scalar autograd `Value` engine, train it for 1,000 Adam steps on next-character prediction (one name per step), and sample 20 names at temperature 0.5.
- **Why this step here:** It combines the earlier ideas — symbols as ids (step 1), learned vectors (step 2) and next-symbol prediction from context (step 3) — but imports none of their code. Its vocabulary is the 26 letters found in `names.txt` plus a BOS token, not the BPE tokenizer, and its embeddings are trained from scratch. Causality comes from the key/value lists: position `t` can only attend to the keys appended at positions `0..t`.
- **Run:** `python 01-foundations/microgpt.py` (add `--interactive` to change `n_embd`, `block_size`, `num_steps` or `learning_rate` and retrain)
- **Data:** `names.txt`, downloaded on first run; the names are shuffled and cycled one per step.
- **Links:** [source](01-foundations/microgpt.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/gpt-1.md) · [primary paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microgpt.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/gpt-1.md)
- **Predict before you run:** What will the `Parameters:` line print?
  <details><summary>Check your prediction</summary>

  `Parameters: 4,192`. Token embeddings 27×16 = 432, position embeddings 16×16 = 256, four attention matrices 4 × 16×16 = 1,024, MLP 64×16 + 16×64 = 2,048 and the output head 27×16 = 432. There are no bias terms.
  </details>
- **Limits:** A character-level toy of about 4,200 parameters. The source follows GPT-2's layout with RMSNorm instead of LayerNorm, ReLU instead of GELU and no biases. GPT-1 adds supervised fine-tuning on downstream tasks and uses a 40,000-merge BPE vocabulary; neither is here. The sampled names show learned character statistics, not language understanding.
- **Time:** 60 min
- [ ] Completed

**5. `01-foundations/microbert.py`**
- **Outcome:** You train an encoder of the same size for 3,000 steps with masked-character prediction (25% of characters replaced by `[MASK]`, at least one per name), then measure top-1 and top-3 fill-in accuracy and compare predictions for one masked slot in different contexts.
- **Why this step here:** It reuses the transformer block from step 4 and changes two things: every position attends to every other position, and the loss covers only the masked positions. The script carries its own copy of the `Value` engine; nothing is imported from `microgpt.py`.
- **Run:** `python 01-foundations/microbert.py`
- **Data:** `names.txt`, downloaded on first run.
- **Links:** [source](01-foundations/microbert.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/bert.md) · [primary paper](https://arxiv.org/abs/1810.04805) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microbert.gif)
- **Predict before you run:** How does the printed parameter count differ from `microgpt.py`'s 4,192?
  <details><summary>Check your prediction</summary>

  `Parameters: 4,224`, 32 more. The vocabulary gains a `[MASK]` token (26 letters + BOS + MASK = 28), which adds one 16-wide row to the token embeddings and one to the prediction head.
  </details>
- **Limits:** BERT masks 15% of tokens with an 80/10/10 mask/random/keep split and also trains next-sentence prediction; the script masks 25% with `[MASK]` only and has neither the split nor sentence pairs. Fill-in accuracy is measured on names taken from the training data.
- **Time:** 45 min
- [ ] Completed

**6. `03-systems/microattention.py`**
- **Outcome:** You run single-head, multi-head (MHA), grouped-query (GQA), multi-query (MQA) and sliding-window attention forward on the same random input (32 positions, width 64, 4 heads) and read an analytic FLOP and memory table alongside each output's cosine similarity to the MHA output.
- **Why this step here:** It generalises the per-head loop of step 4. GQA and MQA reuse the MHA query weights and average groups of MHA key/value weights, so each variant can be compared against the same MHA output.
- **Run:** `python 03-systems/microattention.py` (add `--interactive` to change the sequence length, width, head counts or window)
- **Data:** None to download; inputs and weights are random matrices drawn with seed 42.
- **Links:** [source](03-systems/microattention.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/transformer.md) · [primary paper](https://arxiv.org/abs/1706.03762) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microattention.gif)
- **Predict before you run:** For the GQA row (2 key/value heads) and the MQA row (1), what will the Memory column show?
  <details><summary>Check your prediction</summary>

  GQA: 2 × 2 × 32 × 16 = 2,048 floats; MQA: 2 × 32 × 16 = 1,024 floats, the keys and values cached for 32 positions at head width 16. Careful when comparing rows: the vanilla and MHA rows report the 32×32 score matrix (1,024) and the sliding-window row reports 32 × 8 = 256 scores, which are different quantities.
  </details>
- **Limits:** Nothing is trained. The cosine similarity only says how close each untrained output is to the MHA output when the shared weights are averaged; it is not a quality or accuracy measurement, and the pure-Python timings are not throughput results. Beyond the Transformer card the source cites Shazeer (2019), Ainslie et al. (2023) and Beltagy et al. (2020).
- **Time:** 40 min
- [ ] Completed

---

## Track 2: Weekend Sprint — Alignment (~3 hrs)

How to steer a pretrained model's behavior. This track covers parameter-efficient fine-tuning, preference optimization, and reinforcement learning from human feedback — the techniques that turn a base language model into a useful assistant.

**Prerequisites:** Complete Track 1, or at minimum `01-foundations/microgpt.py` (the autograd `Value` class and transformer architecture are assumed knowledge).

### Steps

**1. `02-alignment/microlora.py`**
- **You'll learn:** How Low-Rank Adaptation freezes pretrained weights and injects small trainable matrices (A and B) that capture task-specific adjustments without modifying the original model.
- **Builds on:** `microgpt` (transformer weights and forward pass).
- **Key moment:** The rank-1 update math — a weight matrix with millions of parameters gets adapted using two tiny matrices whose product has the same shape, dramatically reducing trainable parameters.
- **Time:** 35 min
- [ ] Completed

**2. `02-alignment/microqlora.py`**
- **You'll learn:** How QLoRA combines 4-bit quantization of frozen weights with LoRA adapters, enabling fine-tuning of large models on memory-constrained hardware.
- **Builds on:** `microlora` (LoRA mechanics), `microquant` (quantization concepts, optional but helpful).
- **Key moment:** The double-quantization step — quantizing the quantization constants themselves to squeeze out additional memory savings.
- **Time:** 35 min
- [ ] Completed

**3. `02-alignment/microdpo.py`**
- **You'll learn:** How Direct Preference Optimization converts the RLHF objective into a simple classification loss over preferred vs dispreferred response pairs, eliminating the need for a separate reward model.
- **Builds on:** `microgpt` (language model forward pass and loss computation).
- **Key moment:** The DPO loss derivation — seeing how the Bradley-Terry preference model collapses into a binary cross-entropy loss that directly updates policy weights.
- **Time:** 40 min
- [ ] Completed

**4. `02-alignment/microreinforce.py`**
- **You'll learn:** How the REINFORCE algorithm estimates policy gradients using sampled trajectories and reward signals, forming the foundation of all policy gradient methods.
- **Builds on:** `microgpt` (policy network architecture).
- **Key moment:** The log-probability trick — multiplying the log-prob of each action by its reward creates a gradient that increases the probability of high-reward actions without ever differentiating through the reward function.
- **Time:** 35 min
- [ ] Completed

**5. `02-alignment/microppo.py`**
- **You'll learn:** How Proximal Policy Optimization clips the policy ratio to prevent destructively large updates, making reinforcement learning stable enough for language model training.
- **Builds on:** `microreinforce` (REINFORCE baseline), `microgpt` (model architecture).
- **Key moment:** The clipped surrogate objective — the min-of-two-terms construction that lets the model improve but never stray too far from the previous policy in a single step.
- **Time:** 35 min
- [ ] Completed

---

## Track 3: Deep Dive — Modern Inference (~7 hrs)

Making models fast and small. This track covers every major inference optimization: efficient attention patterns, positional encoding, KV caching, memory management, quantization, decoding strategies, and state-space models.

**Prerequisites:** `01-foundations/microgpt.py` (transformer forward pass and attention mechanics).

### Steps

**1. `03-systems/microattention.py`**
- **You'll learn:** How multi-head, grouped-query, and multi-query attention variants trade quality for throughput by sharing key/value projections.
- **Builds on:** `microgpt` (self-attention fundamentals).
- **Key moment:** The side-by-side output comparison showing grouped-query attention matches multi-head quality with fewer parameters.
- **Time:** 40 min
- [ ] Completed

**2. `03-systems/microflash.py`**
- **You'll learn:** How Flash Attention reorders the attention computation to work in tiles, avoiding materializing the full N x N attention matrix and reducing memory from O(N^2) to O(N).
- **Builds on:** `microattention` (standard attention as baseline).
- **Key moment:** The tiled softmax — computing attention in blocks while maintaining numerical equivalence to the naive implementation via online softmax normalization.
- **Time:** 40 min
- [ ] Completed

**3. `03-systems/microrope.py`**
- **You'll learn:** How Rotary Position Embeddings encode position by rotating query and key vectors in 2D subspaces, giving the model relative position awareness without learned position embeddings.
- **Builds on:** `microattention` (query/key dot product mechanics).
- **Key moment:** The rotation matrix construction — position information is injected by rotating pairs of dimensions, and the dot product between rotated queries and keys naturally depends on their relative distance.
- **Time:** 35 min
- [ ] Completed

**4. `03-systems/microkv.py`**
- **You'll learn:** How KV caching avoids redundant computation during autoregressive generation by storing previously computed key and value tensors and only computing attention for the new token.
- **Builds on:** `microgpt` (autoregressive generation loop).
- **Key moment:** The before/after comparison — generation without caching recomputes all previous tokens at every step; with caching, each new token requires only one new key-value pair.
- **Time:** 35 min
- [ ] Completed

**5. `03-systems/micropaged.py`**
- **You'll learn:** How PagedAttention manages KV cache memory using virtual memory concepts — fixed-size blocks, a page table, and on-demand allocation — eliminating memory fragmentation during batched inference.
- **Builds on:** `microkv` (KV cache fundamentals).
- **Key moment:** The page table lookup — instead of contiguous pre-allocated memory, the cache maps logical positions to physical blocks, enabling efficient memory sharing across sequences.
- **Time:** 40 min
- [ ] Completed

**6. `03-systems/microquant.py`**
- **You'll learn:** How post-training quantization maps 32-bit floating point weights to 8-bit or 4-bit integers using scale and zero-point calibration, shrinking model size with minimal accuracy loss.
- **Builds on:** `microgpt` (trained model weights).
- **Key moment:** The quantization error analysis — seeing exactly where precision loss occurs and how calibration data selection affects the scale/zero-point calculation.
- **Time:** 40 min
- [ ] Completed

**7. `03-systems/microturboquant.py`**
- **You'll learn:** How a single random rotation applied before scalar quantization gives data-oblivious vector compression with provable inner-product preservation — no calibration data required, unlike the methods in `microquant`.
- **Builds on:** `microquant` (scalar quantization mechanics), `microembedding` (vectors as the object being quantized).
- **Key moment:** The rotated-coordinate histogram — raw embedding coordinates have irregular, vector-specific shapes; after one shared random rotation, every vector's coordinates concentrate into the same Beta-shaped marginal, which is what makes a single universal 1-D quantizer optimal for all of them.
- **Time:** 45 min
- [ ] Completed

**8. `03-systems/microbeam.py`**
- **You'll learn:** How beam search, top-k, top-p (nucleus), and temperature sampling explore the output distribution differently, producing outputs that range from deterministic to creative.
- **Builds on:** `microgpt` (autoregressive token generation).
- **Key moment:** Comparing beam search (finds the most probable sequence) against nucleus sampling (samples from the dynamic top-p portion of the distribution) on the same prompt — same model, completely different outputs.
- **Time:** 35 min
- [ ] Completed

**9. `03-systems/microssm.py`**
- **You'll learn:** How state-space models replace attention with a linear recurrence that processes sequences in O(N) time, achieving transformer-competitive quality without the quadratic attention bottleneck.
- **Builds on:** `microgpt` (sequence modeling baseline for comparison).
- **Key moment:** The dual-mode computation — the same SSM parameters support both a parallel convolution mode (fast training) and a sequential recurrence mode (fast inference), unified by the same math.
- **Time:** 35 min
- [ ] Completed

**10. `03-systems/microdiscretize.py`**
- **You'll learn:** How Euler, ZOH, and trapezoidal discretization turn continuous-time SSM equations into discrete recurrences, and why each method creates different stability properties and inductive biases.
- **Builds on:** `microssm` (SSM recurrence mechanics).
- **Key moment:** The stability table — Euler diverges at large delta while ZOH/trapezoidal remain bounded for any step size, because exp() maps the entire negative real line to (0,1).
- **Time:** 40 min
- [ ] Completed

**11. `03-systems/microcomplexssm.py`**
- **You'll learn:** How complex-valued SSM eigenvalues enable rotation (not just decay), why this is mathematically identical to applying data-dependent RoPE rotation matrices, and why real-only SSMs fail at parity.
- **Builds on:** `microssm` (SSM state transitions), `microrope` (rotation matrices, helpful but not required).
- **Key moment:** The equivalence proof — complex and real+RoPE forward passes produce identical outputs to floating-point precision, proving that complex multiply IS 2x2 rotation.
- **Time:** 40 min
- [ ] Completed

**12. `03-systems/microroofline.py`**
- **You'll learn:** How the roofline model classifies operations as memory-bound or compute-bound, and why MIMO SSM state updates (matmul) outperform SISO (outer product) on GPUs despite doing 11x more FLOPs.
- **Builds on:** `microssm` (SSM state updates), `microflash` (hardware-aware algorithm design, helpful but not required).
- **Key moment:** The ASCII roofline plot — seeing SISO at AI≈2 (0.7% GPU utilization) versus MIMO at AI≈32 (shifting toward compute-bound) makes the hardware argument visceral.
- **Time:** 40 min
- [ ] Completed

---

## Track 4: Deep Dive — Generative Models (~4 hrs)

How models create new data. This track covers three generative paradigms: variational autoencoders (compress-and-reconstruct), adversarial networks (generator vs discriminator), and diffusion models (iterative denoising). Each takes a fundamentally different approach to the same problem.

**Prerequisites:** Basic autograd understanding from `01-foundations/microgpt.py` (the `Value` class pattern).

### Steps

**1. `01-foundations/microvae.py`**
- **You'll learn:** How Variational Autoencoders learn a smooth latent space by encoding inputs as distributions (mean + variance) and training with a reconstruction loss plus KL divergence regularizer.
- **Builds on:** `microgpt` (autograd `Value` class for backpropagation).
- **Key moment:** The reparameterization trick — sampling z = mu + sigma * epsilon makes the random sampling step differentiable, which is the key insight that makes VAE training possible.
- **Time:** 50 min
- [ ] Completed

**2. `01-foundations/microgan.py`**
- **You'll learn:** How Generative Adversarial Networks train two competing networks — a generator that creates fake data and a discriminator that tries to detect fakes — pushing both to improve through adversarial pressure.
- **Builds on:** `microgpt` (autograd, neural network training loops).
- **Key moment:** The training instability — watching the generator and discriminator losses oscillate as each adapts to the other's improvements, and seeing how careful learning rate balancing prevents mode collapse.
- **Time:** 50 min
- [ ] Completed

**3. `01-foundations/microdiffusion.py`**
- **You'll learn:** How diffusion models learn to reverse a gradual noising process, generating data by starting from pure noise and iteratively denoising through learned score estimates.
- **Builds on:** `microgpt` (autograd), `microvae` (latent space concepts, helpful but not required).
- **Key moment:** The noise schedule visualization — at each timestep the model predicts and removes a slice of noise, and the generated sample gradually sharpens from static into recognizable structure.
- **Time:** 60 min
- [ ] Completed

---

## Track 5: Deep Dive — Retrieval & Search (~3 hrs)

Connecting models to external knowledge. This track covers how to represent text as vectors, retrieve relevant documents, and tokenize text for downstream tasks. These are the building blocks of RAG systems and search engines.

**Prerequisites:** None — this track is self-contained.

### Steps

**1. `01-foundations/microembedding.py`**
- **You'll learn:** How Word2Vec's skip-gram model learns vector representations where geometric relationships between vectors encode semantic relationships between words.
- **Builds on:** None — this is the entry point.
- **Key moment:** The nearest-neighbor results — querying for similar words returns semantically related terms, despite the model only seeing raw co-occurrence statistics during training.
- **Time:** 40 min
- [ ] Completed

**2. `01-foundations/microrag.py`**
- **You'll learn:** How Retrieval-Augmented Generation combines a vector similarity search over a document store with a language model, grounding generated responses in retrieved evidence.
- **Builds on:** `microembedding` (vector representations for similarity search).
- **Key moment:** The retrieval step — the model's response quality jumps when it conditions on relevant retrieved passages instead of relying solely on its parameters.
- **Time:** 50 min
- [ ] Completed

**3. `01-foundations/microtokenizer.py`**
- **You'll learn:** How BPE tokenization converts arbitrary text into a fixed vocabulary of subword units, balancing vocabulary size against sequence length.
- **Builds on:** None (placed last in this track as a complementary perspective on text preprocessing after seeing embeddings and retrieval).
- **Key moment:** The merge table — each merge reduces total token count, and the final vocabulary captures morphological structure (prefixes, suffixes, stems) without any linguistic rules.
- **Time:** 30 min
- [ ] Completed

---

## Track 6: Full Curriculum (~22 hrs)

36 of the 48 catalog scripts in dependency-respecting order, grouped by conceptual cluster with milestone markers. The remaining 12 are listed under [Scripts not yet in a track](#scripts-not-yet-in-a-track).

### Milestone 1: Text Representation (1.5 hrs)

The foundation — how raw text becomes numbers a model can process.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 1 | `01-foundations/microtokenizer.py` | 30 min | - [ ] |
| 2 | `01-foundations/microembedding.py` | 40 min | - [ ] |

### Milestone 2: Training Fundamentals (1.5 hrs)

Core optimization and regularization techniques that every neural network uses.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 3 | `01-foundations/microoptimizer.py` | 45 min | - [ ] |
| 4 | `02-alignment/microbatchnorm.py` | 25 min | - [ ] |
| 5 | `02-alignment/microdropout.py` | 25 min | - [ ] |

### Milestone 3: Sequence Models (2 hrs)

From recurrence to attention — the architectural evolution that produced modern LLMs.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 6 | `01-foundations/micrornn.py` | 45 min | - [ ] |
| 7 | `01-foundations/microgpt.py` | 60 min | - [ ] |

### Milestone 4: Transformer Variants (1.5 hrs)

Bidirectional models and convolution — alternative architectures built on the same principles.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 8 | `01-foundations/microbert.py` | 45 min | - [ ] |
| 9 | `01-foundations/microconv.py` | 45 min | - [ ] |

### Milestone 5: Retrieval & Grounding (1 hr)

Connecting language models to external knowledge stores.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 10 | `01-foundations/microrag.py` | 50 min | - [ ] |

### Milestone 6: Generative Models (2.5 hrs)

Three paradigms for generating new data — reconstruction, adversarial, and denoising.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 11 | `01-foundations/microvae.py` | 50 min | - [ ] |
| 12 | `01-foundations/microgan.py` | 50 min | - [ ] |
| 13 | `01-foundations/microdiffusion.py` | 60 min | - [ ] |

### Milestone 7: Parameter-Efficient Fine-Tuning (1 hr)

Adapting large models without retraining all parameters.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 14 | `02-alignment/microlora.py` | 35 min | - [ ] |
| 15 | `02-alignment/microqlora.py` | 35 min | - [ ] |

### Milestone 8: Alignment & RL (2 hrs)

Teaching models to follow human preferences through optimization and reinforcement learning.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 16 | `02-alignment/microdpo.py` | 40 min | - [ ] |
| 17 | `02-alignment/microreinforce.py` | 35 min | - [ ] |
| 18 | `02-alignment/microppo.py` | 35 min | - [ ] |
| 19 | `02-alignment/microgrpo.py` | 35 min | - [ ] |

### Milestone 9: Mixture of Experts (0.5 hrs)

Conditional computation — activating only a subset of parameters per input.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 20 | `02-alignment/micromoe.py` | 35 min | - [ ] |

### Milestone 10: Attention Optimization (2 hrs)

Efficient attention patterns, positional encoding, and memory-aware computation.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 21 | `03-systems/microattention.py` | 40 min | - [ ] |
| 22 | `03-systems/microflash.py` | 40 min | - [ ] |
| 23 | `03-systems/microrope.py` | 35 min | - [ ] |

### Milestone 11: Inference Systems (2.5 hrs)

KV caching, memory management, quantization, and decoding strategies.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 24 | `03-systems/microkv.py` | 35 min | - [ ] |
| 25 | `03-systems/micropaged.py` | 40 min | - [ ] |
| 26 | `03-systems/microquant.py` | 40 min | - [ ] |
| 27 | `03-systems/microturboquant.py` | 45 min | - [ ] |
| 28 | `03-systems/microbeam.py` | 35 min | - [ ] |

### Milestone 12: Advanced Systems (1.5 hrs)

State-space models, gradient checkpointing, and parallelism — the frontier of efficient training and inference.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 29 | `03-systems/microssm.py` | 35 min | - [ ] |
| 30 | `03-systems/microcheckpoint.py` | 30 min | - [ ] |
| 31 | `03-systems/microparallel.py` | 30 min | - [ ] |

### Milestone 13: Mamba-3 Deep Dive (2 hrs)

The SSM frontier — discretization methods, complex eigenvalue dynamics, and hardware-aware algorithm design from the Mamba-3 paper.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 32 | `03-systems/microdiscretize.py` | 40 min | - [ ] |
| 33 | `03-systems/microcomplexssm.py` | 40 min | - [ ] |
| 34 | `03-systems/microroofline.py` | 40 min | - [ ] |

### Milestone 14: Agent Algorithms (3 hrs)

How agents search and reason — tree search for planning and tool-augmented reasoning for language agents.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 35 | `04-agents/micromcts.py` | 90 min | - [ ] |
| 36 | `04-agents/microreact.py` | 90 min | - [ ] |

---

## Track 7: Agent Algorithms (~3 hrs)

How autonomous agents find good decisions and reason step-by-step. This track covers tree search and tool-augmented reasoning — the two mechanisms that let agents plan beyond single forward passes.

**Prerequisites:** `02-alignment/microreinforce.py` (the REINFORCE policy gradient that `microreact.py` uses to train its two-layer MLP policy, with hand-written gradients). `micromcts.py` needs no earlier script.

### Steps

**1. `04-agents/micromcts.py`**
- **You'll learn:** How Monte Carlo Tree Search finds strong moves in combinatorial games without exhaustive enumeration, using the UCB1 formula to balance exploring new branches against exploiting known-good ones.
- **Builds on:** no earlier script is required. `micromcts` uses uniformly random rollouts and trains no policy or value model (catalog kind `algorithm_demo`).
- **Key moment:** The UCB1 term in action — early in search, barely-visited nodes have huge exploration bonuses and get selected first; as visit counts grow, the exploitation term dominates and the search concentrates on the best subtree.
- **Time:** 90 min
- [ ] Completed

**2. `04-agents/microreact.py`**
- **You'll learn:** How a Thought→Action→Observation loop interleaves decisions with tool calls, grounding each step in actual observations rather than planning everything upfront. Here a learned policy selects each lookup or compute action over toy in-script tools, and each Thought line is a template describing the selected action; no language model generates the reasoning.
- **Builds on:** `microreinforce` (REINFORCE is used to train the agent's action policy). The policy is a small two-layer MLP over an encoded state, not a language model.
- **Key moment:** The action masking step — the agent's output distribution is zeroed out over illegal or contextually irrelevant actions before sampling, preventing the policy from exploring nonsensical branches and dramatically stabilizing training.
- **Time:** 90 min
- [ ] Completed

---

## Scripts not yet in a track

These catalog scripts are not yet placed in any track above: `01-foundations/attention_vs_none.py`, `01-foundations/microlstm.py`, `01-foundations/microresnet.py`, `01-foundations/microvit.py`, `01-foundations/rnn_vs_gru_vs_lstm.py`, `02-alignment/adam_vs_sgd.py`, `03-systems/microbm25.py`, `03-systems/microspeculative.py`, `03-systems/microvectorsearch.py`, `04-agents/microbandit.py`, `04-agents/micromemory.py` and `04-agents/microminimax.py`.

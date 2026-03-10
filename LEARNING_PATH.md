# Learning Path

A structured guide through all 30 no-magic implementations. Pick a track based on your interest, check off scripts as you complete them, and build intuition for how modern AI/ML systems work under the hood.

## How to Use This Guide

1. **Pick a track** that matches your interest or time budget. Tracks 1-2 are weekend-sized. Tracks 3-5 go deeper on specific topics. Track 6 covers everything.
2. **Check off scripts** as you complete them using the `- [ ]` checkboxes.
3. **Each script runs with zero setup** — just `python <path>`. No virtual environment, no dependencies, no configuration.
4. **Read each script top-to-bottom like a tutorial**, then run it. The comments explain the "why" at every step. After running, experiment: change hyperparameters, swap datasets, break things on purpose.
5. **Prerequisites matter.** Each step lists what it builds on. If you jump into a track mid-way, check the "Builds on" field and backfill gaps.

## Time Estimate Summary

| Track | Focus | Time |
|-------|-------|------|
| 1. Weekend Sprint: Transformers | Tokenization through attention | ~4 hrs |
| 2. Weekend Sprint: Alignment | Steering model behavior post-training | ~3 hrs |
| 3. Deep Dive: Modern Inference | Making models fast and small | ~5 hrs |
| 4. Deep Dive: Generative Models | How models create new data | ~4 hrs |
| 5. Deep Dive: Retrieval & Search | Connecting models to external knowledge | ~3 hrs |
| 6. Full Curriculum | All 30 scripts, dependency-ordered | ~20 hrs |

---

## Track 1: Weekend Sprint — Transformers (~4 hrs)

From raw text to self-attention. This track builds the core transformer pipeline piece by piece: how text becomes tokens, tokens become vectors, vectors flow through recurrent and attention-based architectures, and how BERT inverts the GPT paradigm.

### Steps

**1. `01-foundations/microtokenizer.py`**
- **You'll learn:** How Byte Pair Encoding iteratively merges frequent character pairs to build a subword vocabulary from raw text.
- **Builds on:** None — this is the entry point.
- **Key moment:** Watching the vocabulary grow as the algorithm discovers that common letter pairs like "th" and "er" get merged first, exactly matching linguistic intuition.
- **Time:** 30 min
- [ ] Completed

**2. `01-foundations/microembedding.py`**
- **You'll learn:** How Word2Vec's skip-gram model learns to place semantically similar words near each other in vector space using only co-occurrence patterns.
- **Builds on:** `microtokenizer` (understanding of token vocabularies).
- **Key moment:** The trained vectors capture word relationships — nearest-neighbor queries return semantically related words despite never being told what words mean.
- **Time:** 40 min
- [ ] Completed

**3. `01-foundations/micrornn.py`**
- **You'll learn:** How recurrent neural networks maintain hidden state across time steps, and why GRUs solve the vanishing gradient problem that plagues vanilla RNNs.
- **Builds on:** `microembedding` (vector representations of tokens).
- **Key moment:** Comparing vanilla RNN vs GRU output quality — the gating mechanism visibly improves the model's ability to remember earlier context.
- **Time:** 45 min
- [ ] Completed

**4. `01-foundations/microgpt.py`**
- **You'll learn:** How a decoder-only transformer uses masked self-attention and learned positional encodings to generate text autoregressively, trained with scalar autograd from scratch.
- **Builds on:** `microtokenizer` (tokenization), `micrornn` (sequential modeling concepts), `microembedding` (vector representations).
- **Key moment:** The `Value` class implementing reverse-mode autodiff — a full backward pass through attention, layer norm, and MLP, all in pure Python.
- **Time:** 60 min
- [ ] Completed

**5. `01-foundations/microbert.py`**
- **You'll learn:** How BERT's bidirectional masked language model differs from GPT's autoregressive approach — same transformer blocks, fundamentally different training objective.
- **Builds on:** `microgpt` (transformer architecture, autograd `Value` class).
- **Key moment:** Seeing that BERT attends to tokens both left and right of the mask, while GPT can only look left — a single masking change creates a completely different model.
- **Time:** 45 min
- [ ] Completed

**6. `03-systems/microattention.py`**
- **You'll learn:** How multi-head attention, grouped-query attention, and multi-query attention trade off between quality and compute by sharing key/value heads across query heads.
- **Builds on:** `microgpt` (self-attention fundamentals).
- **Key moment:** The side-by-side comparison showing that grouped-query attention achieves nearly the same output quality as full multi-head attention with significantly fewer parameters.
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

## Track 3: Deep Dive — Modern Inference (~5 hrs)

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

**7. `03-systems/microbeam.py`**
- **You'll learn:** How beam search, top-k, top-p (nucleus), and temperature sampling explore the output distribution differently, producing outputs that range from deterministic to creative.
- **Builds on:** `microgpt` (autoregressive token generation).
- **Key moment:** Comparing beam search (finds the most probable sequence) against nucleus sampling (samples from the dynamic top-p portion of the distribution) on the same prompt — same model, completely different outputs.
- **Time:** 35 min
- [ ] Completed

**8. `03-systems/microssm.py`**
- **You'll learn:** How state-space models replace attention with a linear recurrence that processes sequences in O(N) time, achieving transformer-competitive quality without the quadratic attention bottleneck.
- **Builds on:** `microgpt` (sequence modeling baseline for comparison).
- **Key moment:** The dual-mode computation — the same SSM parameters support both a parallel convolution mode (fast training) and a sequential recurrence mode (fast inference), unified by the same math.
- **Time:** 35 min
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

## Track 6: Full Curriculum (~20 hrs)

All 30 scripts in dependency-respecting order. Grouped by conceptual cluster with milestone markers.

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
| 27 | `03-systems/microbeam.py` | 35 min | - [ ] |

### Milestone 12: Advanced Systems (1.5 hrs)

State-space models, gradient checkpointing, and parallelism — the frontier of efficient training and inference.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 28 | `03-systems/microssm.py` | 35 min | - [ ] |
| 29 | `03-systems/microcheckpoint.py` | 30 min | - [ ] |
| 30 | `03-systems/microparallel.py` | 30 min | - [ ] |

[![no-magic](./assets/banner.png)](https://github.com/Mathews-Tom/no-magic)

---

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/github/license/Mathews-Tom/no-magic?style=flat-square)
![Algorithms](https://img.shields.io/badge/algorithms-41-orange?style=flat-square)
![Version](https://img.shields.io/badge/version-v2.0.0-blue?style=flat-square)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/Mathews-Tom/no-magic?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/Mathews-Tom/no-magic?style=flat-square)

---

# no-magic

**Because `model.fit()` isn't an explanation.**

<video src="https://github.com/user-attachments/assets/f107ed4c-6905-4063-b3f6-a4a3c2f16c8e" width="100%" autoplay loop muted playsinline></video>

---

## What This Is

`no-magic` is a curated collection of single-file, dependency-free Python implementations of the algorithms that power modern AI. Each script is a complete, runnable program that trains a model from scratch and performs inference — no frameworks, no abstractions, no hidden complexity.

Every script in this repository is an **executable proof** that these algorithms are simpler than the industry makes them seem. The goal is not to replace PyTorch or TensorFlow — it's to make you dangerous enough to understand what they're doing underneath.

## See It In Action

<details open>
<summary><h3>01 — Foundations (14 scripts)</h3></summary>
<table>
<tr>
<td align="center"><a href="01-foundations/microgpt.py"><b>Autoregressive GPT</b></a><br/>
<img src="videos/previews/microgpt.gif" width="280"/><br/>
<sub>Token-by-token generation</sub></td>
<td align="center"><a href="01-foundations/micrornn.py"><b>RNN vs GRU</b></a><br/>
<img src="videos/previews/micrornn.gif" width="280"/><br/>
<sub>Vanishing gradients and gating</sub></td>
<td align="center"><a href="01-foundations/microlstm.py"><b>LSTM</b></a><br/>
<img src="videos/previews/microlstm.gif" width="280"/><br/>
<sub>4-gate memory highway</sub></td>
</tr>
<tr>
<td align="center"><a href="01-foundations/microtokenizer.py"><b>BPE Tokenizer</b></a><br/>
<img src="videos/previews/microtokenizer.gif" width="280"/><br/>
<sub>Iterative pair merging → vocabulary</sub></td>
<td align="center"><a href="01-foundations/microembedding.py"><b>Word Embeddings</b></a><br/>
<img src="videos/previews/microembedding.gif" width="280"/><br/>
<sub>Contrastive learning → semantic clusters</sub></td>
<td align="center"><a href="01-foundations/microrag.py"><b>RAG Pipeline</b></a><br/>
<img src="videos/previews/microrag.gif" width="280"/><br/>
<sub>Retrieve → augment → generate</sub></td>
</tr>
<tr>
<td align="center"><a href="01-foundations/microbert.py"><b>BERT</b></a><br/>
<img src="videos/previews/microbert.gif" width="280"/><br/>
<sub>Bidirectional attention + [MASK] prediction</sub></td>
<td align="center"><a href="01-foundations/microconv.py"><b>Convolutional Net</b></a><br/>
<img src="videos/previews/microconv.gif" width="280"/><br/>
<sub>Sliding kernels → feature maps</sub></td>
<td align="center"><a href="01-foundations/microresnet.py"><b>ResNet</b></a><br/>
<img src="videos/previews/microresnet.gif" width="280"/><br/>
<sub>F(x) + x = gradient highway</sub></td>
</tr>
<tr>
<td align="center"><a href="01-foundations/microvit.py"><b>Vision Transformer</b></a><br/>
<img src="videos/previews/microvit.gif" width="280"/><br/>
<sub>Image patches as tokens</sub></td>
<td align="center"><a href="01-foundations/microdiffusion.py"><b>Diffusion</b></a><br/>
<img src="videos/previews/microdiffusion.gif" width="280"/><br/>
<sub>Noise → data via iterative denoising</sub></td>
<td align="center"><a href="01-foundations/microvae.py"><b>VAE</b></a><br/>
<img src="videos/previews/microvae.gif" width="280"/><br/>
<sub>Encode → sample z → decode</sub></td>
</tr>
<tr>
<td align="center"><a href="01-foundations/microgan.py"><b>GAN</b></a><br/>
<img src="videos/previews/microgan.gif" width="280"/><br/>
<sub>Generator vs discriminator minimax</sub></td>
<td align="center"><a href="01-foundations/microoptimizer.py"><b>Optimizers</b></a><br/>
<img src="videos/previews/microoptimizer.gif" width="280"/><br/>
<sub>SGD vs Momentum vs Adam convergence</sub></td>
<td></td>
</tr>
</table>

**Comparison scripts:** [attention_vs_none.py](01-foundations/attention_vs_none.py) · [rnn_vs_gru_vs_lstm.py](01-foundations/rnn_vs_gru_vs_lstm.py)

</details>

<details>
<summary><h3>02 — Alignment & Training (10 scripts)</h3></summary>
<table>
<tr>
<td align="center"><a href="02-alignment/microlora.py"><b>LoRA Fine-tuning</b></a><br/>
<img src="videos/previews/microlora.gif" width="280"/><br/>
<sub>Low-rank weight injection</sub></td>
<td align="center"><a href="02-alignment/microqlora.py"><b>QLoRA</b></a><br/>
<img src="videos/previews/microqlora.gif" width="280"/><br/>
<sub>4-bit base + full-precision adapters</sub></td>
<td align="center"><a href="02-alignment/microdpo.py"><b>DPO Alignment</b></a><br/>
<img src="videos/previews/microdpo.gif" width="280"/><br/>
<sub>Preferred vs. rejected → policy update</sub></td>
</tr>
<tr>
<td align="center"><a href="02-alignment/microppo.py"><b>PPO (RLHF)</b></a><br/>
<img src="videos/previews/microppo.gif" width="280"/><br/>
<sub>Clipped policy gradient for alignment</sub></td>
<td align="center"><a href="02-alignment/microgrpo.py"><b>GRPO</b></a><br/>
<img src="videos/previews/microgrpo.gif" width="280"/><br/>
<sub>Group-relative rewards, no critic</sub></td>
<td align="center"><a href="02-alignment/microreinforce.py"><b>REINFORCE</b></a><br/>
<img src="videos/previews/microreinforce.gif" width="280"/><br/>
<sub>Log P(a) × reward = gradient</sub></td>
</tr>
<tr>
<td align="center"><a href="02-alignment/micromoe.py"><b>Mixture of Experts</b></a><br/>
<img src="videos/previews/micromoe.gif" width="280"/><br/>
<sub>Sparse routing to specialist MLPs</sub></td>
<td align="center"><a href="02-alignment/microbatchnorm.py"><b>Batch Normalization</b></a><br/>
<img src="videos/previews/microbatchnorm.gif" width="280"/><br/>
<sub>Normalize activations → stable training</sub></td>
<td align="center"><a href="02-alignment/microdropout.py"><b>Dropout</b></a><br/>
<img src="videos/previews/microdropout.gif" width="280"/><br/>
<sub>Kill neurons → prevent overfitting</sub></td>
</tr>
</table>

**Comparison scripts:** [adam_vs_sgd.py](02-alignment/adam_vs_sgd.py)

</details>

<details>
<summary><h3>03 — Systems & Inference (13 scripts)</h3></summary>
<table>
<tr>
<td align="center"><a href="03-systems/microattention.py"><b>Attention Mechanism</b></a><br/>
<img src="videos/previews/microattention.gif" width="280"/><br/>
<sub>Q·K<sup>T</sup> → softmax → weighted V</sub></td>
<td align="center"><a href="03-systems/microflash.py"><b>Flash Attention</b></a><br/>
<img src="videos/previews/microflash.gif" width="280"/><br/>
<sub>Tiled O(N) memory computation</sub></td>
<td align="center"><a href="03-systems/microrope.py"><b>RoPE</b></a><br/>
<img src="videos/previews/microrope.gif" width="280"/><br/>
<sub>Position via rotation matrices</sub></td>
</tr>
<tr>
<td align="center"><a href="03-systems/microkv.py"><b>KV-Cache</b></a><br/>
<img src="videos/previews/microkv.gif" width="280"/><br/>
<sub>Memoize keys/values — stop recomputing</sub></td>
<td align="center"><a href="03-systems/micropaged.py"><b>PagedAttention</b></a><br/>
<img src="videos/previews/micropaged.gif" width="280"/><br/>
<sub>OS-style paged KV-cache memory</sub></td>
<td align="center"><a href="03-systems/microquant.py"><b>Quantization</b></a><br/>
<img src="videos/previews/microquant.gif" width="280"/><br/>
<sub>Float32 → Int8 = 4x compression</sub></td>
</tr>
<tr>
<td align="center"><a href="03-systems/microbeam.py"><b>Beam Search</b></a><br/>
<img src="videos/previews/microbeam.gif" width="280"/><br/>
<sub>Tree search with top-k pruning</sub></td>
<td align="center"><a href="03-systems/microcheckpoint.py"><b>Checkpointing</b></a><br/>
<img src="videos/previews/microcheckpoint.gif" width="280"/><br/>
<sub>O(n) → O(√n) memory via recompute</sub></td>
<td align="center"><a href="03-systems/microparallel.py"><b>Model Parallelism</b></a><br/>
<img src="videos/previews/microparallel.gif" width="280"/><br/>
<sub>Tensor + pipeline across devices</sub></td>
</tr>
<tr>
<td align="center"><a href="03-systems/microssm.py"><b>State Space Models</b></a><br/>
<img src="videos/previews/microssm.gif" width="280"/><br/>
<sub>Linear-time selective state transitions</sub></td>
<td align="center"><a href="03-systems/microvectorsearch.py"><b>Vector Search</b></a><br/>
<img src="videos/previews/microvectorsearch.gif" width="280"/><br/>
<sub>Exact vs LSH approximate search</sub></td>
<td align="center"><a href="03-systems/microbm25.py"><b>BM25</b></a><br/>
<img src="videos/previews/microbm25.gif" width="280"/><br/>
<sub>TF → TF-IDF → BM25 evolution</sub></td>
</tr>
<tr>
<td align="center"><a href="03-systems/microspeculative.py"><b>Speculative Decoding</b></a><br/>
<img src="videos/previews/microspeculative.gif" width="280"/><br/>
<sub>Draft fast, verify once</sub></td>
<td></td>
<td></td>
</tr>
</table>

</details>

<details>
<summary><h3>04 — Agents & Planning (2 scripts)</h3></summary>
<table>
<tr>
<td align="center"><a href="04-agents/micromcts.py"><b>Monte Carlo Tree Search</b></a><br/>
<img src="videos/previews/micromcts.gif" width="280"/><br/>
<sub>UCB1 tree search + random rollouts</sub></td>
<td align="center"><a href="04-agents/microreact.py"><b>ReAct Agent</b></a><br/>
<img src="videos/previews/microreact.gif" width="280"/><br/>
<sub>Thought → Action → Observation</sub></td>
<td></td>
</tr>
</table>

</details>

> All algorithms have animated visualizations. Full 1080p60 videos in [Releases](https://github.com/Mathews-Tom/no-magic/releases).
> Video source scenes in [`videos/scenes/`](videos/scenes/) — built with [Manim](https://www.manim.community/).

### Rendering Videos Locally

All visualizations can be rendered from source. System dependencies: `cairo`, `pango`, `ffmpeg`, and optionally `gifsicle` for GIF optimization.

```bash
# Install Manim (one-time)
pip install -r videos/requirements.txt

# macOS system deps (one-time)
brew install cairo pango ffmpeg gifsicle

# Ubuntu/Debian system deps (one-time)
sudo apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg gifsicle
```

**Using the Python renderer** (`render_all.py`):

```bash
# Render all scenes — full 1080p60 MP4 + 480p GIF previews
python videos/render_all.py

# Render specific scenes only
python videos/render_all.py microattention microgpt microlora

# Full MP4s only (no GIFs)
python videos/render_all.py --full-only

# GIF previews only (faster)
python videos/render_all.py --preview-only

# Custom quality (low/medium/high/4k)
python videos/render_all.py --quality medium

# Skip GIF optimization step
python videos/render_all.py --preview-only --skip-optimize
```

**Using the shell renderer** (`render.sh`):

```bash
bash videos/render.sh                    # all scenes (MP4 + GIF)
bash videos/render.sh microattention     # single scene
bash videos/render.sh --preview-only     # GIF previews only
bash videos/render.sh --full-only        # MP4s only
```

Output lands in `videos/renders/` (MP4) and `videos/previews/` (GIF). Full rendering details in [`videos/README.md`](videos/README.md).

## Philosophy

Modern ML education has a gap. There are thousands of tutorials that teach you to call library functions, and there are academic papers full of notation. What's missing is the middle layer: **the algorithm itself, expressed as readable code**.

This project follows a strict set of constraints:

- **One file, one algorithm.** Every script is completely self-contained. No imports from local modules, no `utils.py`, no shared libraries.
- **Zero external dependencies.** Only Python's standard library. If it needs `pip install`, it doesn't belong here.
- **Train and infer.** Every script includes both the learning loop and generation/prediction. You see the full lifecycle.
- **Runs in minutes on a CPU.** No GPU required. No cloud credits. Every script completes on a laptop in reasonable time.
- **Comments are mandatory, not decorative.** Every script must be readable as a guided walkthrough of the algorithm. We are not optimizing for line count — we are optimizing for understanding. See `CONTRIBUTING.md` for the full commenting standard.

## Who This Is For

- **ML engineers** who use frameworks daily but want to understand the internals they rely on.
- **Students** transitioning from theory to practice who want to see algorithms as working code, not just equations.
- **Career switchers** entering ML who need intuition for what's actually happening when they call high-level APIs.
- **Researchers** who want minimal reference implementations to prototype ideas without framework overhead.
- **Anyone** who has ever stared at a library call and thought: _"but what is it actually doing?"_

This is not a beginner's introduction to programming. You should be comfortable reading Python and have at least a surface-level familiarity with ML concepts. The scripts will give you the depth.

## What You'll Find Here

The repository is organized into four tiers based on conceptual dependency:

### 01 — Foundations (14 scripts)

Core algorithms that form the building blocks of modern AI systems. GPT, RNN, LSTM, BERT, CNN, ResNet, ViT, GAN, VAE, diffusion, embeddings, tokenization, RAG, and optimizer comparison. Includes comparison scripts for attention mechanisms and recurrent architectures.

See [`01-foundations/README.md`](01-foundations/README.md) for the full algorithm list, timing data, and roadmap.

### 02 — Alignment & Training Techniques (10 scripts)

Methods for steering, fine-tuning, and aligning models after pretraining. LoRA, QLoRA, DPO, PPO, GRPO, REINFORCE, MoE, batch normalization, dropout/regularization, and optimizer comparison.

See [`02-alignment/README.md`](02-alignment/README.md) for the full algorithm list, timing data, and roadmap.

### 03 — Systems & Inference (13 scripts)

The engineering that makes models fast, small, and deployable. Attention variants, Flash Attention, KV-cache, PagedAttention, RoPE, quantization, beam search, checkpointing, parallelism, SSMs, vector search, BM25, and speculative decoding.

See [`03-systems/README.md`](03-systems/README.md) for the full algorithm list, timing data, and roadmap.

### 04 — Agents & Planning (2 scripts)

Autonomous reasoning and decision-making. Monte Carlo Tree Search for strategic planning and ReAct agents for tool-augmented reasoning loops.

See [`04-agents/README.md`](04-agents/README.md) for the full algorithm list, timing data, and roadmap.

## How to Use This Repo

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/no-magic.git
cd no-magic

# Pick any script and run it
python 01-foundations/microgpt.py
```

That's it. No virtual environments, no dependency installation, no configuration. Each script will download any small datasets it needs on first run.

### Minimum Requirements

- Python 3.10+
- 8 GB RAM
- Any modern CPU (2019-era or newer)

### Quick Start Path

If you're working through the scripts systematically, this subset builds core concepts incrementally:

```text
microtokenizer.py     → How text becomes numbers
microembedding.py     → How meaning becomes geometry
microgpt.py           → How sequences become predictions
micrornn.py           → How recurrence models sequences
microlstm.py          → How gated memory solves vanishing gradients
microbert.py          → How bidirectional context differs from autoregressive
microconv.py          → How spatial filters extract features
microvit.py           → How transformers see images
microbatchnorm.py     → How normalizing activations stabilizes training
microlora.py          → How fine-tuning works efficiently
microdpo.py           → How preference alignment works
microattention.py     → How attention actually works (all variants)
microrope.py          → How position gets encoded through rotation
microquant.py         → How models get compressed
microflash.py         → How attention gets fast
microssm.py           → How Mamba models bypass attention entirely
microreact.py         → How agents reason with tools
```

Each tier's README has the full algorithm list with measured run times for that category.

## Learning Resources

### Challenges

"Predict the behavior" exercises that test your understanding of the algorithms. 5 challenges covering attention, GPT, GAN, DPO, and optimizer edge cases. Each challenge presents a code snippet and asks you to reason about the output before running it.

See [`challenges/README.md`](challenges/README.md) for the full challenge set.

### Flashcards

Anki-compatible flashcard decks for spaced repetition review. 147 cards across 3 tiers (foundations, alignment, systems), covering key concepts, equations, and design decisions from every script.

```bash
# Generate the Anki deck
python resources/flashcards/generate_anki.py
```

See [`resources/flashcards/`](resources/flashcards/) for the raw card data and generation script.

### Learning Path

Structured tracks for different goals — 6 learning tracks ranging from weekend sprints to a full 20-hour curriculum. Each track orders scripts by conceptual dependency and includes time estimates, prerequisites, and milestone markers.

See [`LEARNING_PATH.md`](LEARNING_PATH.md) for the full guide.

## Translations

Comment translations for 6 languages: Spanish, Portuguese, Chinese, Japanese, Korean, and Hindi. The code stays in English — only comments, docstrings, section headers, and print statements are translated.

See [`TRANSLATIONS.md`](TRANSLATIONS.md) for full status and contributor guide.

Want to help translate? See the [translation guide](translations/README.md).

## Dependency Graph

How the algorithms connect conceptually. Arrows mean "understanding A helps with B" — not code imports (every script is fully self-contained).

```mermaid
graph LR
  %% --- Style definitions ---
  classDef foundations fill:#4a90d9,stroke:#2c5f8a,color:#fff
  classDef alignment fill:#e8834a,stroke:#b35f2e,color:#fff
  classDef systems fill:#5bb55b,stroke:#3a823a,color:#fff

  %% === 01-FOUNDATIONS ===
  subgraph F["01 — Foundations"]
    TOK["Tokenizer"]
    EMB["Embedding"]
    OPT["Optimizer"]
    RNN["RNN / GRU"]
    CONV["Conv Net"]
    GPT["GPT"]
    BERT["BERT"]
    RAG["RAG"]
    DIFF["Diffusion"]
    VAE["VAE"]
    GAN["GAN"]
  end

  %% === 02-ALIGNMENT ===
  subgraph A["02 — Alignment"]
    BN["BatchNorm"]
    DROP["Dropout"]
    LORA["LoRA"]
    QLORA["QLoRA"]
    DPO["DPO"]
    REINF["REINFORCE"]
    PPO["PPO"]
    GRPO["GRPO"]
    MOE["MoE"]
  end

  %% === 03-SYSTEMS ===
  subgraph S["03 — Systems"]
    ATTN["Attention"]
    FLASH["Flash Attn"]
    ROPE["RoPE"]
    KV["KV-Cache"]
    PAGED["PagedAttn"]
    QUANT["Quantization"]
    BEAM["Beam Search"]
    CKPT["Checkpointing"]
    PAR["Parallelism"]
    SSM["SSM / Mamba"]
  end

  %% --- Foundation internals ---
  TOK --> GPT
  EMB --> RAG
  RNN --> GPT
  OPT --> GPT
  GPT --> BERT
  DIFF -.-> VAE
  DIFF -.-> GAN

  %% --- Foundations → Alignment ---
  GPT --> LORA
  GPT --> DPO
  GPT --> PPO
  GPT --> MOE
  GPT --> GRPO
  LORA --> QLORA
  REINF --> PPO
  REINF --> GRPO
  OPT --> BN
  OPT --> DROP

  %% --- Foundations → Systems ---
  GPT --> ATTN
  GPT --> KV
  GPT --> QUANT
  GPT --> BEAM
  GPT --> SSM
  RNN --> SSM
  ATTN --> FLASH
  ATTN --> ROPE
  KV --> PAGED

  %% --- Cross-tier into QLoRA ---
  QUANT --> QLORA

  %% --- Apply styles ---
  class TOK,EMB,OPT,RNN,CONV,GPT,BERT,RAG,DIFF,VAE,GAN foundations
  class BN,DROP,LORA,QLORA,DPO,REINF,PPO,GRPO,MOE alignment
  class ATTN,FLASH,ROPE,KV,PAGED,QUANT,BEAM,CKPT,PAR,SSM systems
```

**Legend:** <span style="color:#4a90d9">Foundations</span> · <span style="color:#e8834a">Alignment</span> · <span style="color:#5bb55b">Systems</span> — Solid arrows = strong prerequisite, dashed arrows = conceptual comparison.

## Inspiration & Attribution

This project is directly inspired by [Andrej Karpathy's](https://github.com/karpathy) extraordinary work on minimal implementations — particularly [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), and the `microgpt.py` script that demonstrated the entire GPT algorithm in a single dependency-free Python file.

Karpathy proved that there's enormous demand for "the algorithm, naked." `no-magic` extends that philosophy across the full landscape of modern AI/ML.

## How This Was Built

In the spirit of transparency: this repository was co-authored with Claude (Anthropic). I designed the project — which algorithms to include, the four-tier structure, the constraint system, the learning paths, and how each script should be organized — then directed the implementations and verified that every script trains and infers correctly end-to-end on CPU.

The scope goes beyond code. The animated visualizations (Manim scenes), predict-the-behavior challenges, Anki flashcards, learning path tracks, EPUB generation pipeline, and translation infrastructure were all designed collaboratively — I set the requirements and structure, Claude helped execute. Every artifact was reviewed and validated.

I'm not claiming to have hand-typed every algorithm from scratch. The value of this project is in the curation, the architectural decisions, and the fact that everything works as a self-contained learning resource — from the scripts themselves to the supporting materials that help you internalize what they teach.

This is how I build in 2026. I'd rather be upfront about it.

## Contributing

Contributions are welcome, but the constraints are non-negotiable. See `CONTRIBUTING.md` for the full guidelines. The short version:

- One file. Zero dependencies. Trains and infers.
- If your PR adds a `requirements.txt`, it will be closed.
- Quality over quantity. Each script should be the **best possible** minimal implementation of its algorithm.

## License

MIT — use these however you want. Learn from them, teach with them, build on them.

---

_The constraint is the product. Everything else is just efficiency._

_v2.0.0 — March 2026_

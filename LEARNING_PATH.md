# Learning Path

A structured guide through the no-magic implementations. Pick a track based on your interest, check off scripts as you complete them, and build intuition for how modern AI/ML systems work under the hood. The tracks currently place 38 of the 50 scripts in [`docs/catalog.json`](docs/catalog.json); the other 12 are listed under [Scripts not yet in a track](#scripts-not-yet-in-a-track).

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
| 2. Weekend Sprint: Alignment | Steering model behavior post-training | ~4 hrs 10 min |
| 3. Deep Dive: Modern Inference | Making models fast and small | ~7 hrs |
| 4. Deep Dive: Generative Models | How models create new data | ~4 hrs |
| 5. Deep Dive: Retrieval & Search | Connecting models to external knowledge | ~3 hrs |
| 6. Full Curriculum | 38 of the 50 scripts, dependency-ordered | ~22 hrs |
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
- **Limits:** Nothing is trained. The cosine similarity only says how close each untrained output is to the MHA output when the shared weights are averaged; it is not a quality or accuracy measurement, and the pure-Python timings are not throughput results. The printed takeaway that the sliding window is "4x cheaper" at this length (32 / 8) counts only the attention-score work; the table's totals, which include the projections, are 1,310,720 FLOPs for MHA and 1,114,112 for the window, about 1.18x. Beyond the Transformer card the source cites Shazeer (2019), Ainslie et al. (2023) and Beltagy et al. (2020).
- **Time:** 40 min
- [ ] Completed

---

## Track 2: Weekend Sprint — Alignment (~3 hrs)

The estimate in this heading is kept unchanged so existing links to it keep working; the current exercises below total 4 hrs 10 min.

How to steer a pretrained model's behavior. This track covers parameter-efficient fine-tuning, supervised fine-tuning on demonstrations, preference optimization, reinforcement learning from human feedback, and distillation into a smaller model — the techniques that turn a base language model into a useful assistant. Every preference signal here is synthetic: the scripts prefer names of certain lengths in place of human judgments, so they show the mechanics of each method, not alignment to people. The last step, `microdistill.py`, shows the general teacher-to-student transfer mechanism on a small 2-D classifier, not on a language model.

**Prerequisites:** Complete Track 1, or at minimum `01-foundations/microgpt.py` (the autograd `Value` class and transformer architecture are assumed knowledge).

### Steps

**1. `02-alignment/microlora.py`**
- **Outcome:** You pretrain a `microgpt`-sized base model on names starting A–M (800 steps), freeze it, train rank-2 adapters on the query and value projections on names starting N–Z (500 steps), and compare base and adapted loss on both splits.
- **Why this step here:** Every later step adapts or fine-tunes a small GPT, and LoRA is the cheapest way to change one: the base weights stay fixed and only two small matrices per adapted projection learn. It needs the forward pass and autograd from `microgpt.py`; the script carries its own copy of both.
- **Run:** `python 02-alignment/microlora.py`
- **Data:** `names.txt`, downloaded on first run and split by first letter.
- **Links:** [source](02-alignment/microlora.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/lora.md) · [primary paper](https://arxiv.org/abs/2106.09685) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microlora.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/lora.md)
- **Predict before you run:** The script adds `A @ (B @ x)` to each frozen projection, with `A` of shape 16×2 drawn from N(0, 0.02) and `B` of shape 2×16 set to zero. (1) How many trainable parameters does the results line report against the 4,192 base parameters? (2) On the first adaptation step, which of `A` and `B` receives a nonzero gradient?
  <details><summary>Check your prediction</summary>

  (1) `LoRA: 128 (3.1%)`: each adapter has 16×2 + 2×16 = 64 parameters, and there are two (query and value). (2) Only `B`. Because `B` starts at zero, `B @ x = 0`, so the gradient of `A` (the output-side factor) is zero; the gradient of `B` (the input-side factor) is `Aᵀ (∂L/∂y) xᵀ`, which is generally nonzero (it vanishes only when `Aᵀ (∂L/∂y)` or `x` does). The zero gradient for `A` holds for any input; the size of `B`'s gradient depends on the name, the base weights and the seed. One run of the script's own adaptation loop for a single step, on an untrained base with the name `olivia`, gave exactly 0 for `A` and a nonzero gradient for `B`: after the Adam update `B` had changed and `A` and the base weights had not. Only the adapters are updated, the base stays frozen, and no α/r scaling is applied. The LoRA paper initializes the other way round: it writes `W₀ + BA`, sets its output-side `B` (d×r) to zero and draws its input-side `A` (r×k) from a Gaussian, so there the output-side factor moves first. That is a different initialization, not the same one with the letters swapped.
  </details>
- **Limits:** No full fine-tuning is run: the "Full fine-tune" number in the results line is only the base parameter count, so nothing here shows that LoRA matches full fine-tuning. The adapter output is not scaled by α/r. The loss comparison is one run at rank 2 on one split.
- **Time:** 35 min
- [ ] Completed

**2. `02-alignment/microqlora.py`**
- **Outcome:** You pretrain a base model at full precision on 80% of the shuffled names (800 steps), quantize its attention and MLP weights to 4-bit NF4 levels in blocks of 8 with double-quantized INT8 scales, then train rank-2 adapters on the query and value projections on the other 20% (500 steps) and sample names.
- **Why this step here:** It adds quantized frozen weights to the LoRA recipe from step 1. `03-systems/microquant.py` in Track 3 covers the quantization arithmetic in more depth; it helps but is not required.
- **Run:** `python 02-alignment/microqlora.py`
- **Data:** `names.txt`, downloaded on first run.
- **Links:** [source](02-alignment/microqlora.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/qlora.md) · [primary paper](https://arxiv.org/abs/2305.14314) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microqlora.gif)
- **Predict before you run:** Only the attention and MLP matrices are quantized (3,072 weights); each row is cut into blocks of 8 weights with one scale per block. What will the three memory lines and the compression ratio print?
  <details><summary>Check your prediction</summary>

  There are 384 blocks, so 384 scales. FP32: 3,072 × 4 = 12,288 bytes. NF4: 1,536 bytes of 4-bit codes + 384 × 4 = 1,536 scale bytes = 3,072 bytes. NF4 with double quantization: 1,536 + (384 × 1 + 4) = 1,924 bytes. Compression: 12,288 / 1,924 = 6.4x.
  </details>
- **Limits:** A one-layer, 16-dimensional toy: NF4 levels come from a normal-quantile approximation, blocks hold 8 weights instead of the 64 used in production, the embeddings and output head stay in full precision, and the adapter output is not scaled by α/r even though a docstring mentions it. Unlike `microlora.py`, the zero-initialized factor here is the output-side one, as in the LoRA paper, but the names are swapped: the code's `lora_B` (2×16) is the random input-side factor and `lora_A` (16×2) is the zero output-side factor.
- **Time:** 35 min
- [ ] Completed

**3. `02-alignment/microsft.py`**
- **Outcome:** You pretrain a 1,120-parameter one-layer, one-head decoder on eight cyclic strings over `a`–`h` (200 updates, one sampled string each). Keeping those learned weights, you fine-tune them on 14 synthetic `copy:<s>>` and `next:<s>>` demonstrations with a loss on the response only (300 updates, each averaging all 14 pairs). Then you compare the base model's and the fine-tuned model's greedy answers on the training prompts and on two held-out prompts.
- **Why this step here:** LoRA and QLoRA trained a few adapter weights on more names. SFT updates every weight so the model reproduces demonstrated responses, and it is the first stage of the InstructGPT pipeline that the preference methods in steps 4 and 6 continue from. It needs `microgpt.py`'s decoder and causal next-token loss; the script carries its own smaller copy of both.
- **Run:** `python 02-alignment/microsft.py`
- **Data:** None to download; the base strings and the demonstrations are generated in the script.
- **Links:** [source](02-alignment/microsft.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/instructgpt.md) · [primary paper](https://arxiv.org/abs/2203.02155) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microsft.gif)
- **Predict before you run:** The pretraining strings contain only `a`–`h` and the boundary token. (1) When the program reports whether the prompt-only embedding rows (`n o p t x y : >`) were untouched by pretraining, what will it say? (2) Each SFT pair is 10 tokens long. How many supervised targets per pair does the program report, and does the masked loss still send gradient into those prompt-only rows?
  <details><summary>Check your prediction</summary>

  (1) `True`, and exactly so. These tokens are never inputs during pretraining, so their embedding rows get zero gradient. Adam's moments stay at 0, and the update `0 / (0 + 1e-8)` is exactly zero. (2) 2 of the 9 next-token positions: the response character, predicted at position 7, and the end boundary, predicted at position 8. The gradient is nonzero, because the prompt positions feed the attention cache that position 7 reads. Masking removes prompt targets, not prompt inputs. Both answers were checked on the script's own helpers with reduced budgets, not a default run. Ten pretraining updates left the prompt-only rows bit-for-bit unchanged while other weights moved. The masked loss of one pair equalled the mean of the position-7 and position-8 negative log-likelihoods. At initialization, the full-batch SFT gradient on the prompt-only rows summed to about 0.67 in absolute value, with exactly 0 on the never-used position-9 embedding. The magnitude the program prints depends on the pretrained weights.
  </details>
- **Limits:** A 1,120-parameter toy on synthetic two-command demonstrations, not GPT-3, human labelers or InstructGPT's results. The response-only mask is a teaching choice that the InstructGPT paper does not prescribe. Full-batch SFT was adopted after a one-pair-per-update version failed the program's own `next` check at seed 42 (3/7 against the required 4/7). The support for the change is a post-hoc training-prompt comparison over seeds 0–15 (14/16 passing against 4/16), not a pre-registered test, and two of those seeds still fail. The acceptance check uses training prompts only; the two held-out prompts are reported, with no promise that they improve. The base-corpus loss after SFT is printed to show how much pretraining behavior moved, not to bound it.
- **Time:** 35 min
- [ ] Completed

**4. `02-alignment/microdpo.py`**
- **Outcome:** You pretrain a base model (700 steps), freeze a copy as the reference policy, build up to 150 synthetic preference pairs that prefer a name of 5 or more letters (chosen) over a 3-letter name (rejected) sharing its first two letters — so the rejected completion after that prefix is a single letter — run 60 DPO steps with β = 0.1, and compare the average generated length of the reference and aligned models.
- **Why this step here:** It changes a model from preference pairs with one supervised loss and no reward model, sampling or RL loop. It needs `microgpt.py`'s sequence log-probabilities. Steps 5 and 6 then show the reinforcement-learning route that DPO avoids.
- **Run:** `python 02-alignment/microdpo.py`
- **Data:** `names.txt`, downloaded on first run; the preference pairs are built from it by name length.
- **Links:** [source](02-alignment/microdpo.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/dpo.md) · [primary paper](https://arxiv.org/abs/2305.18290) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdpo.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/dpo.md)
- **Predict before you run:** At DPO step 1 the policy weights are still identical to the frozen reference. What will `dpo_loss` and the two mean rewards print?
  <details><summary>Check your prediction</summary>

  Every log-ratio `log π(y|x) − log π_ref(y|x)` is zero, so the margin is zero and each pair's loss is `log(1 + e⁰) = ln 2 ≈ 0.6931`; both mean rewards are 0.00 (floating-point rounding can leave a sign, as in `-0.00`). Calling the script's `dpo_loss` on eight pairs with the policy equal to its snapshot gave 0.69314718 for each pair and rewards within 4 × 10⁻¹⁶ of zero.
  </details>
- **Limits:** The preferences are a length rule, not human judgments. Each pair is scored as a whole sequence from the BOS token, prompt included; the shared prefix adds the same term to both log-ratios, so it cancels in the margin. The result is a shift in generated length on names; it does not show that DPO matches or beats RLHF, or that it replaces preference-data quality. The comment beside `DPO_BETA` describes β backwards: in the DPO paper β weights the KL penalty toward the reference, so a larger β keeps the policy closer to it and a smaller β lets it move further.
- **Time:** 40 min
- [ ] Completed

**5. `02-alignment/microreinforce.py`**
- **Outcome:** You train a small policy network that emits 8-letter strings scored by hand-written rules, first with raw REINFORCE and then with an exponential-moving-average reward baseline, and compare gradient-norm variance, average reward and samples.
- **Why this step here:** PPO in step 6 builds directly on the REINFORCE gradient, the log-probability of each sampled action weighted by the reward. The policy is a two-layer MLP over the previous letter and position, not a language model; from `microgpt.py` you need only the `Value` engine and softmax log-probabilities, which the script re-implements.
- **Run:** `python 02-alignment/microreinforce.py`
- **Data:** None to download; the policy samples letters from its own 26-letter vocabulary.
- **Links:** [source](02-alignment/microreinforce.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/reinforce.md) · [primary paper](https://link.springer.com/article/10.1007/BF00992696) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microreinforce.gif)
- **Predict before you run:** `generate_trajectory` has no stop action, so every sample is exactly `MAX_SEQ_LEN = 8` letters. Which reward rules can never fire, and what is the highest reward a sample can actually earn?
  <details><summary>Check your prediction</summary>

  The 4–6 letter bonus (+1) and the short-sequence penalty (−2) never apply. The best reachable reward is 1 (vowel first) + 1 (consonant last) + 2.5 (all five vowels) = 4.5, not the "~5.5" the script prints; the script's own `compute_reward` returns 4.5 for `aeioubcd` and gives 5.5 only to a 6-letter string such as `aeiouz`, which the sampler cannot produce.
  </details>
- **Limits:** A toy reward on letter strings with one random seed; the variance comparison comes from gradient norms sampled every 10 episodes in one run.
- **Time:** 35 min
- [ ] Completed

**6. `02-alignment/microppo.py`**
- **Outcome:** You pretrain a smaller GPT (8-dimensional, 2 heads, 500 steps), train an MLP reward model on synthetic pairs that prefer 4–7 letter names, then run 100 policy updates with a clipped surrogate objective, a squared log-ratio penalty against the pretrained policy (coefficient 0.5) and a linear value baseline, and compare rewards and samples before and after.
- **Why this step here:** It is the full RLHF loop — pretrain, reward model, policy optimisation — built from the REINFORCE gradient of step 5 plus a learned baseline and a penalty that keeps the policy near its starting point. DPO (step 4) reaches a related objective without the reward model and sampling.
- **Run:** `python 02-alignment/microppo.py`
- **Data:** `names.txt`, downloaded on first run; the preference pairs are built from it by name length.
- **Links:** [source](02-alignment/microppo.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/ppo.md) · [primary paper](https://arxiv.org/abs/1707.06347) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microppo.gif)
- **Predict before you run:** Each step samples 4 completions, records their log-probabilities as `old_logp` from the current weights, and then makes exactly one gradient update. What is the ratio `π_new / π_old` inside that update, and does the clip at [0.8, 1.2] ever change the objective?
  <details><summary>Check your prediction</summary>

  The ratio is `exp(0) = 1` for every sample, because `old_logp` and the current log-probability come from the same weights; the script's two log-probability functions agreed to within 4 × 10⁻¹⁵ on sampled completions. A ratio of 1 is inside [0.8, 1.2], so the clip never binds here and the update is an advantage-weighted policy gradient plus the penalty. Clipping only matters when one batch is reused for several updates, which this script does not do.
  </details>
- **Limits:** The preferences are synthetic, and the reward adds an explicit length bonus on top of the learned reward model. The printed `kl_div` is the mean absolute difference between policy and reference sequence log-probabilities, not a KL estimate. The reward model (an MLP) and the value function (a linear model) use plain floats and hand-written SGD, not autograd.
- **Time:** 35 min
- [ ] Completed

**7. `02-alignment/microdistill.py`**
- **Outcome:** You train a 99-parameter ReLU MLP teacher on three synthetic 2-D Gaussian clusters (240 points, 400 full-batch SGD epochs) and freeze it. You then train a 27-parameter student for 400 epochs on the teacher's temperature-2 class probabilities mixed with the true labels, and compare the student's objective, KL and agreement with the teacher before and after, with both models predicting at temperature 1.
- **Why this step here:** The earlier steps change one model with demonstrations, preferences or rewards. Here a trained model's whole output distribution becomes the training signal for a different, smaller model, the usual way to make a fine-tuned model cheaper to serve. It needs only softmax, cross-entropy and backpropagation through a small MLP; the script writes the gradients by hand and uses no autograd engine.
- **Run:** `python 02-alignment/microdistill.py`
- **Data:** None to download; the clusters are generated in the script.
- **Links:** [source](02-alignment/microdistill.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/distillation.md) · [primary paper](https://arxiv.org/abs/1503.02531) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdistill.gif)
- **Predict before you run:** (1) What parameter counts will the program print for the 2 → 16 → 3 teacher and the 2 → 4 → 3 student, biases included? (2) After the student trains, will the line `Teacher weight digest unchanged through transfer` print `True`, and what in the code guarantees it?
  <details><summary>Check your prediction</summary>

  (1) Teacher 2×16 + 16 + 16×3 + 3 = 99; student 2×4 + 4 + 4×3 + 3 = 27. (2) `True`. The soft targets are computed once from the trained teacher before student training starts, and `sgd_step` is only ever called on the student during transfer, so no teacher weight is written. The digest hashes every weight's exact 8-byte encoding, so even a change in the last bit would show. Both were checked on the script's own helpers with reduced budgets, not a default run. The counts came out as 99 and 27. After 40 student epochs the teacher digest was unchanged and the student's had changed, while adding 1e-12 to a single teacher weight changed the digest.
  </details>
- **Limits:** Synthetic, nearly separable 2-D clusters and two tiny MLPs, not the paper's MNIST, speech or ensemble experiments. T = 2 and α = 0.9 are toy choices. Teacher and student differ in capacity and no hard-label-only student is trained, so the output does not show that distillation itself improves the student. The 96 held-out points are reported only and never used for tuning or stopping.
- **Time:** 35 min
- [ ] Completed

---

## Track 3: Deep Dive — Modern Inference (~7 hrs)

Making models fast and small. This track covers every major inference optimization: efficient attention patterns, positional encoding, KV caching, memory management, quantization, decoding strategies, and state-space models. Every program here runs on a CPU in pure Python, so speed and memory figures are counts and illustrative timings, not GPU benchmarks.

**Prerequisites:** `01-foundations/microgpt.py` (transformer forward pass and attention mechanics).

### Steps

**1. `03-systems/microattention.py`**
- **Outcome:** You run single-head, multi-head (MHA), grouped-query (GQA), multi-query (MQA) and sliding-window attention forward on the same random input and compare an analytic FLOP and memory table with each output's cosine similarity to the MHA output.
- **Why this step here:** Every later attention optimisation in this track starts from these variants, and GQA/MQA explain why the KV cache of step 4 can be smaller than one key/value pair per head. It needs `microgpt.py`'s per-head attention loop.
- **Run:** `python 03-systems/microattention.py` (add `--interactive` to change the sequence length, width, head counts or window)
- **Data:** None to download; inputs and weights are random matrices drawn with seed 42.
- **Links:** [source](03-systems/microattention.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/transformer.md) · [primary paper](https://arxiv.org/abs/1706.03762) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microattention.gif)
- **Predict before you run:** With 4 query heads, 2 GQA key/value heads, a 32-position sequence and a window of 8, what reductions will the takeaways print for GQA's KV memory, MQA's KV memory and the sliding window's attention cost?
  <details><summary>Check your prediction</summary>

  GQA cuts KV memory 4 / 2 = 2x, MQA cuts it 4x (4 heads to 1), and the sliding window does 32 / 8 = 4x less attention-score work than full attention at this length. That 4x covers the score computation only: with the projections included, the table's FLOPs are 1,310,720 for MHA and 1,114,112 for the window, about 1.18x, and the untrained pure-Python timings do not measure throughput. The Memory column shows the KV saving: 2,048 floats of cached keys and values for GQA and 1,024 for MQA.
  </details>
- **Limits:** Nothing is trained, so the cosine similarities say how close untrained outputs are to the MHA output when the shared weights are averaged, not how well a trained GQA or MQA model performs. Timings are pure Python.
- **Time:** 40 min
- [ ] Completed

**2. `03-systems/microflash.py`**
- **Outcome:** You check that tiled attention with an online softmax matches standard attention to within 10⁻⁶ on five sequence-length/block-size configurations, then tabulate how many score floats each method holds at once.
- **Why this step here:** It keeps the attention result from step 1 exactly and changes only the order of computation, the idea behind memory-aware kernels. Standard attention from step 1 is the baseline it is checked against.
- **Run:** `python 03-systems/microflash.py`
- **Data:** None to download; random query, key and value matrices.
- **Links:** [source](03-systems/microflash.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/flash-attention.md) · [primary paper](https://arxiv.org/abs/2205.14135) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microflash.gif)
- **Predict before you run:** In the block-size table for N = 64, how many score floats does one tile hold and how many tiles are processed when B = 8? How many tiles does the (N = 37, B = 8) verification case need?
  <details><summary>Check your prediction</summary>

  B = 8 holds 8 × 8 = 64 floats per tile and processes ⌈64/8⌉² = 64 tiles. N = 37 with B = 8 needs ⌈37/8⌉² = 25 tiles, the last row and column of tiles being partial.
  </details>
- **Limits:** A simulation of the algorithm, not of the hardware: pure Python is slower than the standard version here, there is no fast on-chip memory, and "memory" counts only the score floats held at once (N² for standard, B² for one tile), not the running output and softmax statistics the tiled version also keeps.
- **Time:** 40 min
- [ ] Completed

**3. `03-systems/microrope.py`**
- **Outcome:** You rotate query and key pairs by position-dependent angles, show that RoPE scores for the same relative distance agree at different absolute positions while additive sinusoidal scores do not, and compare learned, sinusoidal, RoPE and NTK-scaled RoPE scores beyond the learned table's 64 positions.
- **Why this step here:** It changes how position enters the query–key dot product from step 1. Step 11 reuses the same 2×2 rotation inside a state-space model.
- **Run:** `python 03-systems/microrope.py`
- **Data:** None to download; random vectors.
- **Links:** [source](03-systems/microrope.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/rope.md) · [primary paper](https://arxiv.org/abs/2104.09864) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microrope.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/rope.md)
- **Predict before you run:** With head dimension 16 and base 10,000, pair `i` rotates by `θᵢ = 10000^(−2i/16)` per position. After how many positions does pair (0,1) complete a full turn, and pair (14,15)?
  <details><summary>Check your prediction</summary>

  Pair (0,1) has θ₀ = 1, so it turns once every 2π ≈ 6.3 positions. Pair (14,15) has θ₇ = 10000^(−14/16) ≈ 3.16 × 10⁻⁴, a wavelength of about 19,869 positions. The script prints both in its frequency-spectrum table.
  </details>
- **Limits:** The relative-position identity is a property of each query–key score. It does not guarantee that a trained model works at lengths it never saw; the extrapolation table compares untrained scores, and NTK scaling is a separate adjustment. The script also misdescribes that adjustment. `ntk_scaled_frequencies` raises the base to `10000 · s^(d/(d−2))`, so each frequency becomes `θ_i · s^(−2i/(d−2))`: with `d = 16` and `s = 4`, pair (0,1) is unchanged, the last pair is slowed by exactly 4×, and the pairs in between by 1.22× to 3.28×. The script's own explanations put the problem and the fix in the high-frequency pairs, and those statements are incorrect: the comments that long contexts fail because high-frequency pairs alias (lines 191-193) and that NTK scaling slows them more (line 204), the docstring "high-frequency pairs may alias … NTK scaling fixes this" (lines 264-265), and the printed lines "high-freq pairs rotate too fast" (304), "NTK scaling: slows high-freq rotations" (305), "Higher scale factors slow all frequencies proportionally" (346) and Key Takeaway 4, "adjusts the base frequency to prevent high-frequency aliasing" (405-406). NTK scaling leaves the fastest pair untouched and rescales the slow pairs; it does not guarantee that a trained model behaves well at longer lengths.
- **Time:** 35 min
- [ ] Completed

**4. `03-systems/microkv.py`**
- **Outcome:** You train a small model (300 steps), greedily generate 16 characters with and without a KV cache, confirm both produce the same tokens, count the multiplications each needs per step, and track cache size, then run a paged-allocation trace.
- **Why this step here:** It removes the repeated work in `microgpt.py`'s generation loop: without a cache every step recomputes keys and values for the whole prefix. Step 5 manages the memory this cache occupies.
- **Run:** `python 03-systems/microkv.py`
- **Data:** `names.txt`, downloaded on first run.
- **Links:** [source](03-systems/microkv.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/kv-cache.md) · [primary paper](https://arxiv.org/abs/2211.05102) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microkv.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/kv-cache.md)
- **Predict before you run:** The model has 16-wide embeddings, 2 heads, one layer and a 27-symbol vocabulary. How many multiplications do the two methods need at step 1 and at step 16, and how many floats does the cache hold after 16 positions?
  <details><summary>Check your prediction</summary>

  Step 1: 3,536 for both (one position to process either way). Step 16: 53,936 without the cache versus 4,016 with it, a 13.4x difference; over all 16 steps the ratio is 7.5x. The cache grows by 2 × 1 × 16 = 32 floats per position, reaching 512 floats (2,048 bytes as float32). Running the script's two generation functions with random weights gave exactly these counts and identical tokens; the counts do not depend on the weight values.
  </details>
- **Limits:** The match is exact because both paths use the same weights, the same causal attention and greedy decoding. Multi-query attention or a quantized cache would change the outputs and are different designs. The closing signpost's "~5.2 GB" for LLaMA-2 70B does not follow from the script's own per-position formula: 80 layers × 8,192 channels × 4,096 positions × 2 (K and V) × 2 bytes is about 10.7 GB without grouped-query attention, and about 1.3 GB with the 8 key/value heads LLaMA-2 70B actually uses.
- **Time:** 35 min
- [ ] Completed

**5. `03-systems/micropaged.py`**
- **Outcome:** You compare a naive allocator that reserves the maximum length for each request with a paged allocator (16 pages of 4 slots), check that paged attention matches contiguous attention, and walk through a serving timeline, copy-on-write for beam search, continuous batching and an internal-fragmentation table.
- **Why this step here:** It manages the cache from step 4 the way an operating system manages memory pages, which is what lets many variable-length requests share one memory budget.
- **Run:** `python 03-systems/micropaged.py`
- **Data:** None to download; random key and value vectors.
- **Links:** [source](03-systems/micropaged.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/pagedattention.md) · [primary paper](https://arxiv.org/abs/2309.06180) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/micropaged.gif)
- **Predict before you run:** With 4 slots per page and a naive reservation of 20 slots, what does the fragmentation table print for a 13-token sequence?
  <details><summary>Check your prediction</summary>

  `13   4    3  18.8%    7  35.0%`: 4 pages hold 16 slots, so 3 are wasted (3/16 = 18.8%), while the naive reservation wastes 20 − 13 = 7 slots (35.0%).
  </details>
- **Limits:** A simulation of allocation and bookkeeping; there are no GPU kernels or real memory, and the serving numbers come from a fixed toy workload of 8 requests.
- **Time:** 40 min
- [ ] Completed

**6. `03-systems/microquant.py`**
- **Outcome:** You train a model (800 steps), quantize every weight matrix with per-tensor absmax INT8 and INT4, zero-point INT8 and per-channel INT8, and compare loss, round-trip error, size and samples against the float baseline.
- **Why this step here:** Weights, like the KV cache, are memory. This step introduces the scale and zero-point arithmetic that QLoRA (Track 2) and TurboQuant (step 7) build on.
- **Run:** `python 03-systems/microquant.py`
- **Data:** `names.txt`, downloaded on first run.
- **Links:** [source](03-systems/microquant.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/llm-int8.md) · [primary paper](https://arxiv.org/abs/2208.07339) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microquant.gif)
- **Predict before you run:** The model has 4,192 weights. What sizes and compression ratios will the results table report for float32, INT8 and INT4?
  <details><summary>Check your prediction</summary>

  16,768 bytes, 4,192 bytes and 2,096 bytes, giving `float32->INT8 = 4.0x` and `float32->INT4 = 8.0x`. The size count ignores the stored scale factors.
  </details>
- **Limits:** Scales come from the weights themselves; no calibration data is used. The linked LLM.int8() paper's method — vector-wise quantization of activations and weights plus a 16-bit path for outlier features — is not implemented; this is weight-only round-to-nearest on a 4,192-parameter model.
- **Time:** 40 min
- [ ] Completed

**7. `03-systems/microturboquant.py`**
- **Outcome:** You sample one random rotation, quantize 32-dimensional unit vectors with per-vector absmax before and after rotating, compare inner-product error at 1, 2, 4 and 8 bits on anisotropic synthetic vectors and on name embeddings, and try a sign-bit (one bit per random projection) inner-product estimate.
- **Why this step here:** It applies the scalar quantizer from step 6 to vectors such as cached keys, and asks what a shared random rotation changes when no calibration data is available. It also uses the idea of a vector embedding from Track 1.
- **Run:** `python 03-systems/microturboquant.py`
- **Data:** `names.txt`, downloaded on first run; 300 names become embeddings by a random projection of their bigram counts, and 300 anisotropic vectors are synthetic.
- **Links:** [source](03-systems/microturboquant.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/turboquant.md) · [primary paper](https://arxiv.org/abs/2504.19874) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microturboquant.gif) · [optional lesson](https://github.com/no-magic-ai/no-magic-papers/blob/main/lessons/turboquant.md)
- **Predict before you run:** (1) `absmax_quantize` uses `2^(bits−1) − 1` levels per sign, with 1 level when `bits` is 1. What grids do the 1-bit and 2-bit rows use? (2) `qjl_estimate_inner_product` returns `(π/2) × mean(sign agreement)`. What does it return for two identical unit vectors?
  <details><summary>Check your prediction</summary>

  (1) Both use the same three values {−1, 0, +1}, so the two rows quantize every vector identically, and neither is a true 1-bit or 2-bit code. Their printed errors still differ slightly (0.0167 and 0.0173 for the synthetic baseline in one run) because each row draws a fresh random sample of 2,000 vector pairs. (2) π/2 ≈ 1.571 rather than 1. For Gaussian projections the expected sign agreement is 1 − 2·arccos(ρ)/π, so the estimate's expectation is arcsin(ρ), not the cosine ρ. At ρ = 0.5 the expected agreement is exactly 1/3 and the expected estimate is arcsin(0.5) = π/6 ≈ 0.5236, a bias of about +0.024; a finite draw scatters around that (one draw with 40,000 projections returned 0.526, and the standard deviation at that size is about 0.007). The estimator is biased except at ρ = 0.
  </details>
- **Limits:** The paper is Zandieh, Daliri, Hadian and Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874), as the linked card says. The script's header comment and the implementation guide (`docs/implementation.md`) attribute it to "Aamand et al." with the title "…with Optimal Bit Budget"; that attribution and title are incorrect. The script is a toy: it uses per-vector absmax instead of the paper's precomputed Lloyd–Max codebooks, and its sign demo applies paired signs to whole vectors. The paper's inner-product quantizer instead quantizes the residual left by the MSE quantizer and estimates inner products against an unquantized query, which makes that estimate unbiased; none of that is implemented here.
- **Time:** 45 min
- [ ] Completed

**8. `03-systems/microbeam.py`**
- **Outcome:** You train a target model (16-dimensional, 700 steps) and a smaller draft model (8-dimensional, 500 steps), then compare greedy, temperature, top-k, top-p, beam search and speculative decoding on the same prompts, measure diversity over 20 seed letters, and report the draft acceptance rate.
- **Why this step here:** Every program so far either sampled or took the most likely token; this step makes that choice the subject. It reuses the cached generation of step 4.
- **Run:** `python 03-systems/microbeam.py`
- **Data:** `names.txt`, downloaded on first run.
- **Links:** [source](03-systems/microbeam.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/nucleus-sampling.md) · [primary paper](https://arxiv.org/abs/1904.09751) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microbeam.gif)
- **Predict before you run:** `decode_top_p` adds tokens in order of probability until their total reaches `p`. With next-token probabilities 0.5, 0.3, 0.15 and 0.05 and `p = 0.9`, which tokens are kept and with what renormalised probabilities?
  <details><summary>Check your prediction</summary>

  The first three (0.5 + 0.3 + 0.15 = 0.95 ≥ 0.9), renormalised to about 0.526, 0.316 and 0.158; the 0.05 token can never be sampled at this step.
  </details>
- **Limits:** The speculative path picks each draft token by argmax rather than sampling it from the draft distribution, so the exact-distribution guarantee of Leviathan et al. does not carry over; read the acceptance rate as a demo statistic. Nothing runs in parallel, so there is no wall-clock speedup. The linked card covers nucleus sampling only.
- **Time:** 35 min
- [ ] Completed

**9. `03-systems/microssm.py`**
- **Outcome:** You build a one-layer selective state-space model (Euler discretization, input-dependent step size Δ and input-dependent B and C), train it for 800 steps on 250 names, sample names, and read an RNN/transformer/SSM comparison table.
- **Why this step here:** It replaces attention and its growing cache (step 4) with a fixed-size state updated by a linear recurrence, and makes that update depend on the input.
- **Run:** `python 03-systems/microssm.py`
- **Data:** `names.txt`, downloaded on first run; training uses 250 names.
- **Links:** [source](03-systems/microssm.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/mamba-2.md) · [primary paper](https://arxiv.org/abs/2405.21060) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microssm.gif)
- **Predict before you run:** Δ is `softplus(W x + b)` with every bias initialised to −2. What is Δ for a channel whose projected input is 0, and how many floats of state does the layer carry per token, however long the sequence?
  <details><summary>Check your prediction</summary>

  softplus(−2) = ln(1 + e⁻²) ≈ 0.127, a small step that mostly preserves the state. The state is 8 × 16 = 128 floats, printed as `SSM state size per layer: 128`, against a KV cache that grows with every position.
  </details>
- **Limits:** The source cites Mamba (Gu and Dao, 2023) and runs the recurrence sequentially; there is no parallel scan, it uses Euler rather than zero-order-hold discretization, and the linked Mamba-2 card's structured state space duality is not implemented. The comparison table states asymptotic costs; it is not a measurement.
- **Time:** 35 min
- [ ] Completed

**10. `03-systems/microdiscretize.py`**
- **Outcome:** You train the same small SSM three times — Euler, zero-order hold (ZOH) and trapezoidal discretization — on an irregular sine-prediction task and a running-parity task, then print a stability table of |Ā| against the step size Δ.
- **Why this step here:** Step 9 used Euler discretization; this step shows what that choice costs and what the exponential alternatives buy.
- **Run:** `python 03-systems/microdiscretize.py`
- **Data:** None to download; sine and parity sequences are generated in the script.
- **Links:** [source](03-systems/microdiscretize.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/mamba-2.md) · [primary paper](https://arxiv.org/abs/2405.21060) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdiscretize.gif)
- **Predict before you run:** The table uses a = −0.5. What does the row for Δ = 4.0 print for |Euler|, |ZOH| and the stability label?
  <details><summary>Check your prediction</summary>

  |Euler| = |1 + 4 × (−0.5)| = 1.0000, |ZOH| = |Trap| = e⁻² ≈ 0.1353, and the label is `NO — DIVERGES` because the script tests |Ā| < 1 strictly. At exactly 1 the state neither decays nor grows; Euler first grows the state at Δ = 5 (|Ā| = 1.5).
  </details>
- **Limits:** The source cites Mamba-3 (arXiv:2603.15569) Section 3 and S4; its trapezoidal rule splits the exact ZOH input term between `x_t` and `x_(t−1)` with a fixed weight. That is a toy, not Mamba-3's rule: Mamba-3 multiplies the previous input by `Δ·e^(ΔA)` and weights the two terms with a data-dependent λ, and its paper notes that Mamba-1 and Mamba-2, though described as ZOH, implement an exponential-Euler rule (ZOH's `exp(ΔA)` state factor with input term `Δ·B`). Mamba-3 drops the separate short convolution only in combination with added B and C biases, an empirical result this script does not reproduce. The linked Mamba-2 card's structured state space duality is not implemented. Task accuracies come from one seed.
- **Time:** 40 min
- [ ] Completed

**11. `03-systems/microcomplexssm.py`**
- **Outcome:** You show that a complex-valued diagonal SSM and a real SSM that rotates paired state dimensions with 2×2 matrices (a data-dependent RoPE) produce the same outputs to floating-point precision, then train real-only, complex and rotation variants on running parity.
- **Why this step here:** It joins the SSM of steps 9–10 with the rotation of step 3: a complex multiply is a scaled 2×2 rotation.
- **Run:** `python 03-systems/microcomplexssm.py`
- **Data:** None to download; random bit sequences with running-XOR labels.
- **Links:** [source](03-systems/microcomplexssm.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/mamba-2.md) · [primary paper](https://arxiv.org/abs/2405.21060) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microcomplexssm.gif)
- **Predict before you run:** Apply R(θ) with θ = π and r = 1 to the state pair (0.6, 0.0). What comes out, and why can the real-only variant (`a_n = exp(log_A_n)`) never do the same?
  <details><summary>Check your prediction</summary>

  (−0.6, ≈7 × 10⁻¹⁷): a half turn flips the sign, the same result as multiplying 0.6 by e^(iπ). The real-only variant's `a_n = exp(log_A_n)` is always positive, so it can shrink or grow the state but never flip its sign, and a running parity needs a flip on every 1-bit. A negative real eigenvalue could flip the sign; this parameterisation rules it out.
  </details>
- **Limits:** The source cites Mamba-3 (arXiv:2603.15569) Proposition 3 and RoPE (Su et al., 2021); the linked Mamba-2 card's structured state space duality is not implemented. The equivalence check uses fixed angles; the trained "complex" and "rotation" variants run the same real-arithmetic update from the same initialization distribution, with every angle starting near π (the parity solution), so differences between those two come only from their random draws. Parity accuracies come from one training run per variant.
- **Time:** 40 min
- [ ] Completed

**12. `03-systems/microroofline.py`**
- **Outcome:** You compute arithmetic intensity (FLOPs per byte) for a vector add, an outer product and two matrix multiplies, place pure-Python timings on an ASCII roofline built from assumed CPU peaks (50 GFLOPS, 100 GB/s), compare SISO and rank-16 MIMO SSM state updates, and train SISO and rank-4 MIMO SSMs on dual-sine prediction.
- **Why this step here:** It uses hardware arithmetic to show why writing the SSM update of steps 9–11 as a matrix multiply (MIMO) instead of an outer product (SISO) raises arithmetic intensity, which matters on accelerators even though it adds FLOPs.
- **Run:** `python 03-systems/microroofline.py`
- **Data:** None to download; sine sequences are generated in the script.
- **Links:** [source](03-systems/microroofline.py) · [paper card](https://github.com/no-magic-ai/no-magic-papers/blob/main/papers/roofline.md) · [primary paper](https://dl.acm.org/doi/10.1145/1498765.1498785) · [preview GIF](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microroofline.gif)
- **Predict before you run:** With 16 states and 8 channels, the SISO update costs 3 × 16 × 8 FLOPs and reads (16×8 + 16 + 8) float64 values per step; the rank-16 MIMO update costs 16×8 + 2 × 16 × 8 × 16 FLOPs and reads (16×8 + 16×16 + 16×8) values. What arithmetic intensities and FLOP ratio will Phase 3 print?
  <details><summary>Check your prediction</summary>

  SISO: 384 / 1,216 bytes ≈ 0.32 FLOPs/byte; MIMO-16: 4,224 / 4,096 bytes ≈ 1.03, with 11.0x the FLOPs. In the separate operation table the 16×16 outer product sits at 2.00 and the 256×256 rank-16 matrix multiply at 32.00; under the script's byte counts that larger figure comes from the larger matrix size, not from the rank.
  </details>
- **Limits:** The peak figures are assumed constants, not measured, and pure-Python timings are dominated by interpreter overhead, so the roofline placement is illustrative; nothing runs on a GPU. The source cites Mamba-3 (arXiv:2603.15569) for the MIMO update and Williams et al. (2009) for the roofline model.
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

38 of the 50 catalog scripts in dependency-respecting order, grouped by conceptual cluster with milestone markers. The remaining 12 are listed under [Scripts not yet in a track](#scripts-not-yet-in-a-track).

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

The exercises in this milestone now total 3 hrs; the heading estimate is kept unchanged so existing links to it keep working.

Teaching models to follow demonstrations and human preferences through supervised fine-tuning, preference optimization and reinforcement learning.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 16 | `02-alignment/microsft.py` | 35 min | - [ ] |
| 17 | `02-alignment/microdpo.py` | 40 min | - [ ] |
| 18 | `02-alignment/microreinforce.py` | 35 min | - [ ] |
| 19 | `02-alignment/microppo.py` | 35 min | - [ ] |
| 20 | `02-alignment/microgrpo.py` | 35 min | - [ ] |

### Milestone 9: Mixture of Experts (0.5 hrs)

Conditional computation — activating only a subset of parameters per input.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 21 | `02-alignment/micromoe.py` | 35 min | - [ ] |

### Milestone 10: Attention Optimization (2 hrs)

Efficient attention patterns, positional encoding, and memory-aware computation.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 22 | `03-systems/microattention.py` | 40 min | - [ ] |
| 23 | `03-systems/microflash.py` | 40 min | - [ ] |
| 24 | `03-systems/microrope.py` | 35 min | - [ ] |

### Milestone 11: Inference Systems (2.5 hrs)

The exercises in this milestone now total 3 hrs 50 min; the heading estimate is kept unchanged so existing links to it keep working.

KV caching, memory management, quantization, distillation into a smaller model, and decoding strategies.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 25 | `03-systems/microkv.py` | 35 min | - [ ] |
| 26 | `03-systems/micropaged.py` | 40 min | - [ ] |
| 27 | `03-systems/microquant.py` | 40 min | - [ ] |
| 28 | `03-systems/microturboquant.py` | 45 min | - [ ] |
| 29 | `02-alignment/microdistill.py` | 35 min | - [ ] |
| 30 | `03-systems/microbeam.py` | 35 min | - [ ] |

### Milestone 12: Advanced Systems (1.5 hrs)

State-space models, gradient checkpointing, and parallelism — the frontier of efficient training and inference.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 31 | `03-systems/microssm.py` | 35 min | - [ ] |
| 32 | `03-systems/microcheckpoint.py` | 30 min | - [ ] |
| 33 | `03-systems/microparallel.py` | 30 min | - [ ] |

### Milestone 13: Mamba-3 Deep Dive (2 hrs)

The SSM frontier — discretization methods, complex eigenvalue dynamics, and hardware-aware algorithm design from the Mamba-3 paper.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 34 | `03-systems/microdiscretize.py` | 40 min | - [ ] |
| 35 | `03-systems/microcomplexssm.py` | 40 min | - [ ] |
| 36 | `03-systems/microroofline.py` | 40 min | - [ ] |

### Milestone 14: Agent Algorithms (3 hrs)

How agents search and reason — tree search for planning and tool-augmented reasoning for language agents.

| # | Script | Time | Checkbox |
|---|--------|------|----------|
| 37 | `04-agents/micromcts.py` | 90 min | - [ ] |
| 38 | `04-agents/microreact.py` | 90 min | - [ ] |

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

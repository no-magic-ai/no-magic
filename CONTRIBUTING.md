# Contributing to no-magic

Thank you for your interest in contributing. This document is long because the bar is high. Every script in this repository is read by people trying to understand an algorithm for the first time — your code is their teacher. Please read this fully before submitting.

---

## The Non-Negotiable Constraints

These are not guidelines. They are hard requirements. PRs that violate any of these will be closed without review.

| Constraint | Rule |
|---|---|
| **One file** | Every script is a single `.py` file. No local imports, no `utils.py`, no companion files. |
| **Zero dependencies** | Python standard library only. If it needs `pip install`, it doesn't belong here. Allowed modules: `os`, `math`, `random`, `json`, `struct`, `urllib`, `collections`, `itertools`, `functools`, `string`, `hashlib`, `time`. |
| **Trains and infers** | Every script includes both the complete learning loop and inference/generation. The reader sees the full lifecycle. |
| **Runs in minutes** | Under **7 minutes on M-series Mac** or **10 minutes on 2019-era Intel i5**. No GPU required. |
| **Self-contained data** | Datasets are auto-downloaded on first run via `urllib` and cached locally. No manual download steps. Max 5MB. |
| **Reproducible** | `random.seed(42)` at the top of every script. Same input, same output. |
| **Commented** | Every script must follow the commenting standard described below. This is the single most common reason PRs are rejected. |

**If your PR adds a `requirements.txt`, it will be closed.**

---

## What We're Looking For

### New Scripts

We welcome new single-file implementations of algorithms that are widely used but poorly understood. Good candidates share these traits:

- The algorithm powers a significant part of modern AI/ML infrastructure.
- Most practitioners use it through library abstractions without understanding the internals.
- It can be meaningfully demonstrated on a small dataset in under 10 minutes on CPU.
- It doesn't already exist in the repository.

Before writing a new script, open an issue describing the algorithm, what it teaches, and why it belongs here. This saves you from investing time in something that doesn't fit the collection.

### Improvements to Existing Scripts

We welcome PRs that improve existing scripts in the following ways:

- **Better comments**: More clarity, better intuition, improved math-to-code mappings.
- **Bug fixes**: Incorrect implementations, numerical instability, or reproducibility issues.
- **Readability improvements**: Clearer variable names, better code structure, eliminating confusing patterns.
- **Performance within constraints**: Making a script run faster without adding dependencies or sacrificing readability. Readability always wins over performance.

### What We Don't Want

- Scripts that wrap or call external libraries (even "lightweight" ones like NumPy).
- "Improved" versions that add complexity without proportional clarity.
- Refactors that extract shared utilities into common modules — each script stands alone.
- Notebooks, blog posts, or documentation-only PRs (open an issue to discuss these).
- Scripts that only demonstrate forward passes without training (with the documented exception of comparison scripts like `microattention.py`).

---

## The Commenting Standard

This is where most first-time contributors need to adjust. We are **not** optimizing for line count or code elegance. We are optimizing for a reader's understanding. A motivated engineer should be able to open your script and read it top-to-bottom — like a guided walkthrough — without needing a paper, textbook, or external reference.

*Fewer lines is not a goal. Fewer moments of confusion is.*

### Required Comment Types

Your script must include all of the following:

**1. File Thesis**

The first line is a docstring stating what the script proves in one sentence. This is the script's reason for existing.

```python
"""
The complete BPE tokenization algorithm: learning merges from a corpus, encoding text to
tokens, and decoding tokens back to text — in pure Python with zero dependencies.
"""
```

**2. Section Headers**

Block comments separating major phases. A reader skimming only these headers should understand the script's structure without reading any code.

```python
# === DATA LOADING ===
# Fetch and prepare the training corpus.

# === MODEL DEFINITION ===
# Define the encoder and decoder networks.

# === TRAINING LOOP ===
# Train the model using Adam with linear LR decay.

# === INFERENCE ===
# Generate new samples from the trained model.
```

**3. "Why" Comments**

Explain the reasoning behind non-obvious decisions. The code shows *what* happens. Your comment explains *why* it happens this way and what would break or degrade if it were different.

```python
# We use RMSNorm instead of LayerNorm: it drops the mean subtraction and learned
# affine parameters. Fewer ops, fewer params, and modern architectures (LLaMA, Gemma)
# have shown it works just as well.
```

**4. Math-to-Code Mappings**

Where the code implements a known equation, show the equation and map the variables explicitly. The reader should never have to guess which variable corresponds to which symbol in the paper.

```python
# Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
# This lets gradients flow through the sampling operation because
# the randomness (epsilon) is external to the computation graph.
epsilon = random.gauss(0, 1)
z = mu + exp(0.5 * log_var) * epsilon
```

**5. Intuition Comments**

Brief explanations of *why* a technique works, not just how. These are the comments that save someone a trip to Wikipedia. If you understand something deeply enough to explain it in two sentences, that's worth more than a link to a paper.

```python
# The update gate z_t acts as a "gradient highway": when z_t ≈ 1, the new hidden
# state is just a copy of the previous one (h_t = h_{t-1}), so gradients flow
# backward through time without multiplication by weight matrices. This is why
# GRUs don't suffer from vanishing gradients the way vanilla RNNs do.
```

**6. Signpost Comments**

Where the implementation makes a simplifying choice for pedagogical reasons, flag it and note what production systems do differently. This prevents readers from mistaking a toy pattern for a best practice.

```python
# In production, you'd use Generalized Advantage Estimation (GAE) here for lower
# variance. We use simple (reward - baseline) to keep the PPO core visible without
# the GAE machinery obscuring it.
```

**7. No Obvious Comments**

If the code is self-explanatory, let it speak. Every comment must earn its place by adding information the code alone does not convey.

```python
# BAD — restates the code
x = x + 1  # increment x by 1

# BAD — restates the code with more words
total = sum(losses)  # sum up all the losses

# GOOD — explains why
total = sum(losses) / len(losses)  # average over sequence length, not sum, to make
                                    # the loss scale-invariant to document length
```

### Comment Density

Target roughly 30-40% of lines as comments or blank lines. This is not a hard metric — dense math sections need more commentary, data loading boilerplate needs less. The actual test is subjective but strict: *could a motivated engineer with general ML knowledge read this file top-to-bottom in one sitting and understand the algorithm?*

---

## Code Style

### General Principles

- **Readability over cleverness.** If a three-line explicit loop is clearer than a one-line comprehension, use the loop.
- **Flat over nested.** Avoid deep nesting. Extract helper functions with descriptive names.
- **Descriptive names.** Variables should be named for what they represent, not abbreviated for brevity. Use `learning_rate` not `lr`, `hidden_dim` not `hd`. Exceptions: conventional math notation (`x`, `z`, `mu`, `sigma`), loop indices (`i`, `j`, `t`), and universally understood abbreviations in ML (`embd`, `attn`, `mlp`).
- **Functions describe what, not how.** Name functions for *what* they compute: `rmsnorm`, `softmax`, `linear`. Not `normalize_v2` or `process_data`.
- **Consistent structure.** Follow the section ordering: imports → constants/hyperparameters → data loading → model definition → training loop → inference.

### Formatting

- Use 4-space indentation.
- Maximum line length of 100 characters. Break long lines for readability.
- One blank line between logical blocks within a function. Two blank lines between top-level definitions.
- No trailing whitespace.
- End the file with a single newline.

### Python Practices

- Use type hints in function signatures where they aid readability, but don't over-annotate obvious types.
- Prefer `f-strings` for formatted output.
- Use `random.seed(42)` at the top, before any randomness.
- Avoid global mutable state where possible. If globals are necessary (as in `microgpt.py`'s `state_dict`), comment why.
- No classes unless the algorithm fundamentally requires them (e.g., `Value` for autograd). Prefer functions and plain data structures.

---

## Submitting a Pull Request

### Before You Start

1. **Check existing issues and the implementation plan.** The `implementation.md` file documents all scripts with detailed specs.
2. **For new algorithm ideas**, open an issue first. Describe the algorithm, what it teaches, the dataset you'd use, and the expected line count. Wait for approval before writing code.
3. **For improvements to existing scripts**, open an issue describing what you'd change and why. Small fixes (typos, numerical bugs) can go straight to a PR.

### Writing the Script

1. Start from the spec in `implementation.md` if one exists for your algorithm.
2. Write the complete script in a single file.
3. Run it end-to-end on your machine. Verify it completes in under 10 minutes.
4. Run it again with a fresh directory (delete any cached data files) to verify the auto-download works.
5. Review your own comments against the commenting standard above. Be honest — would *you* understand this code if you were seeing the algorithm for the first time?

### PR Requirements

Your pull request must include:

- **The script file** placed in the correct tier directory (`01-foundations/`, `02-alignment/`, or `03-systems/`).
- **A PR description** that includes:
  - The algorithm name and a one-sentence summary.
  - The dataset used and how it's fetched.
  - The total line count and approximate comment density.
  - The runtime on your machine (CPU model + time).
  - Sample output (copy-paste a few lines of training progress and inference results).

Your pull request must **not** include:

- Changes to other scripts (unless fixing a cross-cutting bug).
- New directories outside the established structure.
- Any file other than the single `.py` script (no READMEs per script, no notebooks, no test files).
- Changes to `CONTRIBUTING.md` (open an issue to discuss these).

### Review Process

1. **Automated checks**: The script must run with `python script.py` and exit cleanly. No arguments, no environment variables, no manual setup.
2. **Constraint verification**: One file, zero dependencies, under 10 minutes, reproducible output.
3. **Commenting review**: This is the most thorough part of review. Expect feedback on comment quality, missing intuition explanations, and unclear math mappings. This is not nitpicking — it's the core quality bar of the project.
4. **Correctness review**: The algorithm must be implemented correctly. Simplified, yes. Wrong, no. If your implementation deviates from the standard algorithm, the deviation must be commented with a signpost explaining why.
5. **Readability review**: The "one sitting" test. A reviewer will read your script top-to-bottom and note every point where they had to stop and think. If there are too many, you'll be asked to add comments or restructure.

Expect at least one round of revision. This is normal and not a reflection of code quality — it's the nature of writing for an educational audience.

---

## Attribution and Licensing

- All contributions are made under the MIT license.
- If your script is inspired by a specific paper, blog post, or existing implementation, cite it in a comment at the top of the file, immediately after the thesis docstring.
- Do not copy code from other repositories, even MIT-licensed ones. Write your own implementation. You may reference other implementations for correctness verification, but the code must be original.
- If your script covers similar ground to Karpathy's work (micrograd, makemore, microgpt), ensure it provides distinct pedagogical value rather than replicating what already exists. See `implementation.md` for how we handle this.

```python
"""
Direct Preference Optimization: training a language model to align with human preferences
without a separate reward model — in pure Python with zero dependencies.
"""
# Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is
# Secretly a Reward Model" (2023). https://arxiv.org/abs/2305.18290
# Inspired by the DPO implementation in TRL, but rewritten from scratch for clarity.
```

---

## Code of Conduct

- Be respectful in issues and reviews.
- Assume good intent. Contributors come from different backgrounds and experience levels.
- Critique code and comments, not people.
- If you disagree with a review decision, explain your reasoning. If the maintainer still disagrees, accept the decision gracefully.

---

## Quick Reference Checklist

Before submitting, verify every item:

**Execution**
- [ ] `python script.py` runs with zero arguments and exits cleanly
- [ ] No imports outside Python standard library
- [ ] `random.seed(42)` at the top
- [ ] Completes in under 7 minutes on M-series Mac (or 10 minutes on 2019 Intel i5)
- [ ] Prints training progress (step number, loss)
- [ ] Prints inference results demonstrating the trained model
- [ ] Meets the success criteria defined in `docs/implementation.md` for this script

**Autograd & Numerical Stability** (for scripts using scalar autograd)
- [ ] `Value` class implements the canonical interface from `docs/autograd-interface.md`
- [ ] Autograd callout block present after Value class (documents per-script differences)
- [ ] Stable softmax: `exp(x - max(x))` pattern with explanatory comment
- [ ] Clipped log-probabilities: `max(p, 1e-10)` before `log()` with comment
- [ ] Adam epsilon: `1e-8` in denominator with comment
- [ ] Test vectors pass (from `docs/autograd-interface.md`)

**Commenting (non-negotiable)**
- [ ] File opens with a one-sentence thesis docstring
- [ ] Section headers (`# === SECTION ===`) separate major phases
- [ ] Every non-obvious block has a "why" comment
- [ ] Key equations have math-to-code mapping comments
- [ ] At least one intuition comment per core algorithmic concept
- [ ] Simplifying choices flagged with signpost comments
- [ ] No obvious or redundant comments
- [ ] Comment density approximately 30-40%

**Readability**
- [ ] Passes the "one sitting" test
- [ ] Variable names are descriptive and consistent
- [ ] Functions named for what they compute
- [ ] No unnecessary complexity or cleverness

**Logistics**
- [ ] File placed in correct tier directory (`01-foundations/`, `02-alignment/`, or `03-systems/`)
- [ ] PR description includes runtime, line count, and sample output
- [ ] No extra files included
- [ ] Attribution comments for any referenced papers or implementations

---

*The constraint is the product. The comments are the curriculum. Thank you for helping build it.*

---

## Translating Scripts

We welcome translations of script comments into other languages. The code stays in English — only comments, docstrings, and inline documentation are translated.

### How to Translate

1. Pick a script and a target language from the [translation status table](TRANSLATIONS.md).
2. Copy the script to `translations/<locale>/` (e.g., `translations/es/microgpt.py`).
3. Translate all comments, docstrings, section headers, and print statements.
4. Do NOT translate variable names, function names, or code.
5. Preserve all 7 comment types from the commenting standard.
6. Test that the translated script still runs: `python translations/es/microgpt.py`

### Translation Quality Bar

- Technical accuracy over literary polish — an incorrect translation is worse than an awkward one.
- Preserve math notation as-is (equations are universal).
- Use domain-standard terminology for the target language (e.g., the accepted ML term in that language's community).
- When in doubt, keep the English term with a parenthetical translation.

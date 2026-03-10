"""
Speculative decoding from first principles: a small draft model proposes tokens that a
larger verifier accepts or rejects — achieving the quality of the large model at nearly
the speed of the small one.
"""
# Reference: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
# (2023). Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"

# === TRADEOFFS ===
# + Generates tokens at near-draft-model speed with verifier-model quality
# + Mathematically guaranteed to produce the same distribution as the verifier alone
# + No retraining needed — works with any compatible draft/verifier pair
# - Requires a high-quality draft model (acceptance rate depends on draft-verifier alignment)
# - Overhead per step: must run both models (wasteful when acceptance rate is low)
# - Diminishing returns with longer speculation windows (later tokens less likely to match)
# WHEN TO USE: Inference serving where latency matters and you have a good
#   small model aligned with the large model (e.g., same architecture, different scale).
# WHEN NOT TO: When no suitable draft model exists, when batch throughput
#   matters more than per-request latency, or with very short sequences.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Draft model: small, fast, lower quality (~500 params)
DRAFT_N_EMBD = 8
DRAFT_N_HEAD = 2
DRAFT_N_LAYER = 1
DRAFT_HEAD_DIM = DRAFT_N_EMBD // DRAFT_N_HEAD  # 4

# Verifier model: larger, slower, higher quality (~2000 params)
VERIFIER_N_EMBD = 16
VERIFIER_N_HEAD = 4
VERIFIER_N_LAYER = 1
VERIFIER_HEAD_DIM = VERIFIER_N_EMBD // VERIFIER_N_HEAD  # 4

BLOCK_SIZE = 16  # context window — shared between both models

# Speculative decoding parameters
SPEC_K = 4  # number of draft tokens to propose per speculation round

# Training
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
DRAFT_STEPS = 500
VERIFIER_STEPS = 500

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: production speculative decoding pairs models like Llama-70B (verifier) with
# Llama-7B (draft). Our ~500/~2000 param ratio is far smaller but preserves the full
# algorithmic structure. The acceptance rate — not the wall-clock speedup — is the
# hardware-independent metric that validates the implementation.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput), then backward() replays the graph in
    reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * (other ** -1)
    def __rtruediv__(self, other: float) -> Value: return other * (self ** -1)

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self) -> Value:
        # ReLU: max(0, x). Gradient is 1 for positive, 0 otherwise.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort then chain rule."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# Autograd is used only for training both models. Speculative decoding inference
# uses plain-float forward passes for speed.
# See docs/autograd-interface.md for the full specification.


# === TRAINING HELPERS (Value-based) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    std=0.08 is empirical for this toy scale. Production models use
    std = 1/sqrt(d_in) (Xavier/Glorot) to control activation variance.
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def linear_v(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x. The fundamental neural network operation."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax_v(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow.

    Math: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    Translation-invariant: subtracting max changes nothing mathematically but
    keeps exp() arguments near zero, avoiding float overflow.
    """
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm_v(x: list[Value]) -> list[Value]:
    """RMSNorm: scale to unit root-mean-square magnitude.

    Math: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    Simpler than LayerNorm (no mean centering, no learned affine), used in
    LLaMA, Gemma, and other recent architectures.
    """
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped log: clamp to 1e-10 to prevent log(0).

    Keeps prob as a child node so gradients flow back through the computation
    graph — creating a disconnected Value(clamped) would sever the gradient path.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def gpt_forward_train(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
    n_embd: int, n_head: int, n_layer: int, head_dim: int,
) -> list[Value]:
    """Single-token GPT forward pass for training, parameterized by model config.

    Both draft and verifier share this function — they differ only in the
    dimensions passed (n_embd, n_head, etc.) and the weight matrices in params.
    This is the key insight: speculative decoding works with any compatible pair
    of models sharing the same vocabulary and architecture family.
    """
    # Embedding: token identity + positional information
    x = [t + p for t, p in zip(params['wte'][token_id], params['wpe'][pos_id])]
    x = rmsnorm_v(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm_v(x)

        # Multi-head self-attention with incremental KV cache
        q = linear_v(x, params[f'l{li}.wq'])
        k = linear_v(x, params[f'l{li}.wk'])
        v = linear_v(x, params[f'l{li}.wv'])
        keys[li].append(k)
        values[li].append(v)

        # Each head attends independently over different subspaces
        x_attn: list[Value] = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [kt[hs:hs + head_dim] for kt in keys[li]]
            v_h = [vt[hs:hs + head_dim] for vt in values[li]]
            # Scaled dot-product attention: score = (q . k) / sqrt(d_head)
            # The sqrt scaling prevents softmax saturation as dimensionality grows
            scores = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                for t in range(len(k_h))
            ]
            w = softmax_v(scores)
            # Weighted sum of values — the "attention" mechanism
            x_attn.extend([
                sum(w[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ])

        x = linear_v(x_attn, params[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]  # residual connection
        x_res = x

        # MLP: expand to 4x, nonlinearity, project back
        x = rmsnorm_v(x)
        x = linear_v(x, params[f'l{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear_v(x, params[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]  # residual connection

    return linear_v(x, params['lm_head'])


# === PLAIN-FLOAT INFERENCE ===
# After training, weights are extracted as plain floats. Speculative decoding
# operates entirely here — no autograd overhead during generation.

def extract(w: list[list[Value]]) -> list[list[float]]:
    """Strip autograd wrappers from a weight matrix."""
    return [[v.data for v in row] for row in w]


def linear_f(x: list[float], w: list[list[float]]) -> list[float]:
    """Plain-float matrix-vector multiply."""
    return [sum(w[i][j] * x[j] for j in range(len(x))) for i in range(len(w))]


def softmax_f(logits: list[float]) -> list[float]:
    """Plain-float numerically stable softmax."""
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def rmsnorm_f(x: list[float]) -> list[float]:
    """Plain-float RMSNorm."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    return [xi * (mean_sq + 1e-5) ** -0.5 for xi in x]


# Cfg holds model dimensions — both draft and verifier use the same forward_float
# function, distinguished only by their config and weights.
Cfg = dict


def forward_float(
    tok: int, pos: int,
    kv: list[dict[str, list[list[float]]]],
    wf: dict[str, list[list[float]]],
    c: Cfg,
) -> list[float]:
    """GPT forward pass using plain floats. Processes one token, appends to KV
    cache, returns logits. Used by both models during speculative decoding."""
    x = [wf['wte'][tok][j] + wf['wpe'][pos][j] for j in range(c['n_embd'])]
    x = rmsnorm_f(x)
    for li in range(c['n_layer']):
        x_res = x[:]
        x = rmsnorm_f(x)
        q = linear_f(x, wf[f'l{li}.wq'])
        k = linear_f(x, wf[f'l{li}.wk'])
        v = linear_f(x, wf[f'l{li}.wv'])
        kv[li]['k'].append(k)
        kv[li]['v'].append(v)
        head_cat: list[float] = []
        clen = len(kv[li]['k'])
        hd = c['head_dim']
        for h in range(c['n_head']):
            hs = h * hd
            q_h = q[hs:hs + hd]
            scores = [
                sum(q_h[j] * kv[li]['k'][t][hs + j] for j in range(hd)) / (hd ** 0.5)
                for t in range(clen)
            ]
            w = softmax_f(scores)
            for j in range(hd):
                head_cat.append(sum(w[t] * kv[li]['v'][t][hs + j] for t in range(clen)))
        x = linear_f(head_cat, wf[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x[:]
        x = rmsnorm_f(x)
        x = linear_f(x, wf[f'l{li}.fc1'])
        x = [max(0.0, v) for v in x]
        x = linear_f(x, wf[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_f(x, wf['lm_head'])


def make_kv(c: Cfg) -> list[dict[str, list[list[float]]]]:
    """Fresh empty KV cache for a model config."""
    return [{'k': [], 'v': []} for _ in range(c['n_layer'])]


def clone_kv(cache: list[dict[str, list[list[float]]]]) -> list[dict[str, list[list[float]]]]:
    """Deep copy KV cache — essential so speculative rollbacks don't corrupt state.

    During speculation, we tentatively advance the KV cache. If the verifier
    rejects a draft token, we must roll back to the pre-speculation state.
    Deep copying before speculation makes this rollback trivial.
    """
    return [{'k': [r[:] for r in l['k']], 'v': [r[:] for r in l['v']]} for l in cache]


def feed_prompt(
    toks: list[int],
    wf: dict[str, list[list[float]]],
    c: Cfg,
) -> tuple[list[dict[str, list[list[float]]]], list[float]]:
    """Feed a prompt through the model, returning (kv_cache, last_logits).

    Both models must process the same prompt so their KV caches are aligned
    at the start of speculative decoding.
    """
    kv = make_kv(c)
    logits: list[float] = []
    for i, t in enumerate(toks):
        logits = forward_float(t, i, kv, wf, c)
    return kv, logits


# === MODEL INIT AND TRAINING ===

def init_params(
    vocab_size: int, n_embd: int, n_head: int, n_layer: int,
) -> dict[str, list[list[Value]]]:
    """Initialize all GPT parameters for the given dimensions."""
    p: dict[str, list[list[Value]]] = {}
    p['wte'] = make_matrix(vocab_size, n_embd)
    p['wpe'] = make_matrix(BLOCK_SIZE, n_embd)
    for li in range(n_layer):
        p[f'l{li}.wq'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wk'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wv'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wo'] = make_matrix(n_embd, n_embd)
        # 4x expansion in MLP is the standard GPT convention — gives the feedforward
        # network more capacity without widening the residual stream
        p[f'l{li}.fc1'] = make_matrix(4 * n_embd, n_embd)
        p[f'l{li}.fc2'] = make_matrix(n_embd, 4 * n_embd)
    p['lm_head'] = make_matrix(vocab_size, n_embd)
    return p


def count_params(params: dict[str, list[list[Value]]]) -> int:
    """Count total scalar parameters in a model."""
    return sum(len(row) for matrix in params.values() for row in matrix)


def train_model(
    docs: list[str], chars: list[str], bos: int, vocab_size: int,
    params: dict[str, list[list[Value]]],
    n_embd: int, n_head: int, n_layer: int, head_dim: int,
    num_steps: int,
) -> None:
    """Train a GPT model with Adam optimizer and linear LR decay.

    Identical training procedure for both draft and verifier — they see the
    same data in the same order (via seeded shuffle). The verifier simply has
    more parameters to fit the distribution, which is why its loss should be
    lower. This alignment is what makes speculative decoding effective: the
    draft approximates the verifier because both learned from the same signal.
    """
    plist = [p for w in params.values() for row in w for p in row]
    m_s = [0.0] * len(plist)
    v_s = [0.0] * len(plist)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        toks = [bos] + [chars.index(ch) for ch in doc] + [bos]
        sl = min(BLOCK_SIZE, len(toks) - 1)
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        losses: list[Value] = []
        for pos in range(sl):
            logits = gpt_forward_train(toks[pos], pos, keys, vals, params,
                                       n_embd, n_head, n_layer, head_dim)
            probs = softmax_v(logits)
            losses.append(-safe_log(probs[toks[pos + 1]]))
        loss = (1.0 / sl) * sum(losses)
        loss.backward()

        # Adam with linear LR decay
        lr_t = LEARNING_RATE * (1 - step / num_steps)
        for i, p in enumerate(plist):
            m_s[i] = BETA1 * m_s[i] + (1 - BETA1) * p.grad
            v_s[i] = BETA2 * v_s[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_s[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_s[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{num_steps} | loss: {loss.data:.4f}")

    print(f"  Final loss: {loss.data:.4f}\n")


# === DECODING STRATEGIES ===


def decode_verifier_only(
    prompt: list[int],
    wf: dict[str, list[list[float]]],
    c: Cfg,
    max_len: int = 12,
    temperature: float = 0.8,
) -> tuple[list[int], float, int]:
    """Standard autoregressive decoding: one token per forward pass.

    This is the baseline. Every token requires a full verifier forward pass.
    Speculative decoding aims to produce the same output distribution while
    calling the verifier fewer times.

    Returns: (generated_tokens, log_prob, num_forward_passes)
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    fwd_count = len(prompt)  # prompt tokens required forward passes too

    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE:
            break
        probs = softmax_f([l / temperature for l in logits])
        tok = random.choices(range(c['vocab_size']), weights=probs)[0]
        if tok == c['bos']:
            break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
        fwd_count += 1

    return gen, lp, fwd_count


def decode_speculative(
    prompt: list[int],
    verifier_wf: dict[str, list[list[float]]],
    draft_wf: dict[str, list[list[float]]],
    vc: Cfg,
    dc: Cfg,
    max_len: int = 12,
    draft_k: int = SPEC_K,
    temperature: float = 0.8,
) -> tuple[list[int], float, int, int, int, int]:
    """Speculative decoding: draft proposes, verifier accepts or rejects.

    The algorithm (Leviathan et al., 2023):
    1. Draft model autoregressively generates K candidate tokens
    2. Verifier scores each candidate against its own distribution
    3. Accept token i with probability min(1, p_verifier(x_i) / p_draft(x_i))
    4. On rejection at position i: resample from max(0, p_verifier - p_draft),
       discard all tokens after position i
    5. If all K tokens accepted: sample one bonus token from the verifier

    WHY THIS IS LOSSLESS (proof sketch):
    The acceptance criterion is a form of rejection sampling. For any token x:
    - If p_v(x) >= p_d(x): always accept — the verifier wants this token at least
      as much as the draft, so accepting preserves the verifier's preference
    - If p_v(x) < p_d(x): accept with probability p_v(x)/p_d(x) — this downweights
      tokens the draft over-produces relative to the verifier
    - On rejection: the residual distribution max(0, p_v - p_d) / Z captures exactly
      the probability mass the verifier assigns beyond what the draft already covered
    The combined accept + reject-resample distribution equals p_verifier exactly.

    This is Metropolis-Hastings style reasoning: the draft is the proposal
    distribution, and the acceptance ratio corrects for proposal bias.

    Returns: (tokens, log_prob, verifier_fwd_passes, draft_fwd_passes,
              total_proposed, total_accepted)
    """
    v_kv, v_logits = feed_prompt(prompt, verifier_wf, vc)
    d_kv, d_logits = feed_prompt(prompt, draft_wf, dc)
    gen: list[int] = []
    lp = 0.0
    v_fwd = len(prompt)
    d_fwd = len(prompt)
    total_proposed = 0
    total_accepted = 0  # only counts draft tokens that passed the acceptance criterion

    while len(gen) < max_len:
        cur_pos = len(prompt) + len(gen)
        remaining = min(draft_k, max_len - len(gen))
        if cur_pos >= BLOCK_SIZE or remaining <= 0:
            break

        # --- Phase 1: Draft model proposes K tokens ---
        # The draft runs autoregressively (cheap forward passes) to build
        # a speculative continuation. We save each step's probability distribution
        # because the verifier needs p_draft(x) for the acceptance criterion.
        draft_toks: list[int] = []
        draft_probs_list: list[list[float]] = []
        tmp_d_kv = clone_kv(d_kv)
        tmp_d_logits = d_logits[:]

        for di in range(remaining):
            pos = cur_pos + di
            if pos >= BLOCK_SIZE:
                break
            dp = softmax_f([l / temperature for l in tmp_d_logits])
            draft_probs_list.append(dp)
            # Sample from the draft distribution (not greedy — we need stochastic
            # proposals to align with the verifier's stochastic generation)
            dtok = random.choices(range(dc['vocab_size']), weights=dp)[0]
            if dtok == dc['bos']:
                break
            draft_toks.append(dtok)
            tmp_d_logits = forward_float(dtok, pos, tmp_d_kv, draft_wf, dc)
            d_fwd += 1

        if not draft_toks:
            # Draft immediately produced BOS — fall back to one verifier step
            vp = softmax_f([l / temperature for l in v_logits])
            vtok = random.choices(range(vc['vocab_size']), weights=vp)[0]
            if vtok == vc['bos']:
                break
            lp += math.log(max(vp[vtok], 1e-10))
            gen.append(vtok)
            v_logits = forward_float(vtok, cur_pos, v_kv, verifier_wf, vc)
            d_logits = forward_float(vtok, cur_pos, d_kv, draft_wf, dc)
            v_fwd += 1
            d_fwd += 1
            continue

        total_proposed += len(draft_toks)

        # --- Phase 2: Verifier scores each draft token ---
        # On a GPU, this would be a single batched forward pass over all K
        # positions. In our scalar implementation we run sequentially, but the
        # acceptance logic is identical to the parallel version.
        accepted: list[int] = []
        draft_accepted_count = 0  # how many draft tokens passed acceptance
        tmp_v_kv = clone_kv(v_kv)
        tmp_v_logits = v_logits[:]

        for vi in range(len(draft_toks)):
            vp = softmax_f([l / temperature for l in tmp_v_logits])
            dp = draft_probs_list[vi]
            dtok = draft_toks[vi]

            # Core acceptance criterion: min(1, p_verifier / p_draft)
            # This is the probability of accepting the draft token. When the
            # verifier assigns MORE probability than the draft, we always accept.
            # When less, we accept proportionally — rejecting tokens the draft
            # over-represents relative to the verifier.
            p_v = vp[dtok]
            p_d = max(dp[dtok], 1e-10)  # clamp to avoid division by zero
            acceptance_prob = min(1.0, p_v / p_d)

            if random.random() < acceptance_prob:
                # Accepted: this draft token aligns with the verifier's distribution
                accepted.append(dtok)
                draft_accepted_count += 1
                lp += math.log(max(vp[dtok], 1e-10))
                tmp_v_logits = forward_float(dtok, cur_pos + vi, tmp_v_kv, verifier_wf, vc)
                v_fwd += 1
            else:
                # Rejected: resample from the residual distribution
                # max(0, p_verifier - p_draft) captures the probability mass the
                # verifier assigns beyond the draft's allocation. Normalizing this
                # gives the correction distribution that, combined with the acceptance
                # events, recovers p_verifier exactly.
                adj = [max(0.0, vp[j] - dp[j]) for j in range(len(vp))]
                adj_sum = sum(adj)
                if adj_sum > 0:
                    adj = [a / adj_sum for a in adj]
                    rtok = random.choices(range(vc['vocab_size']), weights=adj)[0]
                else:
                    # Degenerate case: verifier <= draft everywhere (shouldn't happen
                    # in practice but handle gracefully by sampling from verifier)
                    rtok = random.choices(range(vc['vocab_size']), weights=vp)[0]
                if rtok != vc['bos']:
                    accepted.append(rtok)
                    lp += math.log(max(vp[rtok], 1e-10))
                    # Run verifier on the resampled token to update KV cache
                    forward_float(rtok, cur_pos + vi, tmp_v_kv, verifier_wf, vc)
                    v_fwd += 1
                # Discard all remaining draft tokens after rejection — they were
                # conditioned on the rejected token and are no longer valid
                break
        else:
            # All K draft tokens accepted — bonus: sample one more from verifier
            # This is free because we already have tmp_v_logits from the last
            # verification step. It's what makes speculative decoding produce
            # K+1 tokens per round in the best case.
            bonus_vp = softmax_f([l / temperature for l in tmp_v_logits])
            bonus_tok = random.choices(range(vc['vocab_size']), weights=bonus_vp)[0]
            if bonus_tok != vc['bos']:
                accepted.append(bonus_tok)
                lp += math.log(max(bonus_vp[bonus_tok], 1e-10))
                forward_float(bonus_tok, cur_pos + len(draft_toks), tmp_v_kv, verifier_wf, vc)
                v_fwd += 1

        total_accepted += draft_accepted_count

        # Commit accepted tokens to both real KV caches
        for ai, atok in enumerate(accepted):
            v_logits = forward_float(atok, cur_pos + ai, v_kv, verifier_wf, vc)
            d_logits = forward_float(atok, cur_pos + ai, d_kv, draft_wf, dc)
            v_fwd += 1
            d_fwd += 1
            gen.append(atok)

        # If nothing was accepted and no resampled token either, the while loop
        # would spin. This shouldn't happen because rejection always resamples,
        # but guard against edge cases.
        if not accepted:
            vp = softmax_f([l / temperature for l in v_logits])
            vtok = random.choices(range(vc['vocab_size']), weights=vp)[0]
            if vtok == vc['bos']:
                break
            lp += math.log(max(vp[vtok], 1e-10))
            gen.append(vtok)
            v_logits = forward_float(vtok, cur_pos, v_kv, verifier_wf, vc)
            d_logits = forward_float(vtok, cur_pos, d_kv, draft_wf, dc)
            v_fwd += 1
            d_fwd += 1

    return gen, lp, v_fwd, d_fwd, total_proposed, total_accepted


# === MAIN ===

if __name__ == "__main__":
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Loaded {len(docs)} documents, vocab size: {VOCAB_SIZE}\n")

    # === TRAINING PHASE ===
    # Train both models on the same data. The verifier has more capacity (~2000
    # params vs ~500) so it fits the distribution better. The draft model's job
    # is to approximate the verifier cheaply — the closer the approximation,
    # the higher the acceptance rate during speculative decoding.

    print(f"=== Training Verifier Model (n_embd={VERIFIER_N_EMBD}) ===")
    verifier_params = init_params(VOCAB_SIZE, VERIFIER_N_EMBD, VERIFIER_N_HEAD, VERIFIER_N_LAYER)
    print(f"  Parameters: {count_params(verifier_params):,}")
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, verifier_params,
                VERIFIER_N_EMBD, VERIFIER_N_HEAD, VERIFIER_N_LAYER,
                VERIFIER_HEAD_DIM, VERIFIER_STEPS)
    verifier_train_time = time.time() - t0
    print(f"  Trained in {verifier_train_time:.1f}s")

    print(f"\n=== Training Draft Model (n_embd={DRAFT_N_EMBD}) ===")
    draft_params = init_params(VOCAB_SIZE, DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER)
    print(f"  Parameters: {count_params(draft_params):,}")
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, draft_params,
                DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER,
                DRAFT_HEAD_DIM, DRAFT_STEPS)
    draft_train_time = time.time() - t0
    print(f"  Trained in {draft_train_time:.1f}s")

    # Extract weights as plain floats for inference
    vwf = {k: extract(v) for k, v in verifier_params.items()}
    dwf = {k: extract(v) for k, v in draft_params.items()}
    vc: Cfg = {
        'n_embd': VERIFIER_N_EMBD, 'n_head': VERIFIER_N_HEAD,
        'n_layer': VERIFIER_N_LAYER, 'head_dim': VERIFIER_HEAD_DIM,
        'vocab_size': VOCAB_SIZE, 'bos': BOS,
    }
    dc: Cfg = {
        'n_embd': DRAFT_N_EMBD, 'n_head': DRAFT_N_HEAD,
        'n_layer': DRAFT_N_LAYER, 'head_dim': DRAFT_HEAD_DIM,
        'vocab_size': VOCAB_SIZE, 'bos': BOS,
    }

    def tok2str(toks: list[int]) -> str:
        return ''.join(unique_chars[t] if t != BOS else '' for t in toks)

    # === SPECULATIVE DECODING DEMONSTRATION ===
    print("\n=== Speculative Decoding: Draft-Verify Loop ===")
    print(f"Speculation window K={SPEC_K}: draft proposes {SPEC_K} tokens, verifier checks all\n")

    prompts = [
        ("a", [BOS, unique_chars.index('a')]),
        ("m", [BOS, unique_chars.index('m')]),
        ("s", [BOS, unique_chars.index('s')]),
    ]

    for label, ptoks in prompts:
        print(f'Prompt: "{label}"')
        toks, lp, v_fwd, d_fwd, proposed, accepted = decode_speculative(
            ptoks, vwf, dwf, vc, dc, max_len=12, draft_k=SPEC_K
        )
        acc_rate = 100.0 * accepted / max(proposed, 1)
        print(f"  Output: {label}{tok2str(toks)}")
        print(f"  Draft proposed: {proposed} | Draft accepted: {accepted} | Rate: {acc_rate:.0f}%")
        print(f"  Generated {len(toks)} tokens | Verifier fwd: {v_fwd} | Draft fwd: {d_fwd}")
        print()

    # === WALL-CLOCK COMPARISON ===
    # Compare verifier-only decoding vs speculative decoding across many samples.
    # Signpost: on real hardware with GPU batching, the speedup is dramatic because
    # verifying K tokens costs ~1 forward pass. Here in scalar Python, both models
    # run sequentially, so wall-clock speedup is modest. The acceptance rate is the
    # hardware-independent metric that predicts real-world speedup.
    print("=== Wall-Clock Comparison: Verifier-Only vs Speculative ===")
    n_samples = 20
    seeds = list("abcdefghijklmnopqrst")

    # Verifier-only baseline
    # Reset RNG state for fair comparison — both strategies see the same random draws
    rng_state = random.getstate()
    t0 = time.time()
    verifier_names: list[str] = []
    verifier_fwd_total = 0
    for i in range(n_samples):
        pt = [BOS, unique_chars.index(seeds[i])]
        toks, _, fwd = decode_verifier_only(pt, vwf, vc, max_len=12)
        verifier_names.append(seeds[i] + tok2str(toks))
        verifier_fwd_total += fwd
    verifier_time = time.time() - t0

    # Speculative decoding
    random.setstate(rng_state)
    t0 = time.time()
    spec_names: list[str] = []
    spec_v_fwd_total = 0
    spec_d_fwd_total = 0
    spec_proposed_total = 0
    spec_accepted_total = 0
    for i in range(n_samples):
        pt = [BOS, unique_chars.index(seeds[i])]
        toks, _, v_fwd, d_fwd, proposed, accepted = decode_speculative(
            pt, vwf, dwf, vc, dc, max_len=12, draft_k=SPEC_K
        )
        spec_names.append(seeds[i] + tok2str(toks))
        spec_v_fwd_total += v_fwd
        spec_d_fwd_total += d_fwd
        spec_proposed_total += proposed
        spec_accepted_total += accepted
    spec_time = time.time() - t0

    print(f"Generated {n_samples} names with each strategy:\n")
    print(f"{'Metric':<40} {'Verifier-Only':>15} {'Speculative':>15}")
    print("-" * 72)
    print(f"{'Wall-clock time (s)':<40} {verifier_time:>15.3f} {spec_time:>15.3f}")
    print(f"{'Verifier forward passes':<40} {verifier_fwd_total:>15} {spec_v_fwd_total:>15}")
    print(f"{'Draft forward passes':<40} {'N/A':>15} {spec_d_fwd_total:>15}")
    avg_acc_rate = 100.0 * spec_accepted_total / max(spec_proposed_total, 1)
    print(f"{'Draft acceptance rate':<40} {'N/A':>15} {avg_acc_rate:>14.1f}%")
    n_rounds = spec_proposed_total / SPEC_K if spec_proposed_total > 0 else 1
    toks_per_round = spec_accepted_total / max(n_rounds, 1)
    print(f"{'Avg draft tokens accepted per round':<40} {'1.0':>15} {toks_per_round:>15.1f}")
    # Note: each round also produces 1 extra token (bonus or resampled), so the
    # effective tokens-per-round is ~(accepted + 1). The acceptance rate measures
    # only draft-proposed tokens that passed the min(1, p_v/p_d) criterion.

    # === PER-STEP ACCEPTANCE RATE ANALYSIS ===
    # Acceptance rate typically decreases with position in the speculation window.
    # Token 1 is most likely to match because it's conditioned on the same prefix.
    # Token K is least likely because errors compound — each subsequent draft token
    # is conditioned on potentially wrong earlier drafts.
    print("\n=== Acceptance Rate by Position in Speculation Window ===")
    print("(Why K matters: diminishing returns with deeper speculation)\n")

    # Gather per-position statistics over many samples
    position_proposed = [0] * SPEC_K
    position_accepted = [0] * SPEC_K

    for i in range(n_samples):
        pt = [BOS, unique_chars.index(seeds[i])]
        # Run speculative decoding with detailed per-position tracking
        v_kv_tmp, v_logits_tmp = feed_prompt(pt, vwf, vc)
        d_kv_tmp, d_logits_tmp = feed_prompt(pt, dwf, dc)
        gen_tmp: list[int] = []
        max_gen = 12

        while len(gen_tmp) < max_gen:
            cur = len(pt) + len(gen_tmp)
            rem = min(SPEC_K, max_gen - len(gen_tmp))
            if cur >= BLOCK_SIZE or rem <= 0:
                break

            # Draft phase
            d_toks: list[int] = []
            d_probs: list[list[float]] = []
            td_kv = clone_kv(d_kv_tmp)
            td_logits = d_logits_tmp[:]
            for di in range(rem):
                pos = cur + di
                if pos >= BLOCK_SIZE:
                    break
                dp = softmax_f([l / 0.8 for l in td_logits])
                d_probs.append(dp)
                dtok = random.choices(range(dc['vocab_size']), weights=dp)[0]
                if dtok == dc['bos']:
                    break
                d_toks.append(dtok)
                td_logits = forward_float(dtok, pos, td_kv, dwf, dc)

            if not d_toks:
                break

            # Verify phase — track per-position accept/reject
            tv_kv = clone_kv(v_kv_tmp)
            tv_logits = v_logits_tmp[:]
            accepted_here: list[int] = []

            for vi in range(len(d_toks)):
                position_proposed[vi] += 1
                vp = softmax_f([l / 0.8 for l in tv_logits])
                dp = d_probs[vi]
                dtok = d_toks[vi]
                ratio = min(1.0, vp[dtok] / max(dp[dtok], 1e-10))

                if random.random() < ratio:
                    position_accepted[vi] += 1
                    accepted_here.append(dtok)
                    tv_logits = forward_float(dtok, cur + vi, tv_kv, vwf, vc)
                else:
                    # Rejection — resample and stop
                    adj = [max(0.0, vp[j] - dp[j]) for j in range(len(vp))]
                    adj_s = sum(adj)
                    if adj_s > 0:
                        adj = [a / adj_s for a in adj]
                        rtok = random.choices(range(vc['vocab_size']), weights=adj)[0]
                    else:
                        rtok = random.choices(range(vc['vocab_size']), weights=vp)[0]
                    if rtok != vc['bos']:
                        accepted_here.append(rtok)
                    break

            # Commit to real caches
            for ai, atok in enumerate(accepted_here):
                v_logits_tmp = forward_float(atok, cur + ai, v_kv_tmp, vwf, vc)
                d_logits_tmp = forward_float(atok, cur + ai, d_kv_tmp, dwf, dc)
                gen_tmp.append(atok)

            if not accepted_here:
                break

    print(f"{'Position':<12} {'Proposed':>10} {'Accepted':>10} {'Rate':>10}")
    print("-" * 44)
    for pos_idx in range(SPEC_K):
        if position_proposed[pos_idx] > 0:
            rate = 100.0 * position_accepted[pos_idx] / position_proposed[pos_idx]
            print(f"  {pos_idx + 1:<10} {position_proposed[pos_idx]:>10} "
                  f"{position_accepted[pos_idx]:>10} {rate:>9.1f}%")

    # Intuition: early positions have high acceptance because the draft and verifier
    # agree on the most common continuations. Later positions suffer from error
    # compounding — if position 2's draft token was borderline, position 3 is
    # conditioned on that shaky choice, making it even less likely to match.
    # This is why K=4 is typical: beyond that, acceptance drops below the overhead
    # of running the draft model for those positions.

    # === SAMPLE COMPARISON ===
    print("\n=== Generated Names: Verifier-Only vs Speculative ===")
    print(f"{'#':<4} {'Verifier-Only':<20} {'Speculative':<20}")
    print("-" * 44)
    for i in range(min(n_samples, 15)):
        print(f"{i + 1:<4} {verifier_names[i]:<20} {spec_names[i]:<20}")

    # Note: names differ because the RNG state diverges between the two strategies
    # (different number of random.random() calls). The distributions are identical —
    # given infinite samples, the frequency of each name would converge.

    # === K SENSITIVITY ANALYSIS ===
    # How does speculation window size K affect acceptance rate and efficiency?
    # Larger K proposes more tokens per round but later positions have lower
    # acceptance probability. There's an optimal K that maximizes tokens-per-
    # verifier-call.
    print("\n=== Speculation Window Size (K) Sensitivity ===\n")
    print(f"{'K':<6} {'Acc. Rate':>10} {'Tokens/Round':>14} {'Verifier Calls':>16}")
    print("-" * 48)

    for test_k in [1, 2, 4, 6, 8]:
        k_proposed = 0
        k_accepted = 0
        k_v_fwd = 0
        for i in range(n_samples):
            pt = [BOS, unique_chars.index(seeds[i])]
            _, _, v_f, _, prop, acc = decode_speculative(
                pt, vwf, dwf, vc, dc, max_len=12, draft_k=test_k
            )
            k_proposed += prop
            k_accepted += acc
            k_v_fwd += v_f

        k_rate = 100.0 * k_accepted / max(k_proposed, 1)
        k_rounds = k_proposed / test_k if k_proposed > 0 else 1
        k_tpr = k_accepted / max(k_rounds, 1)
        print(f"{test_k:<6} {k_rate:>9.1f}% {k_tpr:>14.1f} {k_v_fwd:>16}")

    # Intuition: K=1 has the highest per-token acceptance rate but generates only
    # 1-2 tokens per verifier call. Higher K values accept fewer tokens per-position
    # but amortize the verification cost over more attempts. The sweet spot depends
    # on draft-verifier alignment and the relative cost of draft vs verifier passes.

    print("\nDone.")

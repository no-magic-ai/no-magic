"""
How to shrink a model by 4x with minimal quality loss -- the math behind INT8 and INT4
weight quantization, demonstrated end-to-end: train, quantize, dequantize, compare.
"""
# Reference: Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers
# at Scale" (2022). https://arxiv.org/abs/2208.07339
# Also: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
# Pre-trained Transformers" (2022). https://arxiv.org/abs/2210.17323

# === TRADEOFFS ===
# + 2-4x model compression with minimal accuracy loss (INT8)
# + Faster inference on hardware with integer math units
# + Reduces memory bandwidth bottleneck (the real constraint in LLM serving)
# - Accuracy degrades at aggressive quantization (INT4 and below)
# - Calibration dataset required for post-training quantization
# - Some architectures quantize poorly (sensitive layers need mixed precision)
# WHEN TO USE: Deploying trained models to production, edge devices, or
#   memory-constrained environments where inference speed matters.
# WHEN NOT TO: During training (gradients need full precision), or when
#   accuracy requirements leave zero margin for quantization error.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture (identical to microgpt for consistency)
N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 16
HEAD_DIM = N_EMBD // N_HEAD  # 4

# Training parameters
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 800

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: 800 steps (vs 1000 in microgpt) is sufficient because this script's focus
# is quantization, not pushing training loss. We need a converged model, not an optimal one.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    return docs


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput), then backward() replays the graph in
    reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Reverse-mode autodiff via topological sort of the computation graph."""
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.
# Autograd is only used for training. Quantization and inference use plain floats.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Gaussian-initialized weight matrix. std=0.08 works for this tiny model."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


# === CORE OPERATIONS (AUTOGRAD) ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """y = W @ x. The fundamental neural network operation."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Stable softmax: exp(x - max(x)) / sum(exp(x_j - max(x_j)))."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps). No learned affine parameters."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped log to prevent log(0) = -inf from breaking gradients."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === CORE OPERATIONS (PLAIN FLOAT -- for quantized inference) ===
# After quantization, weights are dequantized back to floats. These functions
# mirror the autograd versions but operate on raw floats -- no gradient tracking
# because quantized models are inference-only.

def linear_float(x: list[float], w: list[list[float]]) -> list[float]:
    """y = W @ x with plain floats. Used for post-quantization inference."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax_float(logits: list[float]) -> list[float]:
    """Stable softmax on plain float logits."""
    max_val = max(logits)
    exp_vals = [math.exp(v - max_val) for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm_float(x: list[float]) -> list[float]:
    """RMSNorm on plain floats."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# === GPT FORWARD PASS (AUTOGRAD) ===

def gpt_forward(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict,
) -> list[Value]:
    """Single-token forward pass through the GPT. Returns vocab logits."""
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v = linear(x, params[f'layer{layer_idx}.attn_wv'])
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attn = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [k_t[hs:hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_h = [v_t[hs:hs + HEAD_DIM] for v_t in values[layer_idx]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === GPT FORWARD PASS (PLAIN FLOAT -- for quantized inference) ===
# Structurally identical to gpt_forward but operates on dequantized float weights.
# This separation keeps autograd overhead out of the quantization evaluation path.

def gpt_forward_float(
    token_id: int, pos_id: int,
    keys: list[list[list[float]]], values: list[list[list[float]]],
    float_params: dict[str, list[list[float]]],
) -> list[float]:
    """Single-token forward pass with plain float weights. No gradient tracking."""
    tok_emb = float_params['wte'][token_id]
    pos_emb = float_params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm_float(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm_float(x)
        q = linear_float(x, float_params[f'layer{layer_idx}.attn_wq'])
        k = linear_float(x, float_params[f'layer{layer_idx}.attn_wk'])
        v = linear_float(x, float_params[f'layer{layer_idx}.attn_wv'])
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attn: list[float] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [k_t[hs:hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_h = [v_t[hs:hs + HEAD_DIM] for v_t in values[layer_idx]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_h))
            ]
            attn_weights = softmax_float(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear_float(x_attn, float_params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm_float(x)
        x = linear_float(x, float_params[f'layer{layer_idx}.mlp_fc1'])
        x = [max(0.0, xi) for xi in x]  # ReLU on plain floats
        x = linear_float(x, float_params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear_float(x, float_params['lm_head'])


# === QUANTIZATION FUNCTIONS ===
# Quantization maps continuous float weights to a discrete integer grid.
# The core insight: neural network weights are approximately normally distributed
# with small magnitude. Most values cluster near zero, so mapping to [-127, +127]
# (INT8) or [-8, +7] (INT4) loses surprisingly little information. The network's
# nonlinearities and redundancy absorb the rounding error.

def quantize_absmax_int8(weights_float: list[list[float]]) -> tuple[list[list[int]], float]:
    """Absmax quantization: scale = max(|W|) / 127, q = round(W / scale).

    Maps the float range [-max|W|, +max|W|] to integer range [-127, +127].
    Symmetric around zero -- assumes weight distribution is roughly centered.
    This is the simplest quantization scheme and the baseline for everything else.

    Math: q_i = clamp(round(w_i / s), -127, 127)  where s = max(|W|) / 127
    Dequant: w_hat_i = q_i * s
    """
    max_abs = max(abs(w) for row in weights_float for w in row)
    if max_abs == 0:
        return [[0] * len(row) for row in weights_float], 1.0
    scale = max_abs / 127.0
    quantized = [[max(-127, min(127, round(w / scale))) for w in row] for row in weights_float]
    return quantized, scale


def quantize_absmax_int4(weights_float: list[list[float]]) -> tuple[list[list[int]], float]:
    """INT4 quantization: maps to [-8, +7] (4-bit signed integer range).

    8x compression vs float32. The quantization grid is 16x coarser than INT8,
    so rounding errors are substantially larger. Neural nets tolerate this because
    individual weight precision matters less than the collective statistical
    properties of weight matrices.

    Signpost: production INT4 (GPTQ, AWQ) uses calibration data to minimize
    output error rather than naive round-to-nearest. That reduces quality loss
    by 2-5x but requires a calibration dataset.
    """
    max_abs = max(abs(w) for row in weights_float for w in row)
    if max_abs == 0:
        return [[0] * len(row) for row in weights_float], 1.0
    scale = max_abs / 7.0
    quantized = [[max(-8, min(7, round(w / scale))) for w in row] for row in weights_float]
    return quantized, scale


def quantize_zeropoint_int8(
    weights_float: list[list[float]],
) -> tuple[list[list[int]], float, int]:
    """Zero-point (asymmetric) quantization: maps [min_W, max_W] to [0, 255].

    Unlike absmax which centers on zero, zero-point shifts the mapping so the
    full 8-bit range covers the actual weight range. More accurate when weights
    are not symmetric around zero (common after ReLU-heavy architectures where
    biases shift distributions).

    Math: scale = (w_max - w_min) / 255
          zero_point = round(-w_min / scale)
          q_i = clamp(round(w_i / scale) + zero_point, 0, 255)
    Dequant: w_hat_i = (q_i - zero_point) * scale
    """
    all_weights = [w for row in weights_float for w in row]
    w_min = min(all_weights)
    w_max = max(all_weights)
    if w_max == w_min == 0:
        return [[0] * len(row) for row in weights_float], 1.0, 0
    if w_max == w_min:
        # A nonzero constant tensor has an empty range, so the scale formula
        # would divide by zero. Signpost: LLM.int8() and GPTQ define the range
        # only for max > min and prescribe no constant branch. As a teaching
        # boundary extension, widen the range to include zero: the constant then
        # sits on a grid endpoint and the same affine mapping reconstructs it.
        w_min = min(w_min, 0.0)
        w_max = max(w_max, 0.0)
    scale = (w_max - w_min) / 255.0
    zero_point = round(-w_min / scale)
    quantized = [
        [max(0, min(255, round(w / scale) + zero_point)) for w in row]
        for row in weights_float
    ]
    return quantized, scale, zero_point


def quantize_per_channel_int8(
    weights_float: list[list[float]],
) -> tuple[list[list[int]], list[float]]:
    """Per-channel quantization: each output row gets its own scale factor.

    Per-tensor quantization uses one scale for the entire matrix, so a single
    outlier weight forces the entire grid to be coarse. Per-channel (per-row)
    quantization lets each output neuron use its own range, dramatically reducing
    error for matrices with non-uniform row magnitudes.

    Signpost: LLM.int8() (Dettmers 2022) goes further with mixed-precision
    decomposition -- outlier channels stay in fp16 while the rest quantize to INT8.
    """
    quantized = []
    scales = []
    for row in weights_float:
        max_abs = max(abs(w) for w in row)
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        scales.append(scale)
        quantized.append([max(-127, min(127, round(w / scale))) for w in row])
    return quantized, scales


# === DEQUANTIZATION FUNCTIONS ===
# Reverse the quantization mapping to recover approximate float weights.
# The difference between original and dequantized weights is the quantization error.

def dequantize_absmax(quantized: list[list[int]], scale: float) -> list[list[float]]:
    """w_hat = q * scale. Simple multiplication recovers approximate floats."""
    return [[q * scale for q in row] for row in quantized]


def dequantize_zeropoint(
    quantized: list[list[int]], scale: float, zero_point: int,
) -> list[list[float]]:
    """w_hat = (q - zero_point) * scale. Undo the asymmetric shift."""
    return [[(q - zero_point) * scale for q in row] for row in quantized]


def dequantize_per_channel(
    quantized: list[list[int]], scales: list[float],
) -> list[list[float]]:
    """w_hat[i] = q[i] * scale[row_index]. Each row uses its own scale."""
    return [
        [q * scales[i] for q in quantized[i]]
        for i in range(len(quantized))
    ]


# === EVALUATION HELPERS ===

def extract_float_weights(params: dict) -> dict[str, list[list[float]]]:
    """Extract Value.data from all parameter matrices into plain float lists."""
    float_weights: dict[str, list[list[float]]] = {}
    for name, matrix in params.items():
        float_weights[name] = [[v.data for v in row] for row in matrix]
    return float_weights


def compute_model_size(float_weights: dict[str, list[list[float]]], bits: int) -> int:
    """Compute model size in bytes at the given bit width.

    Float32 = 4 bytes/weight, INT8 = 1 byte/weight, INT4 = 0.5 bytes/weight.
    Scale factors add negligible overhead (one float per matrix or per row).
    """
    n_weights = sum(len(row) for matrix in float_weights.values() for row in matrix)
    return int(n_weights * bits / 8)


def compute_roundtrip_error(
    original: dict[str, list[list[float]]],
    dequantized: dict[str, list[list[float]]],
) -> float:
    """Max absolute error across all weights: max |w - dequant(quant(w))|.

    This is the worst-case single-weight error. Mean error is typically 10-100x
    smaller, but max error determines whether any particular computation path
    experiences catastrophic distortion.
    """
    max_err = 0.0
    for name in original:
        for orig_row, deq_row in zip(original[name], dequantized[name]):
            for o, d in zip(orig_row, deq_row):
                max_err = max(max_err, abs(o - d))
    return max_err


def evaluate_loss(
    float_params: dict[str, list[list[float]]],
    eval_docs: list[str],
    unique_chars: list[str],
    bos: int,
) -> float:
    """Average cross-entropy loss on evaluation documents using float forward pass."""
    total_loss = 0.0
    total_tokens = 0
    for doc in eval_docs:
        tokens = [bos] + [unique_chars.index(ch) for ch in doc] + [bos]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
        values: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
        for pos in range(seq_len):
            logits = gpt_forward_float(tokens[pos], pos, keys, values, float_params)
            probs = softmax_float(logits)
            # Cross-entropy: -log(p(target))
            p_target = max(probs[tokens[pos + 1]], 1e-10)
            total_loss += -math.log(p_target)
            total_tokens += 1
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def generate_sample(
    float_params: dict[str, list[list[float]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    temperature: float = 0.5,
) -> str:
    """Generate a single name from the model using temperature-scaled sampling."""
    keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    values: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    token_id = bos
    generated: list[str] = []
    for pos in range(BLOCK_SIZE):
        logits = gpt_forward_float(token_id, pos, keys, values, float_params)
        scaled = [l / temperature for l in logits]
        probs = softmax_float(scaled)
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == bos:
            break
        generated.append(unique_chars[token_id])
    return ''.join(generated)


# === TRAINING AND QUANTIZATION ===

if __name__ == "__main__":
    t_start = time.time()

    # === PHASE 1: TRAIN BASE MODEL ===
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents, vocabulary: {VOCAB_SIZE} tokens")

    # Initialize parameters
    params: dict[str, list[list[Value]]] = {}
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)
    for li in range(N_LAYER):
        params[f'layer{li}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{li}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{li}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,}\n")

    # Adam optimizer state
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    print("Training base model...")
    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
        values: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, values, params)
            probs = softmax(logits)
            loss_t = -safe_log(probs[tokens[pos + 1]])
            losses.append(loss_t)

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, param in enumerate(param_list):
            m[i] = BETA1 * m[i] + (1 - BETA1) * param.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * param.grad ** 2
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    t_train = time.time()
    print(f"\nTraining complete ({t_train - t_start:.1f}s). Final loss: {loss.data:.4f}")

    # === PHASE 2: EXTRACT FLOAT32 WEIGHTS ===
    # From this point forward, autograd is not used. All operations are on plain floats.
    # This mirrors production quantization: you receive a trained model checkpoint and
    # apply post-training quantization (PTQ) without any additional training.
    print("\n=== Extracting Float32 Weights ===")
    float_weights = extract_float_weights(params)

    # Use a fixed evaluation set (first 200 docs) for consistent loss comparison
    eval_docs = docs[:200]

    # Baseline loss with original float32 weights
    baseline_loss = evaluate_loss(float_weights, eval_docs, unique_chars, BOS)
    print(f"Float32 baseline loss: {baseline_loss:.4f}")

    # Seed reset for reproducible generation across all quantization variants
    random.seed(42)
    baseline_sample = generate_sample(float_weights, unique_chars, BOS, VOCAB_SIZE)

    # === PHASE 3: QUANTIZE TO INT8 (ABSMAX) ===
    print("\n=== INT8 Absmax Quantization ===")
    int8_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s = quantize_absmax_int8(matrix)
        int8_weights[name] = dequantize_absmax(q, s)

    int8_loss = evaluate_loss(int8_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    int8_sample = generate_sample(int8_weights, unique_chars, BOS, VOCAB_SIZE)
    int8_err = compute_roundtrip_error(float_weights, int8_weights)
    print(f"INT8 absmax loss: {int8_loss:.4f} (delta: {(int8_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 4: QUANTIZE TO INT4 (ABSMAX) ===
    print("\n=== INT4 Absmax Quantization ===")
    int4_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s = quantize_absmax_int4(matrix)
        int4_weights[name] = dequantize_absmax(q, s)

    int4_loss = evaluate_loss(int4_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    int4_sample = generate_sample(int4_weights, unique_chars, BOS, VOCAB_SIZE)
    int4_err = compute_roundtrip_error(float_weights, int4_weights)
    print(f"INT4 absmax loss: {int4_loss:.4f} (delta: {(int4_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 5: ZERO-POINT QUANTIZATION (ASYMMETRIC INT8) ===
    # Useful when weight distributions are shifted away from zero. After ReLU
    # activations or with certain initialization schemes, weights can be
    # asymmetrically distributed. Zero-point mapping captures this asymmetry.
    print("\n=== INT8 Zero-Point Quantization ===")
    zp_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, s, zp = quantize_zeropoint_int8(matrix)
        zp_weights[name] = dequantize_zeropoint(q, s, zp)

    zp_loss = evaluate_loss(zp_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    zp_sample = generate_sample(zp_weights, unique_chars, BOS, VOCAB_SIZE)
    zp_err = compute_roundtrip_error(float_weights, zp_weights)
    print(f"INT8 zero-point loss: {zp_loss:.4f} (delta: {(zp_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 6: PER-CHANNEL INT8 ===
    # Per-tensor quantization is limited by the row with the largest outlier.
    # Per-channel quantization gives each output channel its own scale, so a
    # single outlier row doesn't degrade precision for the entire matrix.
    print("\n=== INT8 Per-Channel Quantization ===")
    pc_weights: dict[str, list[list[float]]] = {}
    for name, matrix in float_weights.items():
        q, scales = quantize_per_channel_int8(matrix)
        pc_weights[name] = dequantize_per_channel(q, scales)

    pc_loss = evaluate_loss(pc_weights, eval_docs, unique_chars, BOS)
    random.seed(42)
    pc_sample = generate_sample(pc_weights, unique_chars, BOS, VOCAB_SIZE)
    pc_err = compute_roundtrip_error(float_weights, pc_weights)
    print(f"INT8 per-channel loss: {pc_loss:.4f} (delta: {(pc_loss - baseline_loss) / baseline_loss * 100:+.1f}%)")

    # === PHASE 7: COMPARISON TABLE ===
    t_end = time.time()

    size_32 = compute_model_size(float_weights, 32)
    size_8 = compute_model_size(float_weights, 8)
    size_4 = compute_model_size(float_weights, 4)

    print("\n" + "=" * 80)
    print("=== Quantization Results ===")
    print("=" * 80)

    header = f"{'Method':<24} {'Bits':>4} {'Size':>9} {'Loss':>8} {'Delta':>8} {'Max Err':>10} {'Sample':<14}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Float32 (baseline)", 32, size_32, baseline_loss, 0.0, 0.0, baseline_sample),
        ("INT8 absmax", 8, size_8, int8_loss,
         (int8_loss - baseline_loss) / baseline_loss * 100, int8_err, int8_sample),
        ("INT8 per-channel", 8, size_8, pc_loss,
         (pc_loss - baseline_loss) / baseline_loss * 100, pc_err, pc_sample),
        ("INT8 zero-point", 8, size_8, zp_loss,
         (zp_loss - baseline_loss) / baseline_loss * 100, zp_err, zp_sample),
        ("INT4 absmax", 4, size_4, int4_loss,
         (int4_loss - baseline_loss) / baseline_loss * 100, int4_err, int4_sample),
    ]

    for name, bits, size, loss_val, delta, err, sample in rows:
        delta_str = "---" if delta == 0.0 else f"{delta:+.1f}%"
        err_str = "---" if err == 0.0 else f"{err:.6f}"
        # Truncate sample for display
        sample_disp = sample[:12] if len(sample) > 12 else sample
        print(f"{name:<24} {bits:>4} {size:>7,} B {loss_val:>8.4f} {delta_str:>8} {err_str:>10} {sample_disp:<14}")

    print(f"\nCompression ratios: float32->INT8 = {size_32 / size_8:.1f}x, "
          f"float32->INT4 = {size_32 / size_4:.1f}x")

    # Highlight the key finding: per-channel beats per-tensor
    if pc_loss < int8_loss:
        print("Per-channel INT8 outperforms per-tensor INT8 (lower loss delta).")
    else:
        print("Per-tensor and per-channel INT8 performed comparably on this small model.")

    # === WHY QUANTIZATION WORKS ===
    # Neural nets are robust to weight precision loss for two reasons:
    # 1. Redundancy: thousands of weights contribute to each output, so individual
    #    rounding errors average out. The central limit theorem in action.
    # 2. Weight distributions are approximately Gaussian with small variance.
    #    Most weights are near zero where the quantization grid is densest.
    #    Only outlier weights suffer significant rounding error.
    #
    # INT8 preserves ~99% of the information (256 quantization levels).
    # INT4 is the practical limit -- 16 levels cause visible degradation.
    # INT2 (4 levels) typically destroys model quality entirely.
    #
    # Signpost: production quantization adds several sophistications this script omits:
    # - GPTQ/AWQ use calibration data to minimize layer-wise output reconstruction error
    # - Mixed precision keeps sensitive layers (first/last) in higher precision
    # - Activation quantization (not just weights) for full integer inference
    # - Group quantization: per-channel but with groups of 32-128 elements per scale
    # - SmoothQuant migrates quantization difficulty from activations to weights

    print(f"\nTotal runtime: {t_end - t_start:.1f}s")

"""
Supervised fine-tuning from first principles: a tiny causal decoder is first pretrained on plain
text, then the SAME learned weights are trained on prompt/response demonstrations with a loss on
the response only -- turning a next-character predictor into a model that follows two commands.
"""
# Reference: Ouyang et al., "Training language models to follow instructions with human
# feedback" (2022). https://arxiv.org/abs/2203.02155 -- Section 3.1 step 1, Figure 2 and
# Section 3.5 (SFT): start from a pretrained language model and fine-tune it with supervised
# learning on demonstrations, before any reward model or PPO stage (see microppo.py).
# Signpost: the paper does not prescribe a response-only loss mask or give a numbered SFT
# equation (its Eq. 1 is the reward-model ranking loss, not SFT). Masking the prompt targets
# here is a disclosed teaching choice. The decoder reuses the microgpt layout (Radford et al.,
# 2019) at toy scale: 1 layer, 1 head, RMSNorm without a learned scale, ReLU MLP, no biases.
# This is a toy SFT stage on synthetic demonstrations -- not GPT-3, not human labelers and
# not a reproduction of InstructGPT's results.

# === TRADEOFFS ===
# + Demonstrations state the desired behavior directly: no reward model is needed
# + Starts from what pretraining already learned instead of from random weights
# + Full-parameter SFT adds no adapter or extra inference cost
# - Demonstrations are expensive to write
# - The objective rewards imitation: errors in the demonstrations are learned as targets.
#   InstructGPT follows SFT with a reward model and PPO (see microppo.py)
# - Updating every weight can overwrite pretrained behavior; measured below, not assumed
# WHEN TO USE: Teaching a pretrained model a response format or task, and as the first
#   stage before preference tuning (microppo.py, microdpo.py).
# WHEN NOT TO: When only preference comparisons exist (no demonstrations), or when memory
#   forbids full-parameter updates (adapt a frozen base with microlora.py instead).

from __future__ import annotations

import math
import random
from collections.abc import Iterable

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture -- one layer, one attention head, deliberately tiny
N_EMBD = 8  # embedding dimension (d_model)
FF_DIM = 32  # feed-forward hidden width (4 * N_EMBD, the GPT convention)
BLOCK_SIZE = 10  # context: boundary + 7 prompt chars + 1 response char + boundary
INIT_STD = 0.08  # Gaussian init std, as in microgpt

# Vocabulary: 16 ordinary characters plus ONE boundary token shared by start and end.
# "copy" and "next" reuse c and e from the cyclic alphabet; the other eight characters
# (n o p t x y : >) only ever occur inside SFT prompts.
SYMBOLS = "abcdefgh"
PROMPT_ONLY_CHARS = "noptxy:>"
CHARS = SYMBOLS + PROMPT_ONLY_CHARS
BOUNDARY = len(CHARS)  # token id 16
VOCAB_SIZE = len(CHARS) + 1  # 17

COMMANDS = ("copy", "next")
HELDOUT_SYMBOL = (
    "h"  # copy:h> and next:h> are never trained on, tuned on or used to stop
)

# Training. Pretraining: each update uses ONE uniformly sampled base string.
# SFT: each update uses the mean response loss over ALL 14 training pairs (full batch).
# Signpost: full-batch SFT was chosen AFTER a one-sampled-pair-per-update version failed
# the next-command criterion below at this seed (3/7). It is an engineering amendment backed
# by training-side diagnostics, not a pre-registered test or a claim that batching is needed.
PRETRAIN_STEPS = 200
SFT_STEPS = 300
LEARNING_RATE = 0.01  # Adam, constant, same for both stages
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
LOG_EVERY = 50

# Acceptance -- frozen before the first run and checked on TRAINING data only.
# Held-out prompts are reported, never used as a criterion.
MIN_RELATIVE_DECREASE = 0.05  # each stage must cut its training loss by at least 5%
MIN_CORRECT_PER_COMMAND = 4  # exact greedy responses out of the 7 training prompts
EXPECTED_PARAMS = 1120

# Signpost: 1,120 parameters and 22 sequences. InstructGPT fine-tuned 1.3B-175B parameter
# GPT-3 models on ~13k labeler demonstrations, averaging the loss over mini-batches. The
# mechanism -- same weights, new data, supervised next-token likelihood on the demonstrated
# answer -- is what carries over.


# === DATA ===


def encode(text: str) -> list[int]:
    """Map characters to token ids (index into CHARS)."""
    return [CHARS.index(ch) for ch in text]


def build_base_corpus() -> list[list[int]]:
    """Eight length-8 cyclic strings framed by boundaries: 'abcdefgh', 'bcdefgha', ...

    This plain text is the only pretraining data. Each framed string has 10 tokens, so
    next-token training feeds input positions 0-8 -- including positions 7 and 8, where
    the SFT response will later be decided. Position 9 is only ever a target.
    """
    corpus = []
    for start in range(len(SYMBOLS)):
        text = "".join(SYMBOLS[(start + i) % len(SYMBOLS)] for i in range(len(SYMBOLS)))
        corpus.append([BOUNDARY] + encode(text) + [BOUNDARY])
    return corpus


def demonstration_response(command: str, symbol: str) -> str:
    """Write the demonstrated answer for one prompt.

    Plays the role of the human demonstrator: it is called only while BUILDING the
    dataset. Inference below never calls it -- answers must come from model logits.
    """
    if command == "copy":
        return symbol
    if command == "next":
        return SYMBOLS[(SYMBOLS.index(symbol) + 1) % len(SYMBOLS)]
    raise ValueError(f"unknown command {command!r}")


def build_demonstrations() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (training, held-out) prompt/response pairs, e.g. ('next:a>', 'b').

    2 commands x 8 symbols = 16 pairs; the 14 pairs for a-g train, the 2 pairs for 'h'
    are held out. Pretraining saw 'h' and 'h -> a' as plain text, so the hold-out tests
    command transfer on a familiar symbol, not recognition of a never-seen character.
    """
    train: list[tuple[str, str]] = []
    heldout: list[tuple[str, str]] = []
    for command in COMMANDS:
        for symbol in SYMBOLS:
            pair = (f"{command}:{symbol}>", demonstration_response(command, symbol))
            (heldout if symbol == HELDOUT_SYMBOL else train).append(pair)
    return train, heldout


def sft_example(prompt: str, response: str) -> tuple[list[int], int]:
    """Frame a demonstration and return (tokens, index of the first supervised target).

    tokens = [BOUNDARY, c, o, p, y, :, a, >, a, BOUNDARY]
    index     0        1  2  3  4  5  6  7  8  9
    The prompt occupies indices 0-7, so the first supervised target is index 8 (the
    response) and the last is index 9 (the end boundary): 2 supervised targets.
    """
    tokens = [BOUNDARY] + encode(prompt) + encode(response) + [BOUNDARY]
    if len(tokens) > BLOCK_SIZE:
        raise ValueError(
            f"{prompt!r} -> {response!r} exceeds the {BLOCK_SIZE}-token context"
        )
    return tokens, 1 + len(prompt)


# === SCALAR AUTOGRAD ENGINE ===


class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Every forward operation records its local derivative (dout/dinput). backward()
    replays the computation graph in reverse topological order, accumulating gradients
    via the chain rule: dLoss/dx = sum over paths (product of local gradients along path).
    """

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        local_grads: tuple[float, ...] = (),
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        return Value(
            self.data**exponent, (self,), (exponent * self.data ** (exponent - 1),)
        )

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other**-1)

    def __rtruediv__(self, other: float) -> Value:
        return other * (self**-1)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t**2,))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self) -> Value:
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort of the computation graph."""
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
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


def value_sum(terms: Iterable[Value]) -> Value:
    """Sum Values into one graph node chain, starting from an explicit Value(0.0).

    Same values and gradients as the builtin sum(), which starts from the integer 0,
    but the result is always a Value rather than possibly the integer 0.
    """
    total = Value(0.0)
    for term in terms:
        total = total + term
    return total


# === MODEL DEFINITION ===


def make_matrix(nrows: int, ncols: int) -> list[list[Value]]:
    """Gaussian-initialized weight matrix [nrows, ncols]."""
    return [
        [Value(random.gauss(0, INIT_STD)) for _ in range(ncols)] for _ in range(nrows)
    ]


def init_parameters() -> dict[str, list[list[Value]]]:
    """All weights of the 1-layer, 1-head decoder.

    Count: wte 17x8 + wpe 10x8 + Q/K/V/O 4x(8x8) + fc1 32x8 + fc2 8x32 + lm_head 17x8
         = 136 + 80 + 256 + 256 + 256 + 136 = 1,120.
    No biases and no learned normalization scale, so every parameter is a matrix entry.
    """
    return {
        "wte": make_matrix(VOCAB_SIZE, N_EMBD),
        "wpe": make_matrix(BLOCK_SIZE, N_EMBD),
        "attn_wq": make_matrix(N_EMBD, N_EMBD),
        "attn_wk": make_matrix(N_EMBD, N_EMBD),
        "attn_wv": make_matrix(N_EMBD, N_EMBD),
        "attn_wo": make_matrix(N_EMBD, N_EMBD),
        "mlp_fc1": make_matrix(FF_DIM, N_EMBD),
        "mlp_fc2": make_matrix(N_EMBD, FF_DIM),
        "lm_head": make_matrix(VOCAB_SIZE, N_EMBD),
    }


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Collect every Value from the parameter dict into one flat list."""
    return [p for matrix in params.values() for row in matrix for p in row]


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """y = W @ x for W of shape [n_out, n_in] (no bias)."""
    return [value_sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Stable softmax: subtracting the max leaves the result unchanged but avoids overflow."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = value_sum(exp_vals)
    return [e / total for e in exp_vals]


def log_softmax(logits: list[Value]) -> list[Value]:
    """Stable log-probabilities: log p_i = (z_i - m) - log(sum_j exp(z_j - m)), m = max z.

    The sum contains exp(0) = 1, so the log argument is >= 1: finite for any logits,
    with no probability clamp that would silently cut the gradient.
    """
    shift = max(v.data for v in logits)
    log_norm = value_sum((v - shift).exp() for v in logits).log()
    return [(v - shift) - log_norm for v in logits]


def rmsnorm(x: list[Value]) -> list[Value]:
    """x / sqrt(mean(x^2) + eps), with no learned scale (a fixed, parameter-free norm)."""
    mean_sq = value_sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def decoder_step(
    token_id: int,
    pos: int,
    keys: list[list[Value]],
    values: list[list[Value]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """Process one token at position `pos` and return logits for the NEXT token.

    keys/values accumulate the projections of positions 0..pos. Attending only over
    that cache is the causal mask: position pos can never see a later token.
    """
    x = [t + p for t, p in zip(params["wte"][token_id], params["wpe"][pos])]
    x = rmsnorm(x)

    # -- Single-head causal self-attention (pre-norm, residual) --
    residual = x
    x = rmsnorm(x)
    q = linear(x, params["attn_wq"])
    keys.append(linear(x, params["attn_wk"]))
    values.append(linear(x, params["attn_wv"]))
    # score_t = q . k_t / sqrt(d): how much this position reads from position t
    scores = [
        value_sum(q[j] * k_t[j] for j in range(N_EMBD)) / math.sqrt(N_EMBD)
        for k_t in keys
    ]
    weights = softmax(scores)
    attended = [
        value_sum(weights[t] * values[t][j] for t in range(len(values)))
        for j in range(N_EMBD)
    ]
    # This is how the prompt conditions the answer: at the '>' position the query can
    # read the command letters and the symbol from earlier positions' keys and values.
    x = [a + b for a, b in zip(linear(attended, params["attn_wo"]), residual)]

    # -- Feed-forward block (pre-norm, residual) --
    residual = x
    hidden = [h.relu() for h in linear(rmsnorm(x), params["mlp_fc1"])]
    x = [a + b for a, b in zip(linear(hidden, params["mlp_fc2"]), residual)]

    return linear(x, params["lm_head"])


# === OBJECTIVES ===


def mean_target_nll(
    params: dict[str, list[list[Value]]], tokens: list[int], first_target: int
) -> Value:
    """Mean of -log p(tokens[t] | tokens[0..t-1]) over targets t = first_target .. end.

    Causal shift: the decoder output at position t-1 (which has seen tokens 0..t-1)
    predicts token t. Pretraining passes first_target=1, supervising all 9 next tokens.
    SFT passes the prompt length, supervising only the response and end boundary:

        L_SFT = -(1/|R|) * sum_{t in R} log p(x_t | x_<t),   R = {8, 9}, |R| = 2

    Masking removes the prompt TARGETS from the sum, not the prompt INPUTS: every prompt
    position still runs through the decoder and feeds the attention cache, so gradients
    reach prompt embeddings through the response predictions that read them.
    """
    if len(tokens) > BLOCK_SIZE:
        raise ValueError(
            f"sequence of {len(tokens)} tokens exceeds context {BLOCK_SIZE}"
        )
    if not 1 <= first_target < len(tokens):
        raise ValueError(f"first_target={first_target} leaves no supervised target")
    keys: list[list[Value]] = []
    values: list[list[Value]] = []
    losses: list[Value] = []
    for pos in range(len(tokens) - 1):
        logits = decoder_step(tokens[pos], pos, keys, values, params)
        if pos + 1 >= first_target:
            losses.append(-log_softmax(logits)[tokens[pos + 1]])
    return value_sum(losses) * (1.0 / len(losses))


def mean_loss(
    params: dict[str, list[list[Value]]], examples: list[tuple[list[int], int]]
) -> float:
    """Average of mean_target_nll over a complete declared partition (evaluation only)."""
    return sum(mean_target_nll(params, t, first).data for t, first in examples) / len(
        examples
    )


# === OPTIMIZER ===


def adam_update(
    param_list: list[Value], m_state: list[float], v_state: list[float], step: int
) -> None:
    """One Adam step with bias correction at a constant learning rate, then zero grads."""
    for i, param in enumerate(param_list):
        m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * param.grad
        v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * param.grad**2
        m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
        v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
        param.data -= LEARNING_RATE * m_hat / (v_hat**0.5 + EPS_ADAM)
        param.grad = 0.0


def accumulate_batch_gradient(
    params: dict[str, list[list[Value]]], examples: list[tuple[list[int], int]]
) -> float:
    """Add the gradient of the full-batch SFT loss to param.grad; return the loss.

        L_batch = (1/B) * sum_i L_i,   L_i = mean_target_nll over pair i's 2 targets

    With B = 14 pairs and 2 targets each this equals the mean over all 28 supervised
    targets. Each pair gets its own forward pass and attention cache, so no pair can
    attend to another. Gradients are linear, so running backward() on (1/B) * L_i for
    every pair and letting param.grad accumulate gives exactly dL_batch/dparam.
    """
    scale = 1.0 / len(examples)
    batch_loss = 0.0
    for tokens, first_target in examples:
        loss = mean_target_nll(params, tokens, first_target) * scale
        loss.backward()
        batch_loss += loss.data
    return batch_loss


def train_stage(
    params: dict[str, list[list[Value]]],
    examples: list[tuple[list[int], int]],
    num_steps: int,
    full_batch: bool,
) -> None:
    """Run num_steps Adam updates on the given examples.

    full_batch=False (pretraining): each update uses ONE uniformly sampled example.
    full_batch=True (SFT): each update uses the mean loss over ALL examples.
    Moments start at zero on every call: the second stage inherits the weights of the
    first, never its optimizer state. Weights are NOT reinitialized here. Gradients are
    zeroed once per update, inside adam_update, after the step is taken.
    """
    param_list = flatten_params(params)
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)
    for step in range(num_steps):
        if full_batch:
            loss_value = accumulate_batch_gradient(params, examples)
            label = f"mean loss over all {len(examples)} pairs"
        else:
            tokens, first_target = random.choice(examples)
            loss = mean_target_nll(params, tokens, first_target)
            loss.backward()
            loss_value = loss.data
            label = "sampled-sequence loss"
        adam_update(param_list, m_state, v_state, step)
        if (step + 1) % LOG_EVERY == 0 or step == 0:
            print(f"  step {step + 1:>3}/{num_steps} | {label} {loss_value:.4f}")


# === INFERENCE ===


def greedy_response(
    params: dict[str, list[list[Value]]], prompt: str
) -> tuple[str, bool]:
    """Decode an answer from model logits alone: argmax of softmax(logits / 1.0).

    Stops at the boundary token or when the context is full; returns (text, stopped).
    Nothing here can see demonstration_response -- the answer is the model's.
    """
    tokens = [BOUNDARY] + encode(prompt)
    keys: list[list[Value]] = []
    values: list[list[Value]] = []
    for pos in range(len(tokens) - 1):
        decoder_step(tokens[pos], pos, keys, values, params)  # prefill the prompt
    generated: list[str] = []
    token_id, pos = tokens[-1], len(tokens) - 1
    while pos < BLOCK_SIZE - 1:
        probs = [
            p.data for p in softmax(decoder_step(token_id, pos, keys, values, params))
        ]
        token_id = max(range(VOCAB_SIZE), key=lambda i: probs[i])
        if token_id == BOUNDARY:
            return "".join(generated), True
        generated.append(CHARS[token_id])
        pos += 1
    return "".join(generated), False


def decode_report(
    params: dict[str, list[list[Value]]], demos: list[tuple[str, str]]
) -> dict[str, int]:
    """Print each prompt's greedy answer and return exact matches per command.

    Exact means the decoded text equals the demonstration AND the model emitted the end
    boundary: 'b' followed by more characters is not a correct 'next:a>' answer.
    """
    correct = {command: 0 for command in COMMANDS}
    for prompt, response in demos:
        text, stopped = greedy_response(params, prompt)
        exact = stopped and text == response
        correct[prompt.split(":")[0]] += exact
        shown = text + ("<end>" if stopped else "...")
        print(
            f"    {prompt} -> {shown:<8} expected {response}<end>  {'ok' if exact else '--'}"
        )
    return correct


# === TRAINING AND INFERENCE ===

if __name__ == "__main__":
    base_corpus = build_base_corpus()
    train_demos, heldout_demos = build_demonstrations()
    base_examples = [(tokens, 1) for tokens in base_corpus]
    sft_train = [sft_example(p, r) for p, r in train_demos]
    sft_heldout = [sft_example(p, r) for p, r in heldout_demos]
    print(
        f"Base corpus: {len(base_corpus)} cyclic strings, e.g. "
        f"{''.join(CHARS[t] for t in base_corpus[0][1:-1])}"
    )
    print(
        f"SFT demonstrations: {len(train_demos)} train, {len(heldout_demos)} held out "
        f"({', '.join(p for p, _ in heldout_demos)})"
    )

    params = init_parameters()
    param_list = flatten_params(params)
    if len(param_list) != EXPECTED_PARAMS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAMS} parameters, built {len(param_list)}"
        )
    print(
        f"Decoder parameters: {len(param_list):,} (vocab {VOCAB_SIZE}, d={N_EMBD}, "
        f"ff={FF_DIM}, context {BLOCK_SIZE}, 1 layer, 1 head)"
    )

    init_weights = [p.data for p in param_list]
    prompt_only_ids = encode(PROMPT_ONLY_CHARS)
    init_prompt_rows = [[p.data for p in params["wte"][i]] for i in prompt_only_ids]
    init_base_nll = mean_loss(params, base_examples)

    # === Stage 1: pretraining on plain text ===
    # Next-token likelihood on every position. No prompts, no commands: the model only
    # learns which symbol follows which in the a..h cycle.
    print(
        f"\n=== Stage 1: pretraining ({PRETRAIN_STEPS} updates, one sampled string each) ==="
    )
    train_stage(params, base_examples, PRETRAIN_STEPS, full_batch=False)
    base_weights = [p.data for p in param_list]  # the snapshot SFT must start from
    base_nll = mean_loss(params, base_examples)
    base_sft_nll = mean_loss(params, sft_train)
    base_heldout_nll = mean_loss(params, sft_heldout)
    pretrain_changed = sum(a != b for a, b in zip(init_weights, base_weights))
    prompt_rows_untouched = init_prompt_rows == [
        [p.data for p in params["wte"][i]] for i in prompt_only_ids
    ]
    print(f"Base-corpus NLL: {init_base_nll:.4f} (init) -> {base_nll:.4f} (pretrained)")
    print(f"Parameters changed by pretraining: {pretrain_changed}/{len(param_list)}")
    # Zero gradient + zero Adam moments => exactly zero update for these rows. Their
    # first learning signal arrives during SFT; they are NOT pretrained knowledge.
    print(
        f"Prompt-only embedding rows ({PROMPT_ONLY_CHARS}) untouched by pretraining: "
        f"{prompt_rows_untouched}"
    )
    print("Pretrained base on SFT prompts (no demonstrations seen yet):")
    decode_report(params, train_demos + heldout_demos)

    # Masking check on one demonstration: the loss has exactly 2 terms, yet its gradient
    # still reaches the prompt-only embeddings through attention (prompt conditioning).
    probe_tokens, probe_first = sft_train[0]
    mean_target_nll(params, probe_tokens, probe_first).backward()
    prompt_grad = sum(abs(p.grad) for i in prompt_only_ids for p in params["wte"][i])
    for p in param_list:
        p.grad = 0.0
    print(
        f"Supervised targets per SFT pair: {len(probe_tokens) - probe_first} "
        f"(of {len(probe_tokens) - 1} next-token positions)"
    )
    print(
        f"|grad| reaching prompt-only embeddings from the masked loss on "
        f"{train_demos[0][0]}: {prompt_grad:.6f}"
    )

    # === Stage 2: supervised fine-tuning of the same weights ===
    # Same parameters, same objective form, new data and a response-only target set.
    # Adam moments restart at zero; the weights continue from the pretrained snapshot.
    # Each update averages the response loss over all 14 training pairs (full batch).
    print(f"\n=== Stage 2: supervised fine-tuning ({SFT_STEPS} full-batch updates) ===")
    train_stage(params, sft_train, SFT_STEPS, full_batch=True)
    sft_weights = [p.data for p in param_list]
    sft_changed = sum(a != b for a, b in zip(base_weights, sft_weights))
    sft_nll = mean_loss(params, sft_train)
    sft_heldout_nll = mean_loss(params, sft_heldout)
    after_sft_base_nll = mean_loss(params, base_examples)

    print("\n=== Results ===")
    print(f"Parameters changed by SFT: {sft_changed}/{len(param_list)}")
    print(
        f"Response NLL, 14 training pairs: {base_sft_nll:.4f} (pretrained base) -> "
        f"{sft_nll:.4f} (after SFT)"
    )
    # Report-only: plain-text loss after SFT shows how much pretraining behavior moved.
    print(f"Base-corpus NLL after SFT (report-only): {after_sft_base_nll:.4f}")
    print("Fine-tuned model, training prompts:")
    train_correct = decode_report(params, train_demos)
    print(
        "Fine-tuned model, held-out prompts (report-only, never trained on or tuned on):"
    )
    decode_report(params, heldout_demos)
    print(
        f"Held-out response NLL (report-only): {base_heldout_nll:.4f} (pretrained base) -> "
        f"{sft_heldout_nll:.4f} (after SFT)"
    )

    # === Acceptance (frozen training-side criteria) ===
    checks = [
        (
            f"pretraining cuts base-corpus NLL by >= {MIN_RELATIVE_DECREASE:.0%}",
            base_nll <= (1 - MIN_RELATIVE_DECREASE) * init_base_nll,
        ),
        (
            f"SFT cuts training response NLL by >= {MIN_RELATIVE_DECREASE:.0%} vs the base",
            sft_nll <= (1 - MIN_RELATIVE_DECREASE) * base_sft_nll,
        ),
        ("parameters change in both stages", pretrain_changed > 0 and sft_changed > 0),
    ] + [
        (
            f"{command}: >= {MIN_CORRECT_PER_COMMAND}/7 exact training responses "
            f"(got {train_correct[command]})",
            train_correct[command] >= MIN_CORRECT_PER_COMMAND,
        )
        for command in COMMANDS
    ]
    print("\nAcceptance (training data only):")
    for description, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")
    failed = [description for description, passed in checks if not passed]
    if failed:
        raise RuntimeError(f"acceptance failed: {'; '.join(failed)}")

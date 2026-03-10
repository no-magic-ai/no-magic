"""
Long Short-Term Memory from first principles: the 4-gate architecture that solved the
vanishing gradient problem — trained on character-level name generation with zero dependencies.
"""
# Reference: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
# https://www.bioinf.jku.at/publications/older/2604.pdf

# === TRADEOFFS ===
# + Cell state as "information highway" preserves gradients over long sequences
# + 4-gate architecture gives fine-grained control over memory read/write/forget
# + Handles long-range dependencies that RNNs and even GRUs struggle with
# - 4x the parameters of vanilla RNN (4 gate weight matrices vs 1)
# - Slower per-step computation than GRU (4 gates vs 3)
# - Can still struggle with very long sequences (>1000 steps)
# WHEN TO USE: Sequence modeling where long-range dependencies matter and
#   you need explicit control over information flow (speech, music, long text).
# WHEN NOT TO: Short sequences where RNN/GRU suffice, or when you can
#   use transformers (which handle long-range deps via attention instead).

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_HIDDEN = 32       # hidden state / cell state dimension
SEQ_LEN = 16        # maximum sequence length for training
LEARNING_RATE = 0.003  # Adam learning rate — lower than SGD because Adam adapts per-parameter
BETA1 = 0.9            # Adam first moment decay
BETA2 = 0.999          # Adam second moment decay
EPS_ADAM = 1e-8        # Adam epsilon (prevents division by zero)
NUM_STEPS = 800        # training iterations — enough for convergence on this tiny model
TRAIN_SIZE = 200       # small training subset so each name is seen ~4x in 800 steps

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~5,000 parameters total (4 gates * [W_x + W_h + bias] + output projection).
# Production LSTMs have millions. Architecture is identical; this is toy scale for
# pedagogical clarity and CPU-friendly runtime.


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
    its local derivative (∂out/∂input) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # scalar float value
        self.grad = 0.0           # accumulated gradient (∂Loss/∂self)
        self._children = children # parent Values in the computation graph
        self._local_grads = local_grads  # ∂self/∂child for each child

    # Arithmetic operations
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

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # Activation functions
    def tanh(self):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def sigmoid(self):
        # sigmoid(x) = 1 / (1 + exp(-x))
        # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        # Gates need [0,1] outputs: sigmoid is the natural choice. When sigmoid ≈ 0,
        # information is blocked; when ≈ 1, information passes through unchanged.
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule. For a composite function
        f(g(h(x))), the chain rule says df/dx = (df/dg) * (dg/dh) * (dh/dx).
        The topological sort ensures we compute df/dg before we need it for df/dh.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: gradient of loss with respect to itself is 1
        self.grad = 1.0

        # Reverse topological order: gradients flow backward from output to inputs
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: ∂Loss/∂child += ∂Loss/∂v * ∂v/∂child
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the addition of sigmoid(), required for all 4 LSTM gate computations.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    Standard deviation of 0.08 is chosen empirically for this tiny model.
    Larger models typically use std = 1/sqrt(d_in) (Xavier/Glorot initialization).
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_lstm_params(vocab_size: int) -> dict:
    """Initialize all LSTM parameters: 4 gate weight matrices + output projection.

    LSTM has 4 independent gate computations, each requiring:
      - W_x*: [hidden_dim, vocab_size]  — input-to-gate weights
      - W_h*: [hidden_dim, hidden_dim]  — hidden-to-gate weights (recurrent)
      - b_*:  [hidden_dim]              — gate bias

    Total gate params: 4 * (hidden_dim * vocab_size + hidden_dim * hidden_dim + hidden_dim)
    Plus output projection: vocab_size * hidden_dim + vocab_size

    Why 4 separate gates instead of one big matrix?
    Each gate serves a distinct information-routing purpose:
      - Forget gate: what to DISCARD from cell state (old irrelevant info)
      - Input gate: what to WRITE to cell state (new relevant info)
      - Cell candidate: WHAT new information to write (the content)
      - Output gate: what to EXPOSE from cell state to the hidden output
    This decomposition gives the network fine-grained control over its memory.
    """
    params = {}

    # Forget gate — controls what to erase from cell state
    params['W_xf'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hf'] = make_matrix(N_HIDDEN, N_HIDDEN)
    # Forget gate bias initialized to 1.0 (not 0.0).
    # Why? At initialization, we want the forget gate to be open (f_t ≈ 1), meaning
    # "remember everything by default." If bias=0, sigmoid(0) = 0.5, and the network
    # starts by forgetting half its memory before it has learned what's important.
    # This trick from Jozefowicz et al. (2015) is crucial for LSTM training stability.
    params['b_f'] = [Value(1.0) for _ in range(N_HIDDEN)]

    # Input gate — controls what to write to cell state
    params['W_xi'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hi'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['b_i'] = [Value(0.0) for _ in range(N_HIDDEN)]

    # Cell candidate — the new content to potentially write
    params['W_xc'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hc'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['b_c'] = [Value(0.0) for _ in range(N_HIDDEN)]

    # Output gate — controls what to expose from cell state
    params['W_xo'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_ho'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['b_o'] = [Value(0.0) for _ in range(N_HIDDEN)]

    # Output projection: maps hidden state to vocabulary logits
    params['W_hy'] = make_matrix(vocab_size, N_HIDDEN)
    params['b_y'] = [Value(0.0) for _ in range(vocab_size)]

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]], b: list[Value] | None = None) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x + b (bias optional).

    For weight matrix W with shape [n_out, n_in] and input vector x with
    shape [n_in], computes output y with shape [n_out].
    """
    y = [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]
    if b is not None:
        y = [y_i + b_i for y_i, b_i in zip(y, b)]
    return y


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Softmax is translation-invariant: softmax(x) = softmax(x - c) for any c.
    We subtract max(x) before exp() to prevent overflow.
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm for numerical stability in loss computation.

    Prevents log(0) which returns -inf and breaks gradient backpropagation.
    Critical: we must keep `prob` as a child node so gradients flow back through
    the computation graph.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === LSTM CELL ===

def lstm_forward(
    x: list[Value],
    h_prev: list[Value],
    c_prev: list[Value],
    params: dict,
) -> tuple[list[Value], list[Value], list[Value], dict[str, list[float]]]:
    """Single-step LSTM forward pass through all 4 gates.

    The LSTM processes one timestep at a time. At each step, 4 gates operate on
    the concatenated input [h_{t-1}, x_t] to decide what to remember, what to
    forget, what to write, and what to output.

    The key insight: the cell state C_t acts as a "conveyor belt" that runs
    through the entire sequence. Information can be added (via input gate) or
    removed (via forget gate) at each step, but the default operation is to
    pass the cell state through unchanged. This additive update means:

        ∂C_t/∂C_{t-1} = f_t  (just the forget gate value, typically close to 1)

    Compare to vanilla RNN where ∂h_t/∂h_{t-1} involves multiplying by W_hh
    repeatedly, causing exponential gradient decay. The LSTM's cell state avoids
    this by using element-wise multiplication instead of matrix multiplication.

    Args:
        x: one-hot encoded input [vocab_size]
        h_prev: previous hidden state [hidden_dim]
        c_prev: previous cell state [hidden_dim]
        params: weight matrices and biases for all 4 gates

    Returns:
        (logits, h_new, c_new, gate_activations)
        gate_activations is a dict of raw float values for educational display
    """
    # --- Gate 1: Forget gate ---
    # f_t = σ(W_xf · x_t + W_hf · h_{t-1} + b_f)
    # Decides what to ERASE from the cell state. When f_t[i] ≈ 0, dimension i
    # of the cell state is wiped; when f_t[i] ≈ 1, it's preserved.
    f_x = linear(x, params['W_xf'])
    f_h = linear(h_prev, params['W_hf'])
    f_t = [(f_x_i + f_h_i + b_i).sigmoid()
           for f_x_i, f_h_i, b_i in zip(f_x, f_h, params['b_f'])]

    # --- Gate 2: Input gate ---
    # i_t = σ(W_xi · x_t + W_hi · h_{t-1} + b_i)
    # Decides what to WRITE to the cell state. Acts as a "write enable" signal.
    i_x = linear(x, params['W_xi'])
    i_h = linear(h_prev, params['W_hi'])
    i_t = [(i_x_i + i_h_i + b_i).sigmoid()
           for i_x_i, i_h_i, b_i in zip(i_x, i_h, params['b_i'])]

    # --- Gate 3: Cell candidate ---
    # C̃_t = tanh(W_xc · x_t + W_hc · h_{t-1} + b_c)
    # The actual content to potentially write. tanh bounds it to [-1, 1].
    # Unlike the gates (which use sigmoid for [0,1] scaling), the candidate
    # uses tanh because it represents information content, not a gate strength.
    c_x = linear(x, params['W_xc'])
    c_h = linear(h_prev, params['W_hc'])
    c_candidate = [(c_x_i + c_h_i + b_i).tanh()
                   for c_x_i, c_h_i, b_i in zip(c_x, c_h, params['b_c'])]

    # --- Cell state update ---
    # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    # This is the "conveyor belt" equation. Two element-wise operations:
    #   1. f_t ⊙ C_{t-1}: selectively forget old information
    #   2. i_t ⊙ C̃_t:    selectively write new information
    # The additive structure (not multiplicative!) is what preserves gradients.
    # Compare: vanilla RNN does h_t = tanh(W·h_{t-1} + ...) — the W multiplication
    # at every step causes exponential gradient decay. Here, the gradient through
    # the cell state is just f_t (a scalar near 1), not a matrix multiplication.
    c_new = [f_i * c_prev_i + i_i * c_cand_i
             for f_i, c_prev_i, i_i, c_cand_i in zip(f_t, c_prev, i_t, c_candidate)]

    # --- Gate 4: Output gate ---
    # o_t = σ(W_xo · x_t + W_ho · h_{t-1} + b_o)
    # Decides what to EXPOSE from the cell state to the outside world.
    # The cell state is the network's internal memory; the output gate
    # filters what parts of that memory are relevant for the current output.
    o_x = linear(x, params['W_xo'])
    o_h = linear(h_prev, params['W_ho'])
    o_t = [(o_x_i + o_h_i + b_i).sigmoid()
           for o_x_i, o_h_i, b_i in zip(o_x, o_h, params['b_o'])]

    # --- Hidden state ---
    # h_t = o_t ⊙ tanh(C_t)
    # tanh squashes the cell state to [-1, 1], then the output gate selects
    # which dimensions to expose. The hidden state is a "filtered view" of
    # the cell state — the cell remembers more than it reveals.
    h_new = [o_i * c_i.tanh() for o_i, c_i in zip(o_t, c_new)]

    # Output projection: map hidden state to vocabulary logits
    logits = linear(h_new, params['W_hy'], params['b_y'])

    # Capture gate activations for educational display (raw floats, not Values)
    gate_acts = {
        'forget': [f_i.data for f_i in f_t],
        'input': [i_i.data for i_i in i_t],
        'candidate': [c_i.data for c_i in c_candidate],
        'output': [o_i.data for o_i in o_t],
        'cell': [c_i.data for c_i in c_new],
    }

    return logits, h_new, c_new, gate_acts


# === TRAINING ===

def flatten_params(params: dict) -> list[Value]:
    """Collect all trainable parameters into a flat list for the optimizer."""
    param_list = []
    for val in params.values():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], Value):
                param_list.extend(val)
            elif isinstance(val[0], list):
                for row in val:
                    param_list.extend(row)
    return param_list


def train_lstm(
    docs: list[str],
    unique_chars: list[str],
    params: dict,
) -> float:
    """Train the LSTM on character-level next-character prediction using Adam.

    Each training step:
    1. Pick a name from the dataset
    2. Feed it character-by-character through the LSTM
    3. At each position, predict the next character
    4. Backpropagate the cross-entropy loss through all timesteps (BPTT)
    5. Update parameters with Adam optimizer

    Why Adam instead of SGD (which micrornn.py uses)?
    Adam maintains per-parameter learning rates that adapt based on gradient
    history. LSTMs have 4x the parameters of vanilla RNNs across gates with
    very different gradient magnitudes — forget gate gradients tend to be small
    (biased toward 1.0), while candidate gradients can be large. Adam handles
    this naturally; SGD would require careful per-gate learning rate tuning.
    """
    bos = len(unique_chars)
    vocab_size = len(unique_chars) + 1

    param_list = flatten_params(params)
    print(f"Training LSTM...")
    print(f"Parameters: {len(param_list):,}")

    # Adam optimizer state: per-parameter first and second moment estimates
    m = [0.0] * len(param_list)  # first moment (mean of gradients)
    v = [0.0] * len(param_list)  # second moment (mean of squared gradients)

    final_loss = 0.0

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]

        # Tokenize: [BOS, char_0, char_1, ..., char_n, BOS]
        # BOS serves as both start-of-sequence and end-of-sequence marker.
        tokens = [bos] + [unique_chars.index(ch) for ch in doc] + [bos]
        seq_len = min(SEQ_LEN, len(tokens) - 1)

        # Initialize hidden state and cell state to zeros.
        # The cell state starts empty — the network has no prior memory.
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        c = [Value(0.0) for _ in range(N_HIDDEN)]

        # Forward pass through the sequence
        losses = []
        for pos in range(seq_len):
            x_onehot = [Value(1.0 if i == tokens[pos] else 0.0)
                        for i in range(vocab_size)]

            logits, h, c, _ = lstm_forward(x_onehot, h, c, params)

            probs = softmax(logits)
            target = tokens[pos + 1]
            loss_t = -safe_log(probs[target])
            losses.append(loss_t)

        # Average cross-entropy loss over the sequence
        loss = (1.0 / seq_len) * sum(losses)

        # Backward pass (Backpropagation Through Time)
        # Gradients flow backward through all timesteps. The LSTM's cell state
        # acts as a gradient highway: ∂C_t/∂C_{t-1} = f_t (just the forget gate),
        # avoiding the matrix multiplication chain that kills vanilla RNN gradients.
        loss.backward()

        # Adam update
        # Adam combines momentum (exponential moving average of gradients) with
        # RMSProp (exponential moving average of squared gradients) to get
        # per-parameter adaptive learning rates.
        for idx, param in enumerate(param_list):
            g = param.grad
            m[idx] = BETA1 * m[idx] + (1 - BETA1) * g
            v[idx] = BETA2 * v[idx] + (1 - BETA2) * g * g

            # Bias correction: early steps have biased estimates because m and v
            # are initialized to 0. Dividing by (1 - beta^t) compensates.
            m_hat = m[idx] / (1 - BETA1 ** (step + 1))
            v_hat = v[idx] / (1 - BETA2 ** (step + 1))

            param.data -= LEARNING_RATE * m_hat / (math.sqrt(v_hat) + EPS_ADAM)
            param.grad = 0.0

        final_loss = loss.data

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    print(f"Training complete. Final loss: {final_loss:.4f}\n")
    return final_loss


# === INFERENCE ===

def generate_names(
    params: dict,
    unique_chars: list[str],
    num_samples: int = 10,
) -> list[str]:
    """Generate names by sampling from the trained LSTM one character at a time."""
    bos = len(unique_chars)
    vocab_size = len(unique_chars) + 1

    print(f"Generating {num_samples} names from trained LSTM:")

    samples = []
    for _ in range(num_samples):
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        c = [Value(0.0) for _ in range(N_HIDDEN)]
        token_id = bos
        generated = []

        for _ in range(SEQ_LEN):
            x_onehot = [Value(1.0 if i == token_id else 0.0)
                        for i in range(vocab_size)]

            logits, h, c, _ = lstm_forward(x_onehot, h, c, params)

            probs = softmax(logits)
            token_id = random.choices(
                range(vocab_size),
                weights=[p.data for p in probs]
            )[0]

            if token_id == bos:
                break
            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        samples.append(name)
        print(f"  {name}")

    print()
    return samples


# === GATE DYNAMICS VISUALIZATION ===

def show_gate_dynamics(
    params: dict,
    unique_chars: list[str],
    sample_name: str,
) -> None:
    """Feed a name through the trained LSTM and display gate activations per timestep.

    This reveals the internal decision-making of the LSTM:
    - Forget gate high → the network is preserving its memory
    - Forget gate low → the network is clearing memory (e.g., at word boundaries)
    - Input gate high → the network is writing new information
    - Output gate high → the network is using its memory for the current prediction

    Watching these gates across a sequence builds intuition for how LSTMs route
    information — something impossible with vanilla RNNs (which have no gates).
    """
    bos = len(unique_chars)
    vocab_size = len(unique_chars) + 1

    tokens = [bos] + [unique_chars.index(ch) for ch in sample_name] + [bos]
    seq_len = len(tokens) - 1
    chars = ['<BOS>'] + list(sample_name) + ['<EOS>']

    h = [Value(0.0) for _ in range(N_HIDDEN)]
    c = [Value(0.0) for _ in range(N_HIDDEN)]

    print(f"Gate dynamics for '{sample_name}':")
    print(f"{'Step':<5} {'Char':<6} {'Forget':>8} {'Input':>8} {'Output':>8} {'|Cell|':>8}")
    print("-" * 50)

    for pos in range(seq_len):
        x_onehot = [Value(1.0 if i == tokens[pos] else 0.0)
                    for i in range(vocab_size)]

        _, h, c, gate_acts = lstm_forward(x_onehot, h, c, params)

        # Compute mean activation across hidden dimensions for summary display.
        # Individual dimensions specialize (some forget dims track vowel patterns,
        # others track name length), but the mean reveals overall gate behavior.
        mean_f = sum(gate_acts['forget']) / N_HIDDEN
        mean_i = sum(gate_acts['input']) / N_HIDDEN
        mean_o = sum(gate_acts['output']) / N_HIDDEN
        cell_norm = math.sqrt(sum(v * v for v in gate_acts['cell']) / N_HIDDEN)

        print(f"  {pos:<3} {chars[pos]:<6} {mean_f:>8.4f} {mean_i:>8.4f} "
              f"{mean_o:>8.4f} {cell_norm:>8.4f}")

    print()

    # Interpretation guide
    print("Reading the gate dynamics:")
    print("  Forget ≈ 1.0 → preserving cell memory (default at init due to bias=1)")
    print("  Forget ≈ 0.0 → clearing cell memory (resetting context)")
    print("  Input  ≈ 1.0 → writing new information to cell state")
    print("  Input  ≈ 0.0 → ignoring current input")
    print("  Output ≈ 1.0 → exposing cell state to hidden output")
    print("  |Cell| grows → accumulating information over the sequence")
    print()

    # GRU comparison note
    print("Comparison to GRU (from micrornn.py):")
    print("  GRU merges forget + input into a single 'update gate' z_t:")
    print("    h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate")
    print("  When z_t ≈ 0: keep old state (= forget gate open + input gate closed)")
    print("  When z_t ≈ 1: use new state (= forget gate closed + input gate open)")
    print("  LSTM's separate gates allow forget-high AND input-high simultaneously,")
    print("  meaning it can ADD new info WITHOUT erasing old info. GRU cannot do this.")
    print()


# === MAIN ===

if __name__ == "__main__":
    # -- Load and prepare data --
    print("Loading data...")
    all_docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(all_docs)

    # Small training subset: 200 names, each seen ~4x in 800 steps.
    # Same subset size as micrornn.py for direct comparison.
    docs = all_docs[:TRAIN_SIZE]

    # Build vocabulary from all names (so we don't miss any characters)
    unique_chars = sorted(set(''.join(all_docs)))
    vocab_size = len(unique_chars) + 1  # +1 for BOS/EOS token

    print(f"Loaded {len(all_docs)} documents, training on {len(docs)}")
    print(f"Vocabulary size: {vocab_size} (characters + BOS token)\n")

    # === TRAIN LSTM ===
    params = init_lstm_params(vocab_size)
    final_loss = train_lstm(docs, unique_chars, params)

    # === INFERENCE ===
    print("=" * 60)
    print("GENERATED NAMES")
    print("=" * 60)
    print()
    samples = generate_names(params, unique_chars, num_samples=10)

    # === GATE DYNAMICS ===
    print("=" * 60)
    print("GATE DYNAMICS ANALYSIS")
    print("=" * 60)
    print()

    # Show gate activations for a few example names from training data.
    # Picking short and long names to contrast how the LSTM manages memory
    # across different sequence lengths.
    short_name = min(docs[:20], key=len)
    long_name = max(docs[:20], key=len)

    show_gate_dynamics(params, unique_chars, short_name)
    show_gate_dynamics(params, unique_chars, long_name)

    # === HOW LSTM SOLVES VANISHING GRADIENTS ===
    print("=" * 60)
    print("WHY LSTM SOLVES VANISHING GRADIENTS")
    print("=" * 60)
    print()
    print("The vanishing gradient problem in vanilla RNNs:")
    print("  ∂h_t/∂h_{t-1} = diag(tanh'(z)) · W_hh")
    print("  ∂h_T/∂h_0 = product of T such terms → exponential decay if ||W_hh|| < 1")
    print()
    print("LSTM's solution via the cell state gradient:")
    print("  ∂C_t/∂C_{t-1} = f_t  (just the forget gate, a scalar near 1)")
    print("  ∂C_T/∂C_0 = f_T · f_{T-1} · ... · f_1")
    print("  With forget bias = 1.0, each f_t ≈ 0.7-0.9, so the product")
    print("  decays slowly (polynomial) instead of exponentially.")
    print()
    print("The cell state acts as a 'conveyor belt': information placed on it")
    print("at step 0 can survive to step T with minimal degradation, because")
    print("the forget gate only needs to stay open (f_t ≈ 1) — no matrix")
    print("multiplication chain to cause exponential decay.")
    print()

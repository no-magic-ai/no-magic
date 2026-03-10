"""
Vision Transformer from first principles: an image is worth 16x16 words — treating
image patches as tokens and classifying with pure attention, no convolutions needed.
"""
# Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
# Image Recognition at Scale" (2020). https://arxiv.org/abs/2010.11929

# === TRADEOFFS ===
# + No convolutions: pure attention over spatial patches
# + Scales better than CNNs with more data and compute
# + Position embeddings learn 2D spatial structure from 1D ordering
# - Requires large datasets to outperform CNNs (less inductive bias)
# - Quadratic attention cost in number of patches
# - Patch size is a hard design choice: smaller = more tokens = slower
# WHEN TO USE: Image classification when you have large datasets,
#   want a unified transformer architecture, or need global receptive fields.
# WHEN NOT TO: Small datasets where CNN inductive biases help,
#   real-time applications where patch count makes attention too expensive.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Image and patch geometry
IMAGE_H = 6           # image height in pixels
IMAGE_W = 6           # image width in pixels
N_CHANNELS = 1        # grayscale (color would be 3, but adds compute without insight)
PATCH_SIZE = 3        # each patch is 3x3 pixels
NUM_PATCHES = (IMAGE_H // PATCH_SIZE) * (IMAGE_W // PATCH_SIZE)  # 4 patches
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * N_CHANNELS                 # 9 values per patch

# Transformer architecture
# Signpost: real ViT-Base uses embed_dim=768, 12 heads, 12 layers, 86M params.
# We use a tiny model that trains on CPU in minutes — the architecture is identical,
# only the scale differs. Our 6x6 images with 3x3 patches yield only 4 patches,
# keeping the attention matrix at 5x5 (with [CLS]) — tractable for scalar autograd.
EMBED_DIM = 8         # dimension of patch embeddings and transformer hidden state
N_HEADS = 2           # number of attention heads
HEAD_DIM = EMBED_DIM // N_HEADS  # 4 dimensions per head
N_LAYERS = 1          # number of transformer encoder blocks
MLP_DIM = EMBED_DIM * 2  # feedforward expansion (2x instead of standard 4x for speed)

# Sequence length: NUM_PATCHES + 1 for the [CLS] token
# Math: N = HW/P² = 6*6/3² = 4 patches, plus 1 [CLS] = 5 tokens total
# This keeps the attention matrix at 5×5 = 25 entries — fast even in scalar autograd.
SEQ_LEN = NUM_PATCHES + 1

# Training parameters
NUM_CLASSES = 4       # synthetic pattern categories
NUM_SAMPLES = 300     # training + test dataset size
TRAIN_FRAC = 0.8      # 80/20 train/test split
LEARNING_RATE = 0.005
BETA1 = 0.9           # Adam momentum decay
BETA2 = 0.999         # Adam variance decay
EPS_ADAM = 1e-8
NUM_STEPS = 200       # training iterations (SGD, batch_size=1)


# === SCALAR AUTOGRAD ENGINE ===

# This Value class follows the canonical interface from microgpt.py.
# See docs/autograd-interface.md for the full specification.

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Every forward operation stores its local derivative (dout/dinput) as a closure,
    then backward() replays the computation graph in reverse topological order,
    accumulating gradients via the chain rule: dL/dx = dL/dy * dy/dx.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
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

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        """Reverse-mode AD: topological sort then propagate gradients backward."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: dL/dchild += dL/dv * dv/dchild
                child.grad += local_grad * v.grad


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x (no bias).

    Math: y[i] = sum_j W[i,j] * x[j]
    This is the fundamental building block — every projection in the transformer
    (patch embedding, Q/K/V, MLP, classification head) is a linear transform.
    """
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def linear_with_bias(x: list[Value], w: list[list[Value]],
                     b: list[Value]) -> list[Value]:
    """Matrix-vector multiply with bias: y = W @ x + b.

    ViT uses biases in its linear layers, unlike some LLM architectures that drop
    them. The bias gives each output neuron a learnable offset — helpful for the
    classification head where different classes may have different base rates.
    """
    return [sum(w_row[j] * x[j] for j in range(len(x))) + b[i]
            for i, w_row in enumerate(w)]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow.

    Math: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    Translation invariance means subtracting a constant doesn't change the output,
    but it keeps exp() from returning inf for large logits (>~700).
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def layernorm(x: list[Value]) -> list[Value]:
    """Layer normalization: center and scale to unit variance.

    Math: LN(x) = (x - mu) / sqrt(var + eps)
    where mu = mean(x), var = mean((x - mu)^2)

    ViT uses LayerNorm (not RMSNorm) following the original transformer.
    The mean centering helps stabilize attention scores by removing the DC component
    from activations — important when patches from different image regions have
    very different brightness levels.

    Signpost: we omit the learnable affine parameters (gamma, beta) that production
    LayerNorm includes. With our tiny model, the normalization alone is sufficient.
    """
    n = len(x)
    mu = sum(xi.data for xi in x) / n
    # Subtract mean — this centers activations around zero
    centered = [xi - mu for xi in x]
    var = sum(c * c for c in centered) / n
    # The 1e-5 epsilon prevents division by zero when all inputs are identical
    scale = (var + 1e-5) ** -0.5
    return [c * scale for c in centered]


def safe_log(prob: Value) -> Value:
    """Clipped log for numerical stability — prevents log(0) = -inf.

    We clamp to 1e-10 but keep prob as the child node so gradients still flow
    through the computation graph. Without this, a zero probability early in
    training would produce -inf loss and NaN gradients, killing the entire run.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.1) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    std=0.1 is tuned for our tiny model. The ViT paper uses 0.02 for full-scale
    models, but with embed_dim=8, Xavier init gives std ≈ 1/sqrt(8) ≈ 0.35.
    We use 0.1 as a compromise — large enough for meaningful initial features,
    small enough to avoid exploding activations through the residual stream.
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_vector(n: int, std: float = 0.1) -> list[Value]:
    """Initialize a bias vector with Gaussian noise."""
    return [Value(random.gauss(0, std)) for _ in range(n)]


def init_parameters() -> dict:
    """Initialize all ViT parameters.

    Architecture overview (parameter flow):
        Image ∈ R^(6×6) → [patch_embed] → patches ∈ R^(4×8)
        Prepend [CLS] token → sequence ∈ R^(5×8)
        Add position embeddings → sequence ∈ R^(5×8)
        L × TransformerEncoder(LN → MSA → LN → MLP) → features ∈ R^(5×8)
        Extract [CLS] → LN → [cls_head] → logits ∈ R^4
    """
    params: dict = {}

    # --- Patch embedding ---
    # Projects each flattened patch (P²·C = 9 values) to embed_dim (8).
    # Math: Image ∈ R^(H×W×C) → Patches ∈ R^(N×P²C) → Embeddings ∈ R^(N×D)
    # where N = HW/P² = 4 patches, P²C = 9, D = 8
    params['patch_proj_w'] = make_matrix(EMBED_DIM, PATCH_DIM)
    params['patch_proj_b'] = make_vector(EMBED_DIM)

    # --- [CLS] token ---
    # A learnable vector prepended to the patch sequence. Through self-attention,
    # it aggregates information from all patches — like a "summary" that the
    # classification head reads. This avoids global average pooling and lets the
    # model learn what to aggregate.
    params['cls_token'] = make_vector(EMBED_DIM)

    # --- Position embeddings ---
    # Learnable 1D position embeddings for each slot in the sequence (CLS + patches).
    # Despite images being 2D, 1D ordering works because attention can learn spatial
    # relationships: position embeddings for adjacent patches in the same row will
    # become similar, as will patches in the same column. The model discovers 2D
    # structure from 1D indices — empirically matching hand-crafted 2D embeddings.
    #
    # With our 2×2 grid of 3×3 patches, position 1=top-left, 2=top-right,
    # 3=bottom-left, 4=bottom-right. The model must learn that positions 1,2 are
    # "top" and 3,4 are "bottom" — purely from gradient signal.
    params['pos_embed'] = make_matrix(SEQ_LEN, EMBED_DIM)

    # --- Transformer encoder blocks ---
    for layer_idx in range(N_LAYERS):
        prefix = f'layer{layer_idx}'

        # Multi-head self-attention projections
        # Q, K, V: [embed_dim, embed_dim] — project input to query/key/value spaces
        # Each head operates on a HEAD_DIM-sized slice of the full projection.
        params[f'{prefix}.attn_wq'] = make_matrix(EMBED_DIM, EMBED_DIM)
        params[f'{prefix}.attn_wk'] = make_matrix(EMBED_DIM, EMBED_DIM)
        params[f'{prefix}.attn_wv'] = make_matrix(EMBED_DIM, EMBED_DIM)
        params[f'{prefix}.attn_wo'] = make_matrix(EMBED_DIM, EMBED_DIM)

        # MLP: two-layer feedforward with 2x expansion
        # Standard ViT uses 4x, but 2x is sufficient for our 4-class task and
        # halves the MLP parameter count. The expansion gives the network a wider
        # "workspace" to transform attention output before projecting back.
        params[f'{prefix}.mlp_fc1_w'] = make_matrix(MLP_DIM, EMBED_DIM)
        params[f'{prefix}.mlp_fc1_b'] = make_vector(MLP_DIM)
        params[f'{prefix}.mlp_fc2_w'] = make_matrix(EMBED_DIM, MLP_DIM)
        params[f'{prefix}.mlp_fc2_b'] = make_vector(EMBED_DIM)

    # --- Classification head ---
    # Maps the [CLS] token's final representation to class logits.
    # Only the [CLS] token is used — patch tokens are discarded for classification.
    params['cls_head_w'] = make_matrix(NUM_CLASSES, EMBED_DIM)
    params['cls_head_b'] = make_vector(NUM_CLASSES)

    return params


# === SYNTHETIC DATASET ===

def generate_dataset(
    num_samples: int, image_h: int, image_w: int
) -> list[tuple[list[list[float]], int]]:
    """Generate a synthetic image classification dataset with 4 pattern classes.

    Classes encode spatial structure that attention must learn to detect:
      0: top-heavy  — bright pixels concentrated in upper half
      1: bottom-heavy — bright pixels concentrated in lower half
      2: left-heavy  — bright pixels concentrated in left half
      3: right-heavy — bright pixels concentrated in right half

    This is a minimal but non-trivial task for a ViT: the model must attend to
    patches in specific spatial regions and learn that position matters. A bag-of-patches
    model (no position embeddings) would fail because it can't distinguish top from bottom.

    Signpost: real ViT is trained on ImageNet (1.2M images, 1000 classes). Our synthetic
    data isolates the core mechanism — spatial attention over patches — without needing
    hours of training on real images.
    """
    dataset: list[tuple[list[list[float]], int]] = []

    for _ in range(num_samples):
        label = random.randint(0, NUM_CLASSES - 1)
        # Initialize with low-level noise (background)
        image = [[random.uniform(0.0, 0.3) for _ in range(image_w)]
                 for _ in range(image_h)]

        # Paint the signal region with bright pixels
        half_h = image_h // 2
        half_w = image_w // 2

        if label == 0:    # top-heavy
            for r in range(half_h):
                for c in range(image_w):
                    image[r][c] = random.uniform(0.7, 1.0)
        elif label == 1:  # bottom-heavy
            for r in range(half_h, image_h):
                for c in range(image_w):
                    image[r][c] = random.uniform(0.7, 1.0)
        elif label == 2:  # left-heavy
            for r in range(image_h):
                for c in range(half_w):
                    image[r][c] = random.uniform(0.7, 1.0)
        else:             # right-heavy
            for r in range(image_h):
                for c in range(half_w, image_w):
                    image[r][c] = random.uniform(0.7, 1.0)

        dataset.append((image, label))

    random.shuffle(dataset)
    return dataset


# === PATCH EMBEDDING ===

def image_to_patches(image: list[list[float]]) -> list[list[float]]:
    """Split an image into non-overlapping patches and flatten each.

    Math: Image ∈ R^(H×W) → N patches ∈ R^(P²) each
    where N = (H/P) × (W/P), and each patch is a flattened P×P block.

    Patches are extracted in raster order (left-to-right, top-to-bottom), which
    defines the 1D sequence ordering that position embeddings will augment.
    A 3×3 patch from a 6×6 image captures one quadrant — the transformer's
    job is to combine these local features into a global classification.

    This is the "tokenization" step for images: just as a text transformer converts
    words to token IDs, ViT converts image regions to patch vectors. The key insight
    from the paper: no convolutions are needed — a linear projection of raw pixels
    is sufficient when combined with enough data and attention.
    """
    patches: list[list[float]] = []
    rows_of_patches = len(image) // PATCH_SIZE
    cols_of_patches = len(image[0]) // PATCH_SIZE

    for pr in range(rows_of_patches):
        for pc in range(cols_of_patches):
            patch: list[float] = []
            for r in range(PATCH_SIZE):
                for c in range(PATCH_SIZE):
                    patch.append(image[pr * PATCH_SIZE + r][pc * PATCH_SIZE + c])
            patches.append(patch)

    return patches


def embed_patches(
    patches: list[list[float]], params: dict
) -> list[list[Value]]:
    """Project raw patches to embedding space and prepend [CLS] token.

    Pipeline:
      1. Linear project each patch: R^(P²C) → R^(D)  via patch_proj
      2. Prepend [CLS] token: sequence grows from N to N+1
      3. Add position embeddings: each token gets a learnable position vector

    The position embeddings are critical. Without them, the transformer sees a
    bag of patches with no spatial ordering — it couldn't distinguish "bright on top"
    from "bright on bottom." The 1D position indices implicitly encode 2D layout
    because patches are extracted in raster order: position 1=top-left, 2=top-right,
    3=bottom-left, 4=bottom-right. Attention learns these spatial relationships.
    """
    w = params['patch_proj_w']
    b = params['patch_proj_b']
    cls_token = params['cls_token']
    pos_embed = params['pos_embed']

    # Project each patch through the embedding layer
    embedded: list[list[Value]] = []
    for patch in patches:
        # Convert raw floats to Values for autograd tracking
        patch_vals = [Value(p) for p in patch]
        proj = linear_with_bias(patch_vals, w, b)
        embedded.append(proj)

    # Prepend [CLS] token — a learnable "summary" vector at position 0.
    # Through self-attention, [CLS] attends to all patches and accumulates
    # a global image representation. This is ViT's alternative to global
    # average pooling: instead of averaging all features, let attention
    # decide what matters.
    sequence = [[v for v in cls_token]] + embedded  # [CLS] at position 0

    # Add position embeddings: sequence[i] += pos_embed[i]
    for i in range(len(sequence)):
        sequence[i] = [tok + pos for tok, pos in zip(sequence[i], pos_embed[i])]

    return sequence


# === TRANSFORMER ENCODER ===

def multi_head_attention(
    sequence: list[list[Value]], params: dict, prefix: str
) -> list[list[Value]]:
    """Multi-head self-attention over the full sequence (no causal mask).

    Unlike GPT's causal attention (where each token only sees past tokens), ViT uses
    bidirectional attention: every patch attends to every other patch, including [CLS].
    This is because image classification doesn't have a sequential generation order —
    all patches exist simultaneously.

    Math per head h:
      Q_h = X @ W_q[:, h*d_h:(h+1)*d_h]   (queries)
      K_h = X @ W_k[:, h*d_h:(h+1)*d_h]   (keys)
      V_h = X @ W_v[:, h*d_h:(h+1)*d_h]   (values)
      A_h = softmax(Q_h @ K_h^T / sqrt(d_h))  (attention weights)
      O_h = A_h @ V_h                     (attention output)
    Then concatenate all heads and project: O = concat(O_1,...,O_H) @ W_o

    Multiple heads let the model attend to different aspects simultaneously:
    one head might focus on spatial neighbors, another on patches with similar
    brightness. With our 2 heads and 4 dims each, each head captures a different
    spatial relationship.
    """
    seq_len = len(sequence)
    wq = params[f'{prefix}.attn_wq']
    wk = params[f'{prefix}.attn_wk']
    wv = params[f'{prefix}.attn_wv']
    wo = params[f'{prefix}.attn_wo']

    # Project all tokens to Q, K, V
    all_q = [linear(tok, wq) for tok in sequence]
    all_k = [linear(tok, wk) for tok in sequence]
    all_v = [linear(tok, wv) for tok in sequence]

    # Process each head independently
    output_per_token: list[list[Value]] = [[] for _ in range(seq_len)]

    for head in range(N_HEADS):
        h_start = head * HEAD_DIM

        for i in range(seq_len):
            # Query for token i, this head's slice
            qi = all_q[i][h_start:h_start + HEAD_DIM]

            # Compute attention scores against all keys
            # score(i,j) = q_i · k_j / sqrt(d_head)
            # The 1/sqrt(d) scaling prevents dot products from growing with dimension,
            # which would push softmax into saturation (near-zero gradients).
            attn_logits: list[Value] = []
            for j in range(seq_len):
                kj = all_k[j][h_start:h_start + HEAD_DIM]
                score = sum(qi[d] * kj[d] for d in range(HEAD_DIM))
                attn_logits.append(score / (HEAD_DIM ** 0.5))

            attn_weights = softmax(attn_logits)

            # Weighted sum of values: output = sum_j attn(i,j) * v_j
            head_out: list[Value] = []
            for d in range(HEAD_DIM):
                val = sum(attn_weights[j] * all_v[j][h_start + d]
                          for j in range(seq_len))
                head_out.append(val)

            output_per_token[i].extend(head_out)

    # Project concatenated heads back to embed_dim
    projected = [linear(tok, wo) for tok in output_per_token]
    return projected


def transformer_block(
    sequence: list[list[Value]], params: dict, layer_idx: int
) -> list[list[Value]]:
    """One transformer encoder block with pre-norm residual connections.

    Architecture (pre-LN formulation):
      z' = MSA(LN(z)) + z       (attention sub-layer)
      z  = MLP(LN(z')) + z'      (feedforward sub-layer)

    Pre-norm vs post-norm: the original transformer (Vaswani 2017) applied LayerNorm
    after the residual connection (post-norm). ViT and most modern architectures use
    pre-norm because it stabilizes training — the residual stream stays unnormalized,
    which preserves gradient magnitude through deep networks. Post-norm can cause
    training instability without careful learning rate warmup.
    """
    prefix = f'layer{layer_idx}'
    seq_len = len(sequence)

    # --- Attention sub-layer with residual ---
    # Pre-norm: normalize before attention
    normed = [layernorm(tok) for tok in sequence]
    attn_out = multi_head_attention(normed, params, prefix)
    # Residual connection: add input back to attention output
    sequence = [[a + r for a, r in zip(attn_out[i], sequence[i])]
                for i in range(seq_len)]

    # --- MLP sub-layer with residual ---
    residual = sequence
    normed = [layernorm(tok) for tok in sequence]

    fc1_w = params[f'{prefix}.mlp_fc1_w']
    fc1_b = params[f'{prefix}.mlp_fc1_b']
    fc2_w = params[f'{prefix}.mlp_fc2_w']
    fc2_b = params[f'{prefix}.mlp_fc2_b']

    mlp_out: list[list[Value]] = []
    for tok in normed:
        # Expand: R^D → R^(2D) — gives network a wider workspace for computation
        h = linear_with_bias(tok, fc1_w, fc1_b)
        # GELU approximation via ReLU — production ViT uses GELU, but ReLU is
        # simpler and produces qualitatively identical results at this scale.
        h = [v.relu() for v in h]
        # Contract: R^(2D) → R^D — project back to residual stream width
        h = linear_with_bias(h, fc2_w, fc2_b)
        mlp_out.append(h)

    # Residual connection around MLP
    sequence = [[m + r for m, r in zip(mlp_out[i], residual[i])]
                for i in range(seq_len)]

    return sequence


# === VIT FORWARD PASS ===

def vit_forward(
    image: list[list[float]], params: dict
) -> list[Value]:
    """Full ViT forward pass: image → patches → transformer → class logits.

    Pipeline:
      1. Patchify: split 6x6 image into 3x3 patches (4 patches)
      2. Embed: linear project patches + prepend [CLS] + add position embeddings
      3. Encode: pass through L transformer blocks with bidirectional attention
      4. Classify: extract [CLS] token → LayerNorm → linear → logits

    The [CLS] token is the key architectural choice. It starts as a random learnable
    vector, but through self-attention in each layer, it reads information from all
    patch tokens. By the final layer, it contains a global image summary — which
    patches are bright, their spatial arrangement, and the resulting class. This is
    more flexible than global average pooling because attention weights are learned
    and content-dependent.

    How ViT relates to microgpt.py: both use the same transformer building blocks
    (multi-head attention, layernorm, MLP, residual connections). The differences:
    ViT uses bidirectional attention (no causal mask), patch embeddings instead of
    token embeddings, and a classification head instead of a language model head.
    """
    # Step 1-2: Patchify + embed + position + [CLS]
    patches = image_to_patches(image)
    sequence = embed_patches(patches, params)

    # Step 3: Transformer encoder stack
    for layer_idx in range(N_LAYERS):
        sequence = transformer_block(sequence, params, layer_idx)

    # Step 4: Classification head
    # Extract [CLS] token (position 0) — it has attended to all patches
    cls_representation = layernorm(sequence[0])
    logits = linear_with_bias(
        cls_representation,
        params['cls_head_w'],
        params['cls_head_b']
    )

    return logits


# === TRAINING LOOP ===

if __name__ == "__main__":
    start_time = time.time()

    # --- Generate synthetic data ---
    print("Generating synthetic dataset...")
    dataset = generate_dataset(NUM_SAMPLES, IMAGE_H, IMAGE_W)
    split_idx = int(len(dataset) * TRAIN_FRAC)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    class_names = ["top-heavy", "bottom-heavy", "left-heavy", "right-heavy"]
    print(f"  {len(train_data)} training samples, {len(test_data)} test samples")
    print(f"  {NUM_CLASSES} classes: {', '.join(class_names)}")
    print(f"  Image: {IMAGE_H}x{IMAGE_W}, Patch: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN} ({NUM_PATCHES} patches + 1 [CLS])\n")

    # --- Initialize model ---
    print("Initializing ViT parameters...")
    params = init_parameters()

    # Flatten all parameters for optimizer
    param_list: list[Value] = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, list) and isinstance(val[0], list):
            for row in val:
                param_list.extend(row)
        else:
            param_list.extend(val)

    print(f"  Parameters: {len(param_list):,}")
    print(f"  Architecture: embed_dim={EMBED_DIM}, heads={N_HEADS}, "
          f"layers={N_LAYERS}, mlp_dim={MLP_DIM}\n")

    # --- Adam optimizer state ---
    # Per-parameter momentum (m) and variance (v) running averages.
    # Adam adapts the learning rate for each parameter individually based on
    # gradient history — critical for transformers where different components
    # (attention weights, embeddings, MLP) can have very different gradient scales.
    adam_m = [0.0] * len(param_list)
    adam_v = [0.0] * len(param_list)

    # --- Training ---
    # SGD with batch_size=1: each step processes one image. This is noisier than
    # mini-batch SGD but faster per step in scalar autograd (no batch graph overhead).
    # The noise acts as implicit regularization, preventing overfitting on our tiny dataset.
    print("Training...")
    for step in range(NUM_STEPS):
        # Sample one training example (stochastic gradient descent)
        image, label = random.choice(train_data)

        # Forward pass
        logits = vit_forward(image, params)
        probs = softmax(logits)

        # Cross-entropy loss: -log(p(correct_class))
        # This pushes the model to assign high probability to the true class.
        loss = -safe_log(probs[label])

        # --- Backward pass ---
        loss.backward()

        # --- Adam optimizer step ---
        # Linear learning rate decay: prevents overshooting as the model converges.
        # Without decay, the fixed step size can cause the optimizer to bounce
        # around the minimum rather than converging.
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, param in enumerate(param_list):
            # Adam update:
            #   m_t = β1*m_{t-1} + (1-β1)*g        (momentum — smoothed gradient)
            #   v_t = β2*v_{t-1} + (1-β2)*g²       (RMS — adaptive scaling)
            #   θ -= lr * m_hat / (sqrt(v_hat) + ε) (bias-corrected update)
            adam_m[i] = BETA1 * adam_m[i] + (1 - BETA1) * param.grad
            adam_v[i] = BETA2 * adam_v[i] + (1 - BETA2) * param.grad ** 2

            # Bias correction: m and v are initialized to 0, so early estimates are
            # biased toward zero. Dividing by (1 - β^t) compensates.
            m_hat = adam_m[i] / (1 - BETA1 ** (step + 1))
            v_hat = adam_v[i] / (1 - BETA2 ** (step + 1))

            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        # Progress reporting (no extra forward pass — just report the training loss)
        if step == 0 or (step + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  step {step + 1:>4}/{NUM_STEPS} | "
                  f"loss: {loss.data:.4f} | time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s. Final loss: {loss.data:.4f}\n")

    # === INFERENCE ===

    print("=" * 60)
    print("INFERENCE: Evaluating on held-out test set")
    print("=" * 60)

    # --- Test set evaluation ---
    correct = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    for image, label in test_data:
        logits = vit_forward(image, params)
        probs = softmax(logits)
        pred = max(range(NUM_CLASSES), key=lambda c: probs[c].data)

        per_class_total[label] += 1
        confusion[label][pred] += 1

        if pred == label:
            correct += 1
            per_class_correct[label] += 1

    total_acc = correct / len(test_data) * 100
    print(f"\nOverall test accuracy: {correct}/{len(test_data)} ({total_acc:.1f}%)\n")

    # Per-class breakdown
    print("Per-class accuracy:")
    for c in range(NUM_CLASSES):
        if per_class_total[c] > 0:
            cls_acc = per_class_correct[c] / per_class_total[c] * 100
            print(f"  {class_names[c]:>14s}: "
                  f"{per_class_correct[c]}/{per_class_total[c]} ({cls_acc:.1f}%)")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "            " + "".join(f"{name:>14s}" for name in class_names)
    print(header)
    for r in range(NUM_CLASSES):
        row_str = f"{class_names[r]:>12s}"
        for c_val in range(NUM_CLASSES):
            row_str += f"{confusion[r][c_val]:>14d}"
        print(row_str)

    # --- Sample predictions ---
    print("\nSample predictions on test images:")
    num_show = min(10, len(test_data))
    for idx in range(num_show):
        image, label = test_data[idx]
        logits = vit_forward(image, params)
        probs = softmax(logits)
        pred = max(range(NUM_CLASSES), key=lambda c: probs[c].data)
        conf = probs[pred].data * 100
        status = "CORRECT" if pred == label else "WRONG"
        print(f"  [{status:>7s}] true={class_names[label]:>12s}, "
              f"pred={class_names[pred]:>12s} ({conf:.1f}%)")

    # === ATTENTION ANALYSIS ===

    # Show what the [CLS] token attends to in the first layer for one example per class.
    # This reveals whether the model has learned to focus on the relevant spatial region.
    # For a "top-heavy" image, we expect [CLS] to attend more to patches 1-2 (top row).
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS: What does [CLS] attend to?")
    print("=" * 60)
    print("(Patch layout: TL=top-left, TR=top-right, BL=bottom-left, BR=bottom-right)")

    patch_labels = ["[CLS]", "TL", "TR", "BL", "BR"]

    for target_class in range(NUM_CLASSES):
        for image, label in test_data:
            if label == target_class:
                # Run embedding + first-layer attention manually to extract weights
                patches = image_to_patches(image)
                sequence = embed_patches(patches, params)

                # Compute first layer attention for [CLS] query
                normed = [layernorm(tok) for tok in sequence]
                prefix = 'layer0'
                wq = params[f'{prefix}.attn_wq']
                wk = params[f'{prefix}.attn_wk']

                cls_q = linear(normed[0], wq)
                all_k = [linear(tok, wk) for tok in normed]

                # Average attention across heads for interpretability
                avg_attn: list[float] = []
                for j in range(SEQ_LEN):
                    score = 0.0
                    for head in range(N_HEADS):
                        h_start = head * HEAD_DIM
                        qi = cls_q[h_start:h_start + HEAD_DIM]
                        kj = all_k[j][h_start:h_start + HEAD_DIM]
                        dot = sum(qi[d].data * kj[d].data for d in range(HEAD_DIM))
                        score += dot / (HEAD_DIM ** 0.5)
                    avg_attn.append(score / N_HEADS)

                # Softmax over raw scores for visualization
                max_s = max(avg_attn)
                exp_s = [math.exp(s - max_s) for s in avg_attn]
                total_s = sum(exp_s)
                attn_probs = [e / total_s for e in exp_s]

                print(f"\n  {class_names[target_class]:>14s} → ", end="")
                for j in range(SEQ_LEN):
                    bar = "#" * int(attn_probs[j] * 20)
                    print(f" {patch_labels[j]}:{attn_probs[j]:.2f}", end="")
                print()
                break

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s")

"""
Rank-One Model Editing (ROME) from first principles: a fact that a trained decoder recalls is
rewritten by one closed-form rank-one update of a single MLP weight matrix, solved from a key,
a value and a key covariance -- no gradient step on any weight, no lookup table, no text patch.
"""
# Reference: Meng, Bau, Andonian & Belinkov, "Locating and Editing Factual Associations in GPT",
# NeurIPS 2022, https://arxiv.org/abs/2202.05262 -- Sec. 2 (Eq. 1, causal tracing), Sec. 3.1
# (Eq. 2-4: the MLP output matrix W_proj as a linear key-value memory, the key k*, the value
# v* and the rank-one insertion), App. A (Eq. 5-17: the constrained least-squares derivation),
# App. B.1 (trace noise) and App. E.5 (edit constants). The paper edits GPT-2 XL (1.5B
# parameters) at layer 18; this script edits a 3,936-parameter toy decoder it trains itself.

# === TRADEOFFS ===
# + One fact changes with one closed-form matrix update: no fine-tuning loop over weights
# + The covariance C = K K^T makes the change nearly invisible to every other key, so
#   neighboring facts survive (measured below with the C = I control)
# + The edit is inspectable: Delta W = Lambda u^T is rank one, so it has one "address" (u)
#   and one "content" (Lambda)
# - One fact per edit; edits are directional (s -> o is stored apart from o -> s, Sec. 3.7)
# - A value v* fitted on a few contexts may transfer only partly to unseen prompts
#   (measured below: paraphrase success is lower than efficacy)
# - Needs a trained model, an estimate of C and a choice of layer; none of this is free at scale
# WHEN TO USE: Correcting or inserting a single (subject, relation, object) association in a
#   trained network while leaving its other behavior intact.
# WHEN NOT TO: Thousands of simultaneous edits (MEMIT, a different paper), changing skills or
#   style rather than one stored association, or when retraining on corrected data is cheap.

# === SOURCE ASSUMPTIONS AND DEVIATIONS FROM THE PAPER ===
# This is a pedagogical adaptation of the mechanism, not a reproduction of the paper's results.
# Every deviation below is deliberate and is repeated where it matters in the code:
#  1. Toy scale: 24 invented subjects, 8 invented cities, 4 categories, a 3,936-parameter
#     decoder trained here from random weights. No pretrained LM, no Wikipedia, no zsRE or
#     COUNTERFACT, and no paper-scale claim of any kind. Invented names avoid real-world facts.
#  2. Single-token subjects and no attention before the MLP: the "last subject token" is simply
#     the subject position, and a key depends only on (subject token, position). The paper's
#     multi-token subjects with attention-assembled keys are not exercised; prefix variation
#     reaches the key only through the position embedding.
#  3. No layer normalization (gamma in Eq. 1 and Eq. 3 is omitted), one MLP block, no biases.
#  4. The softmax is masked to the answer type (8 cities or 4 categories) instead of running
#     over the whole vocabulary.
#  5. The information flow subject -> MLP at the subject token -> attention at the last token
#     is IMPOSED by the architecture. The causal trace below confirms that flow; it does not
#     discover it, unlike the paper's Sec. 2 experiments. The same holds for the wrong-token
#     control: its paraphrase score of 0 is guaranteed by construction, so it illustrates the
#     architecture and is not evidence for the paper's claim that the subject site matters.
#  6. C is estimated from keys at every position of sampled training-distribution prompts, not
#     from Wikipedia (the paper's "every token" sampling, App. E.5, is kept).
#  7. Key prefixes are 0-3 random filler tokens (N = 20) instead of model-generated text. The
#     value search uses Adam lr 0.1, L2 decay on z added to the gradient only (not part of L(z)
#     or of the early stop), <= 60 steps, early stop L(z) < 0.05 (paper: lr 0.5, <= 20 steps),
#     because the toy's embedding scale differs. The KL factor lambda = 100 is the number App.
#     E.5 reports and was not tuned. The authors' released code differs from the paper text
#     (pinned commit 0874014cd9837e4365f3e6f3c71400ef11509e04, hparams/ROME/gpt2-xl.json and
#     rome/compute_v.py): kl_factor 0.0625, v_lr 0.5, 20 steps, a perturbation delta = z - z0
#     optimised instead of z, weight decay inside the loss on ||delta||/||z0||, and a norm clamp
#     at 4 ||z0||. This script follows the paper text, not the released code.
#  8. Evaluation uses fixed small prompt sets (20 and 8 prefixes) and the paper's success
#     comparisons ES / PS / NS (Sec. 3.2-3.4); no magnitude scores (EM/PM/NM), no fluency or
#     consistency, no generation. Neighbors share the old city by construction.
#  9. Six edits are made one at a time, each on a fresh copy of W_proj; no sequential or batch
#     editing.
# 10. Seed sensitivity: the numbers printed by this file are one realization (seed 42). In a
#     design study with other seeds, neighborhood, locality and essence stayed at 1.000, but
#     efficacy met the 0.95 threshold on only 6 of 17 seeds (mean 0.885). Changing the seed,
#     the number of random draws or the summation order changes which facts edit cleanly.
# 11. Interpreter note: since CPython 3.12, built-in sum() over floats uses compensated
#     summation, and nearly all arithmetic here goes through sum(). The reference output was
#     produced with CPython 3.12.8; CPython 3.10/3.11 print different low-order digits and,
#     given point 10, could in principle land on different edit outcomes.

from __future__ import annotations

import math
import random
import time

random.seed(42)
# Wall-clock time is read here and printed ONLY on lines starting with "[time]", so every
# other output line is deterministic and can be compared byte for byte between runs.
WALL_START = time.perf_counter()

# === CONSTANTS AND HYPERPARAMETERS ===

# Model size. D is the residual-stream width and H the MLP hidden width (the paper's key
# dimension). W_proj is D x H = 16 x 64: the matrix ROME edits.
D = 16
H = 64

# Synthetic world: every subject has one city and one category.
NUM_SUBJECTS = 24
NUM_CITIES = 8
NUM_CATEGORIES = 4
NUM_FILLERS = 8
MAX_PREFIX = 3  # prompts start with 0-3 filler tokens, so the subject sits at position 0-3
NUM_POS = MAX_PREFIX + 2  # longest prompt: 3 fillers + subject + template
NUM_TEMPLATES = 4  # T0..T3 ask for the city; the separate ESS template asks for the category
NUM_ANSWERS = NUM_CITIES + NUM_CATEGORIES

# Invented names: the facts are synthetic and make no claim about the real world.
SUBJECT_NAMES = [
    "bavo", "kelt", "morn", "sird", "tavi", "wexo", "yulm", "zarn", "doph", "felk", "gruv", "hisk",
    "jorp", "lamb", "nask", "ovik", "pelm", "quor", "ristu", "silv", "thom", "ulga", "vern", "wick",
]
CITY_NAMES = ["Alda", "Brenn", "Cavo", "Dunmar", "Elsk", "Fyra", "Gorv", "Halen"]
CATEGORY_NAMES = ["tower", "bridge", "garden", "harbor"]
FILLER_NAMES = ["the", "old", "big", "a", "one", "that", "new", "this"]
TEMPLATE_NAMES = ["sits-in", "stands-in", "lies-in", "is-found-in", "is-a"]

# Token ids: [subjects | fillers | T0..T3 | ESS]. Answers live in a separate output space:
# ids 0..7 are cities and 8..11 categories, read out by the unembedding U.
SUBJECT_BASE = 0
FILLER_BASE = NUM_SUBJECTS
TEMPLATE_BASE = FILLER_BASE + NUM_FILLERS
ESS = TEMPLATE_BASE + NUM_TEMPLATES  # the "essence" prompt "<subject> is-a" (paper: p')
NUM_TOKENS = ESS + 1
CITY_CANDIDATES = list(range(NUM_CITIES))
CATEGORY_CANDIDATES = list(range(NUM_CITIES, NUM_ANSWERS))

# Training (tuned only for training loss and recall, before any edit was evaluated)
LEARNING_RATE = 0.01
BATCH_SIZE = 24
TRAIN_STEPS = 1500
INIT_EMB = 0.5
INIT_POS = 0.1
INIT_MLP_IN = 0.25
INIT_MLP_OUT = 0.1
INIT_ATTN = 0.25
INIT_UNEMB = 0.25
LR_FINAL_FRACTION = 0.01  # linear decay to 1% of LEARNING_RATE

# Edit constants (frozen before the first edit was run; see deviation 7)
NUM_EDIT_CONTEXTS = 20  # N in Eq. 3 and Eq. 4 (paper: 20 generated texts)
VALUE_LR = 0.1  # paper: 0.5
VALUE_WEIGHT_DECAY = 1.5e-3  # paper value, applied as L2 on z in the gradient only
VALUE_MAX_STEPS = 60  # paper: 20
VALUE_EARLY_STOP = 0.05  # paper value: stop once L(z) < 5e-2
KL_FACTOR = 100.0  # lambda in Eq. 4, the paper-reported 1e2; the scale is not toy-tuned
COV_PROMPTS = 1500  # prompts whose keys at every position estimate C (paper: 100,000 tokens)
NUM_EVAL_CONTEXTS = 20
NUM_LOCALITY_CONTEXTS = 8
TRACE_NOISE_SAMPLES = 6
# Every 4th subject is edited; these six never train the (subject, T3) prompt, so T3 is an
# unseen paraphrase of the fact for them.
EDIT_SUBJECTS = [s for s in range(NUM_SUBJECTS) if s % 4 == 0]
PARAM_NAMES = ["E", "P", "Wfc", "Wproj", "Wq", "Wk", "Wv", "Wo", "U"]

Vector = list[float]
Matrix = list[list[float]]
Network = dict[str, Matrix]
Prompt = list[int]
ForwardCache = dict[str, list]  # every cached value is a list (tokens, vectors or matrices)


def stream(offset: int) -> random.Random:
    """A separate random stream per purpose (data, init, batches, covariance, edit contexts,
    evaluation, trace noise). Code that is reordered without changing how many numbers a
    purpose draws therefore cannot shift any other purpose's numbers."""
    return random.Random(42 * 1000 + offset)


# === DATA: SYNTHETIC FACTS AND PROMPTS ===


def build_facts() -> tuple[list[int], list[int]]:
    """Assign each subject a city (3 subjects per city) and a category (6 per category) by a
    seeded shuffle. Sharing a city is what makes "neighborhood" prompts exist: after editing
    one subject's city, the other two subjects of the old city must keep it."""
    rng = stream(1)
    cities = [c for c in range(NUM_CITIES) for _ in range(NUM_SUBJECTS // NUM_CITIES)]
    cats = [c for c in range(NUM_CATEGORIES) for _ in range(NUM_SUBJECTS // NUM_CATEGORIES)]
    rng.shuffle(cities)
    rng.shuffle(cats)
    return cities, cats


def trained_template_ids(subject: int) -> list[int]:
    """Template indices trained for a subject. Edit subjects never train (subject, T3)."""
    return [0, 1, 2] if subject in EDIT_SUBJECTS else [0, 1, 2, 3]


def make_prompt(prefix: Prompt, subject: int, template: int | str) -> tuple[Prompt, list[int], int]:
    """Build [fillers..., subject, template]; the next token is the answer.

    Returns (tokens, answer candidates, subject position). template 0..3 asks for the city;
    template "ess" asks for the category. The candidate list implements deviation 4: the
    softmax only ranks answers of the right type."""
    if template == "ess":
        return prefix + [SUBJECT_BASE + subject, ESS], CATEGORY_CANDIDATES, len(prefix)
    return prefix + [SUBJECT_BASE + subject, TEMPLATE_BASE + template], CITY_CANDIDATES, len(prefix)


def sample_prefix(rng: random.Random) -> Prompt:
    """0-3 random filler tokens. They move the subject to a different position, which is the
    only way context changes the subject's key here (deviation 2)."""
    length = rng.randint(0, MAX_PREFIX)
    return [FILLER_BASE + rng.randrange(NUM_FILLERS) for _ in range(length)]


# === MODEL DEFINITION (manual forward and backward) ===
# Plain-list linear algebra. The summation order inside these helpers is part of the reference
# output (deviation 11), so they are written once and reused everywhere.


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix: Matrix, vec: Vector) -> Vector:
    """matrix @ vec, one dot product per row."""
    return [sum(x * y for x, y in zip(row, vec)) for row in matrix]


def mat_t_vec(matrix: Matrix, vec: Vector) -> Vector:
    """matrix^T @ vec, accumulated row by row (the backward pass of matvec)."""
    out = [0.0] * len(matrix[0])
    for row, scale in zip(matrix, vec):
        if scale != 0.0:
            out = [o + scale * r for o, r in zip(out, row)]
    return out


def add_outer(grad: Matrix, left: Vector, right: Vector) -> None:
    """grad += left right^T in place: the weight gradient of y = W x is dy x^T."""
    for i, scale in enumerate(left):
        if scale != 0.0:
            grad[i] = [g + scale * r for g, r in zip(grad[i], right)]


def softmax(logits: Vector) -> Vector:
    # Subtracting the max leaves the result unchanged and keeps exp() from overflowing.
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def init_net() -> Network:
    """Gaussian initialisation of the nine weight tensors (3,936 numbers, no biases).

    E: token embeddings, P: position embeddings, Wfc (H x D) and Wproj (D x H): the MLP,
    Wq/Wk/Wv/Wo: one attention head, U: unembedding for the 12 answers."""
    rng = stream(2)

    def mat(rows: int, cols: int, std: float) -> Matrix:
        return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]

    return {
        "E": mat(NUM_TOKENS, D, INIT_EMB),
        "P": mat(NUM_POS, D, INIT_POS),
        "Wfc": mat(H, D, INIT_MLP_IN),
        "Wproj": mat(D, H, INIT_MLP_OUT),
        "Wq": mat(D, D, INIT_ATTN),
        "Wk": mat(D, D, INIT_ATTN),
        "Wv": mat(D, D, INIT_ATTN),
        "Wo": mat(D, D, INIT_ATTN),
        "U": mat(NUM_ANSWERS, D, INIT_UNEMB),
    }


def zeros_like(net: Network) -> Network:
    return {name: [[0.0] * len(row) for row in net[name]] for name in PARAM_NAMES}


def forward(
    net: Network,
    tokens: Prompt,
    cand: list[int],
    patch: dict[tuple[int, str], Vector] | None = None,
    noise: dict[int, Vector] | None = None,
) -> ForwardCache:
    """Run one prompt and cache every intermediate state.

    The block is the paper's Eq. 1 without layer norm (deviation 3) and with the MLP placed
    before attention:
        h0_i = E[x_i] + P[i]                      residual stream entering the block
        k_i  = ReLU(W_fc h0_i)                    the MLP "key" (Eq. 3 reads it here)
        m_i  = W_proj k_i                         the MLP output, the memory's "value"
        h1_i = h0_i + m_i                         residual add
        out  = h1_T + W_o attn(q = W_q h1_T, keys W_k h1_i, values W_v h1_i)
        logits = U out  (answer-type candidates only)
    Nothing mixes positions before the MLP, so position i's key depends only on (x_i, i). The
    only route from the subject to the answer is the last token's attention reading h1 at the
    subject position: the paper's "early site, then attention at the last token" picture, here
    built in rather than discovered (deviation 5).

    `patch` maps (position, state) to a replacement vector, state in 'h0', 'm', 'h1' or
    'attn' (last position only). Patching 'm' is how Eq. 4 injects a candidate value z, and
    how the causal trace restores a clean state. `noise` adds a vector to h0 at a position."""
    length = len(tokens)
    h0s, keys, ms, h1s, pre = [], [], [], [], []
    for i, tok in enumerate(tokens):
        x = [a + b for a, b in zip(net["E"][tok], net["P"][i])]
        if noise and i in noise:
            x = [a + b for a, b in zip(x, noise[i])]
        if patch and (i, "h0") in patch:
            x = list(patch[(i, "h0")])
        u = matvec(net["Wfc"], x)
        k = [v if v > 0.0 else 0.0 for v in u]  # ReLU: sigma in Eq. 1 (GPT-2 uses GELU)
        m = matvec(net["Wproj"], k)  # W_proj k: one lookup in the linear memory W K ~= V
        if patch and (i, "m") in patch:
            m = list(patch[(i, "m")])
        h1 = [a + b for a, b in zip(x, m)]
        if patch and (i, "h1") in patch:
            h1 = list(patch[(i, "h1")])
        h0s.append(x)
        pre.append(u)
        keys.append(k)
        ms.append(m)
        h1s.append(h1)
    # One attention head, queried from the LAST position only: it is the sole place where
    # information moves between positions.
    q = matvec(net["Wq"], h1s[-1])
    ks = [matvec(net["Wk"], h) for h in h1s]
    vs = [matvec(net["Wv"], h) for h in h1s]
    scale = 1.0 / math.sqrt(D)
    alpha = softmax([dot(q, kk) * scale for kk in ks])
    ctx = [0.0] * D
    for a, v in zip(alpha, vs):
        ctx = [c + a * x for c, x in zip(ctx, v)]
    attn = matvec(net["Wo"], ctx)
    if patch and (length - 1, "attn") in patch:
        attn = list(patch[(length - 1, "attn")])
    out = [a + b for a, b in zip(h1s[-1], attn)]
    probs = softmax([dot(net["U"][j], out) for j in cand])
    return {
        "tokens": tokens, "cand": cand, "h0": h0s, "pre": pre, "k": keys, "m": ms, "h1": h1s,
        "q": q, "K": ks, "V": vs, "alpha": alpha, "ctx": ctx, "attn": attn, "out": out,
        "probs": probs,
    }


def backward(
    net: Network, cache: ForwardCache, dlogits: Vector, grads: Network | None = None
) -> Matrix:
    """Backpropagate d(loss)/d(logits) through the cached forward pass by hand.

    Returns d(loss)/d(h1_i) for every position i. Because h1_i = h0_i + m_i, this is also
    d(loss)/d(m_i): exactly the gradient Eq. 4 needs when m_i is replaced by a free vector z.
    When `grads` is given, parameter gradients are accumulated into it (training only).
    These hand-written gradients were compared with central finite differences during the
    design of this script (parameters at initialization, and dL/dz at an unsaturated z; worst
    scaled error 3.9e-9). The check is not repeated here, to keep the run short."""
    length = len(cache["tokens"])
    out = cache["out"]
    # logits_j = U_j . out  ->  dU_j += dlogit_j out,  d_out += dlogit_j U_j
    d_out = [0.0] * D
    for dj, j in zip(dlogits, cache["cand"]):
        if grads is not None:
            grads["U"][j] = [g + dj * o for g, o in zip(grads["U"][j], out)]
        d_out = [a + dj * b for a, b in zip(d_out, net["U"][j])]
    # out = h1_T + attn: the residual passes d_out straight to the last position.
    dh1 = [[0.0] * D for _ in range(length)]
    dh1[-1] = list(d_out)
    if grads is not None:
        add_outer(grads["Wo"], d_out, cache["ctx"])
    d_ctx = mat_t_vec(net["Wo"], d_out)
    alpha, vs, ks, q = cache["alpha"], cache["V"], cache["K"], cache["q"]
    # Softmax backward: d score_i = alpha_i (d alpha_i - sum_j alpha_j d alpha_j)
    d_alpha = [dot(d_ctx, v) for v in vs]
    inner = sum(a * d for a, d in zip(alpha, d_alpha))
    d_scores = [a * (d - inner) for a, d in zip(alpha, d_alpha)]
    scale = 1.0 / math.sqrt(D)
    dq = [0.0] * D
    for i in range(length):
        d_v = [alpha[i] * x for x in d_ctx]  # ctx = sum_i alpha_i v_i
        d_k = [d_scores[i] * scale * x for x in q]  # score_i = q . k_i / sqrt(D)
        dq = [a + d_scores[i] * scale * b for a, b in zip(dq, ks[i])]
        if grads is not None:
            add_outer(grads["Wv"], d_v, cache["h1"][i])
            add_outer(grads["Wk"], d_k, cache["h1"][i])
        back_v = mat_t_vec(net["Wv"], d_v)
        back_k = mat_t_vec(net["Wk"], d_k)
        dh1[i] = [a + b + c for a, b, c in zip(dh1[i], back_v, back_k)]
    if grads is not None:
        add_outer(grads["Wq"], dq, cache["h1"][-1])
    dh1[-1] = [a + b for a, b in zip(dh1[-1], mat_t_vec(net["Wq"], dq))]
    if grads is not None:
        # Through the MLP at each position: m = W_proj k, k = ReLU(u), u = W_fc h0, and
        # h1 = h0 + m sends dh1 to h0 both directly and through the MLP.
        for i, tok in enumerate(cache["tokens"]):
            d_m = dh1[i]
            add_outer(grads["Wproj"], d_m, cache["k"][i])
            d_k = mat_t_vec(net["Wproj"], d_m)
            d_u = [d if u > 0.0 else 0.0 for d, u in zip(d_k, cache["pre"][i])]
            add_outer(grads["Wfc"], d_u, cache["h0"][i])
            d_h0 = [a + b for a, b in zip(dh1[i], mat_t_vec(net["Wfc"], d_u))]
            grads["E"][tok] = [g + d for g, d in zip(grads["E"][tok], d_h0)]
            grads["P"][i] = [g + d for g, d in zip(grads["P"][i], d_h0)]
    return dh1


def ce_loss_and_dlogits(probs: Vector, target_answer: int, cand: list[int]) -> tuple[float, Vector]:
    """Cross-entropy -log p_target and its gradient p - onehot(target) over the candidates."""
    t = cand.index(target_answer)
    loss = -math.log(max(probs[t], 1e-300))
    d = list(probs)
    d[t] -= 1.0
    return loss, d


# === TRAINING ===


def adam_step(
    net: Network,
    grads: Network,
    state: dict[str, Network],
    step: int,
    learning_rate: float,
    scale: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """One Adam update of every weight. `scale` = 1 / batch size turns summed gradients into
    a batch mean. m and v are the first and second moment estimates, bias-corrected."""
    for name in PARAM_NAMES:
        for r, row in enumerate(net[name]):
            g_row = grads[name][r]
            m_row, v_row = state["m"][name][r], state["v"][name][r]
            new_m, new_v, new_row = [], [], []
            for w, g, m, v in zip(row, g_row, m_row, v_row):
                g = g * scale
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                mh = m / (1 - b1 ** step)
                vh = v / (1 - b2 ** step)
                new_m.append(m)
                new_v.append(v)
                new_row.append(w - learning_rate * mh / (math.sqrt(vh) + eps))
            net[name][r] = new_row
            state["m"][name][r] = new_m
            state["v"][name][r] = new_v


def training_items() -> list[tuple[int, int | str]]:
    """The training grid: (subject, template) for every trained city template, plus
    (subject, ESS) for the category. 114 items; edit subjects lack (subject, T3)."""
    items: list[tuple[int, int | str]] = []
    for s in range(NUM_SUBJECTS):
        for t in trained_template_ids(s):
            items.append((s, t))
        items.append((s, "ess"))
    return items


def target_answer(subject: int, template: int | str, cities: list[int], cats: list[int]) -> int:
    return cities[subject] if template != "ess" else NUM_CITIES + cats[subject]


def train(net: Network, cities: list[int], cats: list[int]) -> Vector:
    """Train every weight from random initialisation with Adam on minibatches of fresh prompts.

    The model must genuinely learn the facts first: ROME edits what a network has stored, so
    an untrained or hand-written network would make the edit meaningless."""
    rng = stream(3)
    items = training_items()
    state = {"m": zeros_like(net), "v": zeros_like(net)}
    history = []
    for step in range(1, TRAIN_STEPS + 1):
        grads = zeros_like(net)
        batch_loss = 0.0
        for _ in range(BATCH_SIZE):
            s, t = items[rng.randrange(len(items))]
            tokens, cand, _ = make_prompt(sample_prefix(rng), s, t)
            cache = forward(net, tokens, cand)
            loss, dl = ce_loss_and_dlogits(cache["probs"], target_answer(s, t, cities, cats), cand)
            batch_loss += loss
            backward(net, cache, dl, grads)
        # Linear decay from LEARNING_RATE to LR_FINAL_FRACTION of it over the run.
        learning_rate = LEARNING_RATE * (1.0 - (1.0 - LR_FINAL_FRACTION) * (step - 1) / TRAIN_STEPS)
        adam_step(net, grads, state, step, learning_rate, 1.0 / BATCH_SIZE)
        history.append(batch_loss / BATCH_SIZE)
        if step % 100 == 0 or step == 1:
            tail = history[-50:]
            print("train step %4d  batch NLL %.4f  (mean last <=50: %.4f)" % (
                step, history[-1], sum(tail) / len(tail)), flush=True)
    return history


# === INFERENCE: PREDICTION HELPERS ===


def predict(
    net: Network,
    prefix: Prompt,
    subject: int,
    template: int | str,
    patch: dict[tuple[int, str], Vector] | None = None,
) -> tuple[list[int], Vector]:
    """Answer candidates and their probabilities for one prompt. Every evaluation below goes
    through this function, i.e. through the network's weights only: no lookup of the facts
    and no knowledge of the edit target o*."""
    tokens, cand, _ = make_prompt(prefix, subject, template)
    return cand, forward(net, tokens, cand, patch=patch)["probs"]


def argmax_answer(cand: list[int], probs: Vector) -> int:
    best = max(range(len(cand)), key=lambda i: probs[i])
    return cand[best]


def prob_of(cand: list[int], probs: Vector, answer: int) -> float:
    return probs[cand.index(answer)]


def training_grid_accuracy(
    net: Network, cities: list[int], cats: list[int], contexts: list[Prompt]
) -> tuple[int, int]:
    """Exact recall (argmax) on every trained (subject, template) at the given prefixes."""
    ok = total = 0
    for s in range(NUM_SUBJECTS):
        for t in trained_template_ids(s) + ["ess"]:
            for prefix in contexts:
                cand, probs = predict(net, prefix, s, t)
                ok += argmax_answer(cand, probs) == target_answer(s, t, cities, cats)
                total += 1
    return ok, total


# === INFERENCE: ROME STEP 2 -- THE VALUE v* (Eq. 4) ===
# Paper Eq. 4 (Sec. 3.1, Step 2), with G(m_i := z) the network whose MLP output at the subject
# position i is replaced by a free vector z:
#   v* = argmin_z  (1/N) sum_j -log P_G(m_i:=z)[o* | x_j + p]                      (4a)
#                  + lambda * D_KL( P_G(m_i':=z)[x | p'] || P_G[x | p'] )            (4b)
# (4a) makes the new object o* likely after every sampled prefix x_j. (4b), "essence drift",
# keeps the subject's other properties: p' is "<subject> is-a" (our ESS prompt), whose
# category distribution must not move. Printed Eq. 4 has no scalar on the KL term; App. E.5
# names the factor lambda and reports 1e2, used here unchanged (deviation 7).
# Code <-> math: z = z, contexts = {x_j}, target = o*, template 0 = p, ("ess", no prefix) = p',
# base_ess_probs = P_G[x | p'] of the unedited network, KL_FACTOR = lambda.
# o* is used ONLY here, as the loss target of the search for v*. No weight sees it, and the
# evaluation never reads it except to score the result.


def value_objective(
    net: Network,
    z: Vector,
    subject: int,
    target: int,
    contexts: list[Prompt],
    site: str,
    base_ess_probs: Vector,
) -> tuple[float, Vector, float, float]:
    """L(z) of Eq. 4 and dL/dz. Returns (L, dL/dz, mean NLL term, KL term)."""
    n = len(contexts)
    dz = [0.0] * D
    nll = 0.0
    for prefix in contexts:
        tokens, cand, subj_pos = make_prompt(prefix, subject, 0)
        # site "last" is the wrong-token control: z replaces the MLP output at the final
        # (template) token instead of the subject token.
        pos = subj_pos if site == "subject" else len(tokens) - 1
        cache = forward(net, tokens, cand, patch={(pos, "m"): z})
        loss, dl = ce_loss_and_dlogits(cache["probs"], target, cand)
        nll += loss
        # The 1/N of (4a) is folded into the logit gradient. backward() returns dL/dh1 at
        # every position; at the patched position that equals dL/dm = dL/dz.
        d_h1 = backward(net, cache, [x / n for x in dl])
        dz = [a + b for a, b in zip(dz, d_h1[pos])]
    tokens, cand, subj_pos = make_prompt([], subject, "ess")
    pos = subj_pos if site == "subject" else len(tokens) - 1
    cache = forward(net, tokens, cand, patch={(pos, "m"): z})
    pe = cache["probs"]
    # KL(p || b) = sum_x p_x (log p_x - log b_x). With p = softmax(logits), its gradient with
    # respect to logit x is p_x (log p_x - log b_x - KL), scaled here by lambda.
    kl = sum(p * (math.log(max(p, 1e-300)) - math.log(b)) for p, b in zip(pe, base_ess_probs))
    dl = [p * (math.log(max(p, 1e-300)) - math.log(b) - kl) * KL_FACTOR
          for p, b in zip(pe, base_ess_probs)]
    d_h1 = backward(net, cache, dl)
    dz = [a + b for a, b in zip(dz, d_h1[pos])]
    return nll / n + KL_FACTOR * kl, dz, nll / n, kl


def optimise_value(
    net: Network,
    subject: int,
    target: int,
    contexts: list[Prompt],
    site: str,
    z0: Vector,
    base_ess_probs: Vector,
) -> tuple[Vector, int, tuple[int, float, float, float], tuple[float, Vector, float, float]]:
    """Minimise Eq. 4 over z with Adam, starting from z0 = W_proj k* (the current value).

    Only the 16 numbers of z are optimised; every network weight stays frozen. Returns
    (v*, steps taken, first trace entry, final objective). The L2 decay pulls z toward 0 and
    is added to the gradient only, so it is neither part of L(z) nor of the early stop."""
    z = list(z0)
    m_state = [0.0] * D
    v_state = [0.0] * D
    steps = 0
    trace = []
    for step in range(1, VALUE_MAX_STEPS + 1):
        loss, dz, nll, kl = value_objective(net, z, subject, target, contexts, site, base_ess_probs)
        trace.append((step - 1, loss, nll, kl))
        if loss < VALUE_EARLY_STOP:
            break
        steps = step
        new_z = []
        for i in range(D):
            g = dz[i] + VALUE_WEIGHT_DECAY * z[i]
            # Adam with beta1 = 0.9, beta2 = 0.999. The literals 0.1 and 0.001 are kept as
            # written: in floating point 1 - 0.9 is not exactly 0.1, and the reference output
            # depends on the exact arithmetic.
            m_state[i] = 0.9 * m_state[i] + 0.1 * g
            v_state[i] = 0.999 * v_state[i] + 0.001 * g * g
            mh = m_state[i] / (1 - 0.9 ** step)
            vh = v_state[i] / (1 - 0.999 ** step)
            new_z.append(z[i] - VALUE_LR * mh / (math.sqrt(vh) + 1e-8))
        z = new_z
    final = value_objective(net, z, subject, target, contexts, site, base_ess_probs)
    return z, steps, trace[0], final


# === INFERENCE: ROME STEPS 1 AND 3 -- KEY, COVARIANCE, RANK-ONE UPDATE (Eq. 2, 3; App. A) ===
# The memory view (Sec. 3.1). Stack the keys the MLP has seen as columns K = [k_1 | k_2 | ...]
# and the values it produced as V = [v_1 | v_2 | ...]. A trained W_proj acts as a linear
# associative memory W K ~= V. App. A derives the edit from the least-squares picture:
#   W minimizes ||W K - V||_F^2                                                    (Eq. 5)
#   so it solves the normal equations  W K K^T = V K^T                              (Eq. 6)
# Insert one new association, keeping the least-squares fit to everything else:
#   minimize ||W_hat K - V||_F^2  subject to  W_hat k* = v*                     (Eq. 2, Eq. 7)
# Lagrangian, with one multiplier per output row (Lambda in R^D):
#   L(W_hat, Lambda) = 1/2 ||W_hat K - V||_F^2 - Lambda^T (W_hat k* - v*)                (Eq. 8)
#                    = 1/2 (W_hat K)(W_hat K)^T - V (W_hat K)^T + 1/2 V V^T
#                      - Lambda^T (W_hat k* - v*)                                       (Eq. 9)
#   0 = dL/dW_hat = W_hat (K K^T) - V K^T - Lambda k*^T                                (Eq. 10)
#   W_hat K K^T = V K^T + Lambda k*^T                                                  (Eq. 11)
# Subtract Eq. 6 from Eq. 11:  (W_hat - W) K K^T = Lambda k*^T                         (Eq. 12)
# With C = K K^T (symmetric, assumed invertible) and u = C^-1 k*:
#   W_hat = W + Lambda (C^-1 k*)^T = W + Lambda u^T                     (Eq. 13, Eq. 2)
#   W_hat I - Lambda u^T = W                                                           (Eq. 14)
# Eq. 14 and Eq. 7 together form one linear system in (W_hat, Lambda)                 (Eq. 15)
# Substituting Eq. 13 into Eq. 7:  W_hat k* = W k* + Lambda (u^T k*) = v*              (Eq. 16)
#   Lambda = (v* - W k*) / (u^T k*) = (v* - W k*) / ((C^-1 k*)^T k*)                   (Eq. 17)
# Dimensions here: W, W_hat, Delta W in R^{16 x 64}; k*, u in R^64; v*, W k*, Lambda in R^16.
# Delta W = Lambda u^T is an outer product, so it is rank one by construction: it changes the
# output only along the direction Lambda, by an amount u^T k for an incoming key k.
#
# Why C^-1 matters: for another subject's key k the change is Delta W k = Lambda (u^T k). With
# C = I that scalar is k*^T k, which is large because ReLU keys are nonnegative and overlap;
# u = C^-1 k* instead points along the part of k* that is unusual relative to typical keys,
# so u^T k stays small for other subjects. The C = I control below measures exactly this.


def estimate_covariance(net: Network, cities: list[int], cats: list[int]) -> tuple[Matrix, int]:
    """C = (1/n) sum k k^T over the keys at EVERY position of COV_PROMPTS sampled training
    prompts (App. E.5 samples every token). Uncentered (a second moment, not a covariance
    around the mean) and with no ridge term, as in Eq. 2. The 1/n scale cancels in Eq. 17."""
    rng = stream(4)
    items = training_items()
    cov = [[0.0] * H for _ in range(H)]
    count = 0
    for _ in range(COV_PROMPTS):
        s, t = items[rng.randrange(len(items))]
        tokens, cand, _ = make_prompt(sample_prefix(rng), s, t)
        for k in forward(net, tokens, cand)["k"]:
            for i in range(H):
                if k[i] != 0.0:  # ReLU keys are sparse; zero rows add nothing
                    ki = k[i]
                    cov[i] = [c + ki * b for c, b in zip(cov[i], k)]
            count += 1
    return [[c / count for c in row] for row in cov], count


def solve_gauss_jordan(matrix: Matrix, rhs: Vector) -> tuple[Vector, float]:
    """Solve matrix @ x = rhs by Gauss-Jordan elimination with partial pivoting.

    Returns (x, smallest pivot magnitude). u = C^-1 k* is computed by solving C u = k*
    rather than forming C^-1; a near-zero pivot means C is singular and the edit undefined."""
    n = len(matrix)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    min_pivot = float("inf")
    for col in range(n):
        # Partial pivoting: swap in the row with the largest entry in this column.
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        min_pivot = min(min_pivot, abs(pivot))
        if abs(pivot) < 1e-12:
            raise ZeroDivisionError("singular covariance: pivot %.3e" % abs(pivot))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv = 1.0 / aug[col][col]
        aug[col] = [v * inv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)], min_pivot


def numeric_rank(matrix: Matrix, tol: float) -> int:
    """Row-space rank by modified Gram-Schmidt: count rows with a component, larger than tol,
    outside the span of the rows already kept."""
    basis: Matrix = []
    for row in matrix:
        vec = list(row)
        for b in basis:
            coef = dot(vec, b)
            vec = [v - coef * x for v, x in zip(vec, b)]
        norm = math.sqrt(dot(vec, vec))
        if norm > tol:
            basis.append([v / norm for v in vec])
    return len(basis)


def edit_contexts(subject: int) -> list[Prompt]:
    """The N = 20 prefixes x_j used for both k* (Eq. 3) and v* (Eq. 4). Drawn from a stream
    that the evaluation never uses, so the edit is scored on prefixes it was not fitted to."""
    rng = stream(100 + subject)
    return [sample_prefix(rng) for _ in range(NUM_EDIT_CONTEXTS)]


def rome_edit(
    net: Network, cov: Matrix, subject: int, target: int, variant: str
) -> tuple[Matrix, dict]:
    """Compute an edited copy of W_proj. Returns (W_hat, diagnostics).

    variant 'rome': the paper's update. 'identity': the C = I control (u = k*). 'wrong_site':
    key and value taken at the final template token instead of the subject token. Both
    controls are closed-form edits of the SAME trained weights, reported only."""
    site = "last" if variant == "wrong_site" else "subject"
    contexts = edit_contexts(subject)
    # Step 1, Eq. 3: k* = (1/N) sum_j k(x_j + s), the MLP key at the (last) subject token
    # averaged over prefixed contexts. The paper reads k after the nonlinearity, as here.
    # The current values at the same site give the starting point z0 = mean_j W_proj k_j.
    keys, values = [], []
    for prefix in contexts:
        tokens, cand, subj_pos = make_prompt(prefix, subject, 0)
        pos = subj_pos if site == "subject" else len(tokens) - 1
        cache = forward(net, tokens, cand)
        keys.append(cache["k"][pos])
        values.append(cache["m"][pos])
    n = len(contexts)
    k_star = [sum(k[i] for k in keys) / n for i in range(H)]
    z0 = [sum(v[i] for v in values) / n for i in range(D)]
    # Step 2, Eq. 4: v*, against the unedited essence distribution P_G[x | p'].
    base_tokens, base_cand, _ = make_prompt([], subject, "ess")
    base_ess = forward(net, base_tokens, base_cand)["probs"]
    v_star, steps, first, final = optimise_value(net, subject, target, contexts, site, z0, base_ess)
    # Step 3, Eq. 2 / 13 / 17: W_hat = W + Lambda u^T,  Lambda = (v* - W k*) / (u^T k*).
    w_k = matvec(net["Wproj"], k_star)
    if variant == "identity":
        u, min_pivot = list(k_star), float("nan")
    else:
        u, min_pivot = solve_gauss_jordan(cov, k_star)  # u = C^-1 k*
    denom = dot(u, k_star)  # u^T k* = (C^-1 k*)^T k*
    lam = [(v - w) / denom for v, w in zip(v_star, w_k)]  # Lambda, Eq. 17
    # W_hat[i][j] = W[i][j] + Lambda_i u_j, built as a NEW matrix: the trained W_proj is never
    # modified, so every edit starts from the same weights.
    new_w = [[w + l * ui for w, ui in zip(row, u)] for row, l in zip(net["Wproj"], lam)]
    diag = {
        "k_star": k_star, "v_star": v_star, "u": u, "lam": lam, "w_k": w_k, "denom": denom,
        "steps": steps, "first": first, "final": final, "min_pivot": min_pivot,
    }
    return new_w, diag


def mechanical_checks(
    net: Network, new_w: Matrix, diag: dict, cov: Matrix, variant: str
) -> dict[str, float | int | None]:
    """Runtime identities that make the edit verifiably ROME and not something else:
    rank(Delta W) = 1; W_hat k* = v* (the Eq. 7 constraint holds); Delta W = Lambda u^T
    entrywise; and C u = k* (the linear solve is accurate). Tolerances are absolute bounds
    scaled by the magnitudes involved."""
    delta = [[a - b for a, b in zip(r1, r0)] for r1, r0 in zip(new_w, net["Wproj"])]
    peak = max(abs(v) for row in delta for v in row)
    rank = numeric_rank(delta, 1e-10 * peak)
    achieved = matvec(new_w, diag["k_star"])
    mc2 = max(abs(a - b) for a, b in zip(achieved, diag["v_star"]))
    mc2_bound = 1e-9 * (1 + max(abs(v) for v in diag["v_star"]))
    mc3 = max(abs(delta[i][j] - diag["lam"][i] * diag["u"][j]) for i in range(D) for j in range(H))
    mc3_bound = 1e-12 * (1 + peak)
    solve_resid = None
    if variant != "identity":
        cu = matvec(cov, diag["u"])
        solve_resid = max(abs(a - b) for a, b in zip(cu, diag["k_star"]))
    return {"rank": rank, "vk_err": mc2, "vk_bound": mc2_bound, "formula_err": mc3,
            "formula_bound": mc3_bound, "solve_resid": solve_resid, "delta_max": peak}


# === INFERENCE: EVALUATING AN EDITED NETWORK (Sec. 3.2-3.4) ===


def make_eval_sets() -> list[Prompt]:
    """20 evaluation prefixes from their own stream, disjoint from the edit prefixes."""
    rng = stream(5)
    return [sample_prefix(rng) for _ in range(NUM_EVAL_CONTEXTS)]


def baseline_predictions(
    net: Network, eval_ctx: list[Prompt]
) -> dict[tuple[int, int | str, int], tuple[list[int], Vector]]:
    """Pre-edit predictions for every (subject, template, prefix): the reference that locality,
    essence and paraphrase eligibility compare against."""
    base = {}
    for s in range(NUM_SUBJECTS):
        for t in [0, 1, 2, 3, "ess"]:
            for ci, prefix in enumerate(eval_ctx):
                base[(s, t, ci)] = predict(net, prefix, s, t)
    return base


def evaluate(
    net_edit: Network,
    base: dict[tuple[int, int | str, int], tuple[list[int], Vector]],
    eval_ctx: list[Prompt],
    subject: int,
    old: int,
    new: int,
    cities: list[int],
) -> dict[str, float]:
    """Score one edited network. new is o*, old is the true city o^c.

    ES  efficacy:      P[o*] > P[o^c] on the edited prompt template T0, 20 prefixes
    PS  paraphrase:    same comparison on T1, T2 and T3 (T3 never trained for this subject);
                       only prompts the unedited model answered with o^c count ("eligible")
    NS  neighborhood:  the other subjects of the old city must keep P[o^c] > P[o*]
    LO  locality:      argmax unchanged for the 23 other subjects, 4 templates, 8 prefixes
    BLEED              how often those prompts newly answer o* (the edit leaking)
    ESSENCE            the category (ESS) argmax unchanged for all 24 subjects
    ES, PS and NS are the paper's success metrics; LO, BLEED and ESSENCE are this toy's
    extra collateral checks."""
    neighbors = [s for s in range(NUM_SUBJECTS) if s != subject and cities[s] == old]
    out = {}
    ef = [predict(net_edit, p, subject, 0) for p in eval_ctx]
    out["ES"] = sum(prob_of(c, pr, new) > prob_of(c, pr, old) for c, pr in ef) / len(ef)
    out["EF_pnew"] = sum(prob_of(c, pr, new) for c, pr in ef) / len(ef)
    out["EF_pold"] = sum(prob_of(c, pr, old) for c, pr in ef) / len(ef)
    seen_ok = seen_tot = held_ok = held_tot = 0
    elig = elig_total = 0
    for t in (1, 2, 3):
        for ci, p in enumerate(eval_ctx):
            c, pr = predict(net_edit, p, subject, t)
            bc, bp = base[(subject, t, ci)]
            eligible = argmax_answer(bc, bp) == old
            elig_total += 1
            if not eligible:
                continue
            elig += 1
            hit = prob_of(c, pr, new) > prob_of(c, pr, old)
            if t == 3:
                held_ok += hit
                held_tot += 1
            else:
                seen_ok += hit
                seen_tot += 1
    out["PS_elig_fraction"] = elig / elig_total
    out["PS"] = (seen_ok + held_ok) / max(1, seen_tot + held_tot)
    out["PS_seen"] = seen_ok / max(1, seen_tot)
    out["PS_T3"] = held_ok / max(1, held_tot)
    out["PS_T3_n"] = held_tot
    nb_ok = nb_tot = 0
    for s in neighbors:
        for t in (0, 1, 2, 3):
            for p in eval_ctx:
                c, pr = predict(net_edit, p, s, t)
                nb_ok += prob_of(c, pr, old) > prob_of(c, pr, new)
                nb_tot += 1
    out["NS"] = nb_ok / nb_tot
    lo_same = lo_tot = bleed = 0
    for s in range(NUM_SUBJECTS):
        if s == subject:
            continue
        for t in (0, 1, 2, 3):
            for ci in range(NUM_LOCALITY_CONTEXTS):
                c, pr = predict(net_edit, eval_ctx[ci], s, t)
                bc, bp = base[(s, t, ci)]
                now = argmax_answer(c, pr)
                lo_same += now == argmax_answer(bc, bp)
                bleed += now == new and argmax_answer(bc, bp) != new
                lo_tot += 1
    out["LO"] = lo_same / lo_tot
    out["BLEED"] = bleed / lo_tot
    es_same = es_tot = 0
    for s in range(NUM_SUBJECTS):
        for ci in range(NUM_LOCALITY_CONTEXTS):
            c, pr = predict(net_edit, eval_ctx[ci], s, "ess")
            bc, bp = base[(s, "ess", ci)]
            es_same += argmax_answer(c, pr) == argmax_answer(bc, bp)
            es_tot += 1
    out["ESSENCE"] = es_same / es_tot
    return out


# === INFERENCE: COMPACT CAUSAL TRACE (Sec. 2.1, simplified) ===
# Paper Sec. 2.1: run the prompt clean, then corrupted (Gaussian noise added to the subject's
# embedding), then corrupted while restoring ONE internal state to its clean value.
#   total effect     TE = P_clean[o] - P_corrupt[o]
#   indirect effect  IE = P_corrupt, restore state[o] - P_corrupt[o]
# Noise std is 3x the standard deviation of the embedding entries (App. B.1). Here the prompt
# is [filler, subject, T0] and the states are h0, m (MLP output) and h1 at each position plus
# the attention output at the last token. Deviation 5: with no attention before the MLP the
# subject's information can only sit at the subject position, so this trace CONFIRMS a flow
# the architecture imposes; the informative number is how much of it the MLP output m carries
# compared with the residual embedding h0.


def causal_trace(
    net: Network, cities: list[int]
) -> tuple[dict[tuple[int, str], float], float, float, float]:
    rng = stream(6)
    emb_vals = [v for row in net["E"] for v in row]
    mean = sum(emb_vals) / len(emb_vals)
    sigma = 3.0 * math.sqrt(sum((v - mean) ** 2 for v in emb_vals) / len(emb_vals))
    states = ["h0", "m", "h1"]
    grid = {(pos, st): 0.0 for pos in range(3) for st in states}
    attn_effect = 0.0
    te_sum = 0.0
    trials = 0
    for s in range(NUM_SUBJECTS):
        prefix = [FILLER_BASE + rng.randrange(NUM_FILLERS)]
        tokens, cand, subj_pos = make_prompt(prefix, s, 0)
        clean = forward(net, tokens, cand)
        p_clean = prob_of(cand, clean["probs"], cities[s])
        for _ in range(TRACE_NOISE_SAMPLES):
            noise = {subj_pos: [rng.gauss(0.0, sigma) for _ in range(D)]}
            corrupt = forward(net, tokens, cand, noise=noise)
            p_corrupt = prob_of(cand, corrupt["probs"], cities[s])
            te_sum += p_clean - p_corrupt
            for pos in range(3):
                for st in states:
                    patch = {(pos, st): clean[st][pos]}
                    r = forward(net, tokens, cand, patch=patch, noise=noise)
                    grid[(pos, st)] += prob_of(cand, r["probs"], cities[s]) - p_corrupt
            r = forward(net, tokens, cand, patch={(2, "attn"): clean["attn"]}, noise=noise)
            attn_effect += prob_of(cand, r["probs"], cities[s]) - p_corrupt
            trials += 1
    return {k: v / trials for k, v in grid.items()}, attn_effect / trials, te_sum / trials, sigma


# === MAIN: TRAIN, TRACE, EDIT, EVALUATE ===


def fmt(x: float) -> str:
    return "%.4f" % x


def main() -> None:
    print("--- stage 1: synthetic facts and training from random weights ---")
    cities, cats = build_facts()
    print("facts: subject city category")
    for s in range(NUM_SUBJECTS):
        note = "  [edit subject; (subject,T3) held out]" if s in EDIT_SUBJECTS else ""
        print("  %-6s %-7s %-6s%s" % (SUBJECT_NAMES[s], CITY_NAMES[cities[s]],
                                      CATEGORY_NAMES[cats[s]], note))
    net = init_net()
    print("parameters:", sum(len(row) for name in PARAM_NAMES for row in net[name]),
          "tensors:", PARAM_NAMES, "H=%d D=%d" % (H, D))
    t0 = time.perf_counter()
    history = train(net, cities, cats)
    print("[time] training seconds %.1f" % (time.perf_counter() - t0))
    # results[name] = (value, passed). Thresholds were frozen before this program existed.
    results: dict[str, tuple[object, bool]] = {}
    tail = history[-50:]
    results["TS1"] = (sum(tail) / len(tail), sum(tail) / len(tail) <= 0.05)
    print("initial batch NLL %.4f -> final mean(last 50) %.4f" % (history[0], results["TS1"][0]))
    eval_ctx = make_eval_sets()
    ok, total = training_grid_accuracy(net, cities, cats, eval_ctx[:8])
    results["TS2"] = ((ok, total), ok == total)
    print("training-grid exact recall: %d/%d" % (ok, total))
    held_ok = held_tot = 0
    for s in EDIT_SUBJECTS:
        for p in eval_ctx[:8]:
            c, pr = predict(net, p, s, 3)
            held_ok += argmax_answer(c, pr) == cities[s]
            held_tot += 1
    print("held-out (edit subject, T3) pre-edit recall (report-only): %d/%d" % (held_ok, held_tot))

    # The key second moment C is a property of the trained network, shared by every edit.
    print("--- stage 2: key covariance C and causal trace ---")
    t0 = time.perf_counter()
    cov, n_keys = estimate_covariance(net, cities, cats)
    sym = max(abs(cov[i][j] - cov[j][i]) for i in range(H) for j in range(H))
    diag_mean = sum(cov[i][i] for i in range(H)) / H
    print("C estimated from %d keys; symmetric to %.1e; mean diagonal %.4f"
          % (n_keys, sym, diag_mean))
    # A probe solve C x = 1 reports the smallest pivot: evidence that C is invertible here.
    probe_u, min_pivot = solve_gauss_jordan(cov, [1.0] * H)
    print("C Gauss-Jordan min pivot (probe solve) %.4e" % min_pivot)
    results["MC5"] = ((sym, min_pivot), sym <= 1e-12 and min_pivot >= 1e-12)
    print("[time] covariance seconds %.1f" % (time.perf_counter() - t0))

    t0 = time.perf_counter()
    grid, attn_eff, te, sigma = causal_trace(net, cities)
    print("causal trace: prompt [filler, subject, T0]; sigma=%.3f; mean TE=%.4f" % (sigma, te))
    print("  mean indirect effect (restore clean state while subject embedding corrupted)")
    print("  %-10s %8s %8s %8s" % ("position", "h0", "m(MLP)", "h1"))
    for pos, name in enumerate(["filler", "subject", "relation"]):
        print("  %-10s %8.4f %8.4f %8.4f" % (name, grid[(pos, "h0")], grid[(pos, "m")],
                                              grid[(pos, "h1")]))
    print("  attention output at last token: %.4f" % attn_eff)
    print("[time] trace seconds %.1f" % (time.perf_counter() - t0))

    # Each edit: city of subject s from old to new = (old + 3) mod 8, made three ways on fresh
    # copies of the trained weights. `snapshot` proves afterwards that the trained network
    # itself was never modified and that each edited copy differs from it only in W_proj.
    print("--- stage 3: six rank-one edits, each with the C = I and wrong-token controls ---")
    base = baseline_predictions(net, eval_ctx)
    snapshot = {name: [list(r) for r in net[name]] for name in PARAM_NAMES}
    agg: dict[str, dict[str, list[float]]] = {v: {} for v in ("rome", "identity", "wrong_site")}
    mech_ok = True
    t0 = time.perf_counter()
    for s in EDIT_SUBJECTS:
        old = cities[s]
        new = (old + 3) % NUM_CITIES
        print("edit: %s  %s -> %s" % (SUBJECT_NAMES[s], CITY_NAMES[old], CITY_NAMES[new]))
        for variant in ("rome", "identity", "wrong_site"):
            new_w, diag = rome_edit(net, cov, s, new, variant)
            # A deep copy in which only W_proj is replaced by W_hat; every other tensor is a
            # copy of the trained one, so the comparison against `snapshot` is not tautological.
            edited = {name: [list(r) for r in (new_w if name == "Wproj" else net[name])]
                      for name in PARAM_NAMES}
            mc = mechanical_checks(net, new_w, diag, cov, variant)
            unchanged = (all(edited[name] == snapshot[name] for name in PARAM_NAMES
                             if name != "Wproj") and edited["Wproj"] != snapshot["Wproj"]
                         and all(net[name] == snapshot[name] for name in PARAM_NAMES))
            # The identities are hard checks for the paper's update and for C = I. The
            # wrong-token control is reported only (it is not the method being demonstrated).
            if variant != "wrong_site":
                solve_bound = 1e-8 * (1 + max(abs(v) for v in diag["k_star"]))
                pass_all = (mc["rank"] == 1 and mc["vk_err"] <= mc["vk_bound"]
                            and mc["formula_err"] <= mc["formula_bound"]
                            and (mc["solve_resid"] is None or mc["solve_resid"] <= solve_bound)
                            and unchanged)
                mech_ok = mech_ok and pass_all
            ev = evaluate(edited, base, eval_ctx, s, old, new, cities)
            for key, val in ev.items():
                agg[variant].setdefault(key, []).append(val)
            # steps: Adam steps on z; L: Eq. 4 objective before -> after; rank, |Wk*-v*|,
            # formula (|Delta W - Lambda u^T|) and solve (|C u - k*|) are the identities above.
            print("  %-10s steps=%2d L:%.3f->%.3f (nll %.3f kl %.5f) rank=%d |Wk*-v*|=%.1e "
                  "formula=%.1e solve=%s" % (
                      variant, diag["steps"], diag["first"][1], diag["final"][0],
                      diag["final"][2], diag["final"][3], mc["rank"], mc["vk_err"],
                      mc["formula_err"],
                      "n/a" if mc["solve_resid"] is None else "%.1e" % mc["solve_resid"]))
            print("             ES=%s PS=%s (seen %s, T3 %s, elig %s) NS=%s LO=%s BLEED=%s ESS=%s "
                  "P[o*]=%s P[old]=%s" % (
                      fmt(ev["ES"]), fmt(ev["PS"]), fmt(ev["PS_seen"]), fmt(ev["PS_T3"]),
                      fmt(ev["PS_elig_fraction"]), fmt(ev["NS"]), fmt(ev["LO"]), fmt(ev["BLEED"]),
                      fmt(ev["ESSENCE"]), fmt(ev["EF_pnew"]), fmt(ev["EF_pold"])), flush=True)
    print("[time] edit seconds %.1f" % (time.perf_counter() - t0))
    results["MC1-MC4"] = (None, mech_ok)

    def mean(variant: str, key: str) -> float:
        vals = agg[variant][key]
        return sum(vals) / len(vals)

    print("--- stage 4: results ---")
    print("AGGREGATE over %d edit subjects (mean)" % len(EDIT_SUBJECTS))
    print("  %-10s %7s %7s %7s %7s %7s %7s %7s %7s" % (
        "variant", "ES", "PS", "PS_T3", "NS", "LO", "BLEED", "ESS", "elig"))
    for variant in ("rome", "identity", "wrong_site"):
        print("  %-10s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f" % (
            variant, mean(variant, "ES"), mean(variant, "PS"), mean(variant, "PS_T3"),
            mean(variant, "NS"), mean(variant, "LO"), mean(variant, "BLEED"),
            mean(variant, "ESSENCE"), mean(variant, "PS_elig_fraction")))
    # Paraphrase success is only claimable if most paraphrase prompts were eligible, i.e.
    # answered with the old city before the edit.
    eb2_eval = mean("rome", "PS_elig_fraction") >= 0.8
    results["EB1"] = (mean("rome", "ES"), mean("rome", "ES") >= 0.95)
    results["EB2"] = (mean("rome", "PS"), eb2_eval and mean("rome", "PS") >= 0.80)
    results["EB3"] = (mean("rome", "NS"), mean("rome", "NS") >= 0.90)
    results["EB4"] = (mean("rome", "LO"), mean("rome", "LO") >= 0.95)
    results["EB5"] = (mean("rome", "ESSENCE"), mean("rome", "ESSENCE") >= 0.98)
    # TS1/TS2 training; MC1-MC4 edit identities; MC5 covariance; EB1-EB5 behavior of the
    # paper's update only. The controls and the trace never decide acceptance.
    print("CRITERIA (thresholds frozen before this program was written; controls report-only)")
    for key in ("TS1", "TS2", "MC1-MC4", "MC5", "EB1", "EB2", "EB3", "EB4", "EB5"):
        val, ok = results[key]
        print("  %-8s %s  value=%s" % (key, "PASS" if ok else "FAIL", val))
    total_time = time.perf_counter() - WALL_START
    print("[time] total seconds %.1f  (limit: under 7 minutes on an M-series Mac)" % total_time)
    failed = [key for key, (_, ok) in results.items() if not ok]
    if failed:
        raise RuntimeError("acceptance failed: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()

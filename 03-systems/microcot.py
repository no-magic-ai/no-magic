"""
Self-consistency on learned reasoning traces: sample several traces from a tiny model, then vote.
Two copies of one tiny language model are trained on the same digit-sum problems: one learns to
answer directly, the other learns to write a short reasoning trace (a route token and three
partial sums) before the answer. The trace model is then decoded greedily and by
self-consistency: sample 40 traces, read the answer after '#', and take a majority vote. Every
trace is generated token by token from learned weights; no decoder ever sees a gold trace or a
gold answer.
"""
# Reference (primary; this program implements its decoder): Wang, Wei, Schuurmans, Le, Chi,
# Narang, Chowdhery & Zhou, "Self-Consistency Improves Chain of Thought Reasoning in Language
# Models", ICLR 2023, https://arxiv.org/abs/2203.11171 -- Sec. 2 (sample reasoning paths r_i with
# answers a_i and marginalize by a majority vote over a fixed answer set; footnote 1: a
# task-specific parser reads the answer after a fixed marker), Eq. 1 and Table 1 (length-
# normalized path probability; weighted and unweighted aggregations), Sec. 3.1-3.2 (temperature
# sampling; greedy chain-of-thought decoding is the baseline), Sec. 3.4-3.5 (diversity of the
# reasoning paths is the key to the gain), Sec. 5 (extra inference cost).
# Related (summarized, not implemented): Wei et al., "Chain-of-Thought Prompting Elicits
# Reasoning in Large Language Models", NeurIPS 2022, https://arxiv.org/abs/2201.11903 -- few-shot
# prompting of frozen models of about 100B parameters (see D2 below).
# Training lineage actually used by the trace arm: Nye et al., "Show Your Work: Scratchpads for
# Intermediate Computation with Language Models", 2021, https://arxiv.org/abs/2112.00114 --
# supervised training on written intermediate steps.

# === TRADEOFFS ===
# + A vote over sampled traces can recover from one bad route or a bad greedy route choice
# + Needs no extra training, verifier or reward model: only sampling and an answer parser
# + The parser makes each sample's contribution explicit, so the vote can be counted by hand
# - Costs one full decode per sample (40 here) instead of one; the gain saturates early (D5)
# - Needs a fixed answer set and a parser; free-form answers cannot be counted (Sec. 2)
# - Helps only when sampled routes differ; samples that share one systematic mistake simply
#   agree on the wrong answer (an earlier single-route design tied greedy; D12)
# WHEN TO USE: tasks with a short, checkable final answer, several valid ways to derive it, and
#   spare inference compute.
# WHEN NOT TO: open-ended generation with no answer to count, latency-bound serving, or a model
#   whose routes all fail the same way.

# === SOURCE ASSUMPTIONS AND DISCLOSURES (D1-D13) ===
# This is a toy teaching adaptation, not a reproduction of either paper. The numbers quoted here
# are the seed-42 values of the pre-registered design study that this file must reproduce
# exactly under CPython 3.12.8; the program prints and checks its own values when it runs.
# D1  Route diversity is built into the training data: three fixed operand orders (L, R, T),
#     announced by a route token the model must emit first. In Wang et al. diversity comes from
#     sampling free-form reasoning of a large pretrained model (Sec. 2, 3.4-3.5).
# D2  The trace arm is supervised scratchpad training (Nye et al. lineage). Wei et al.'s few-shot
#     prompting of frozen ~100B-parameter models is not implemented; their Sec. 3.2 reports that
#     small models write fluent but illogical chains, and no model there was fine-tuned.
# D3  The vote beats greedy by +0.104 at seed 42, but part of that undoes greedy's own poor route
#     choice: greedy picks L/R/T in 100/71/329 of 500 problems and scores 0.702, below forcing
#     route L (0.738) or T (0.728). Against the best single forced route, chosen after seeing
#     the test slice, the margin is +0.068 (+0.028 to +0.070 across four design-study seeds).
#     Greedy is the baseline because Wang et al. compare with greedy chain of thought (Sec. 3.2).
# D4  Trace versus direct answering is NOT claimed: +0.052 at seed 42 with exact McNemar
#     p = 0.083, and the direct arm sits on a long loss plateau (0.650 / 0.054 / 0.486 / 0.192
#     accuracy across design seeds 42 / 0 / 1 / 2). Budgets are unequal: trace 150 epochs and
#     2,100,000 supervised tokens, direct 90 epochs and 360,000 tokens -- about 5.8x the tokens
#     and 5.2-5.6x the training time for the trace arm.
# D5  The vote costs 40 sampled traces per problem and saturates near k = 20 (0.810 at k = 20,
#     0.806 at k = 40).
# D6  The faithfulness probes support no claim of faithful step-by-step reasoning: the answer
#     copies the last written sum in 0.831 of internally consistent corrupted traces, but later
#     steps follow a corrupted first step in only about 30-34 percent of cases.
# D7  The trace defaults (150 epochs, learning rate 0.03) were the best development cell at the
#     edge of the explored grid; longer training was not explored.
# D8  The held-out problems lean toward repeated digits (all-distinct operands are 0.386 of the
#     test slice against 0.504 of all 10,000 problems), a by-product of excluding every multiset
#     an earlier probe had evaluated. Accuracies and gains are not full-task estimates.
# D9  Exact numbers depend on the partition, the defaults, the floating-point order of every sum
#     and CPython 3.12.8 on one host; other interpreters may differ in the last bits of exp,
#     tanh and log. The claim is reproduction under that setup, not portability.
# D10 Runtime: the frozen design prototype of this program took 316.14 s on an Apple M1 Pro with
#     CPython 3.12.8 (pre-registered gate 360 s; product limit 420 s on M-series). That is a
#     prototype figure, not a measurement of this file; legacy Intel is unmeasured.
# D11 Greedy trace accuracy is only 0.702 at seed 42 (0.606-0.702 across four design seeds; the
#     0.60 floor held narrowly at seed 0). The other seeds are design evidence and not shipped.
# D12 History: a first single-route design was recorded as DESIGN NO-GO (the vote tied greedy at
#     0.904 and a pre-registered trace-use criterion scored 0.351 against 0.80). The multi-route
#     idea came after seeing that result, so this round was pre-registered before any code and
#     evaluated once on a fresh partition.
# D13 Everything is a toy: a position-slot MLP, not a transformer; a synthetic 17-token digit
#     task; no emergence, no paper-scale result and no general-reasoning claim.
# Further deviations from Wang et al.: no pretrained model and no prompting; no top-k cut (their
# top-k of 40 is vacuous with 17 tokens); Eq. 1 probabilities come from the unscaled model
# (temperature 1) even when sampling hotter; the tie rule is ours; 40 samples per problem in one
# run instead of 40 averaged over 10 runs; defaults were chosen on a reserved development split
# that this program excludes from training but never evaluates.

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from itertools import combinations

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===
# Every value below is frozen by the pre-registered design study; changing any of them changes
# the reproduced numbers (D9). The random streams are separate generators on purpose:
# Random(SEED) initializes, Random(SEED + 1) shuffles training data (one fresh stream per arm),
# Random(SEED + 2) drives the greedy evaluation and printed inspection samples, and
# Random(SEED + 3) draws every self-consistency sample. Separate streams mean that adding a
# print or a report-only probe that consumes no random numbers cannot move any vote count.

Problem = tuple[int, ...]  # four operand digits, e.g. (3, 7, 2, 9)
Model = dict  # keys "E", "Q", "b1", "W2", "b2" (built in init_model)
Sample = tuple  # (parsed answer or None, sum of log-probs at temperature 1, generated length)

SEED = 42
SPLIT_SEED = 4242  # partition shuffles, independent of the model seed
N_TRAIN = 2000  # training problems sampled from the 8,439-problem training pool
N_TEST = 500  # test slice: first 500 problems of the shuffled 937-problem test pool
HIDDEN_DIM = 48
TRACE_LEARNING_RATE = 0.03
TRACE_EPOCHS = 150
DIRECT_LEARNING_RATE = 0.1  # the direct arm needs a larger step to leave its loss plateau
DIRECT_EPOCHS = 90
LR_FINAL_FRACTION = 0.1  # linear learning-rate decay to 10 percent of the start value
INIT_STD_SLOT = 0.3  # init std of the slot (E) and prediction-position (Q) tables
NUM_SAMPLES = 40  # self-consistency samples per problem (Wang et al. use 40)
PRIMARY_TAU = 0.7  # the paper's temperature for PaLM and GPT-3 (Sec. 3.1)
TAU_SWEEP = (0.3, 0.5, 0.7, 1.0, 1.5)  # sampled in this order; 0.7 results depend on it
KS = (1, 5, 10, 20, 40)  # vote sizes: the first k of each problem's 40 samples
MAX_NEW = 9  # generation budget; a well-formed trace needs 7 tokens

# Pre-registered gate thresholds (G1-G5 are checked in this run; G6-G8 need other runs).
G1_NLL_RATIO = 0.5  # final training NLL at most half the initial
G2_LOOKUP_MAX = 0.15  # no input-lookup shortcut may answer the test slice
G3_FLOOR = 0.60  # greedy trace accuracy floor
G3_STYLE_MIN_SAMPLES = 4  # a route "occurs" for a problem if it starts >= 4 of its 40 samples
G3_PROBLEM_FRACTION = 0.50  # at least half the problems must show two such routes
G4_GAIN = 0.05  # vote minus greedy at k = 40, tau 0.7, in accuracy
G4_ALPHA = 0.01  # exact two-sided McNemar significance level
CLASS_MIN_N = 20  # permutation classes smaller than this are descriptive only

# === VOCABULARY ===
# 17 tokens: digits 0-9, '+', '=', '#' (answer marker), '.' (end) and the three route tokens.
# Signpost: a real LLM has tens of thousands of subword tokens and free-form text; this closed
# vocabulary is what makes "parse the answer" a one-line rule here.
CHARS = "0123456789+=#.LRT"
V = len(CHARS)
PLUS, EQ, HASH, END = 10, 11, 12, 13
STYLES = (14, 15, 16)  # route tokens L, R, T
N_OPERANDS = 4
PROMPT_LEN = 2 * N_OPERANDS  # "3+7+2+9=" is 8 tokens
TRACE_LEN = 7  # route, p1, p2, p3, '#', answer, '.'
SLOTS = PROMPT_LEN + MAX_NEW + 1  # one learned table per sequence position
CLASSES = ("all-distinct", "one-pair", "two-pairs", "three-of-a-kind", "four-of-a-kind")


def decode_text(tokens: list[int]) -> str:
    return "".join(CHARS[t] for t in tokens)


# === DATA: TASK AND ROUTES (targets, probe inputs and scoring only) ===
# Task: the sum of four digits modulo 10. A direct completion is "answer ." and a trace
# completion is "route p1 p2 p3 # answer ." -- three equally valid routes to one answer:
#   L (left to right):  p1 = a + b,  p2 = p1 + c,  p3 = p2 + d
#   R (right to left):  p1 = d + c,  p2 = p1 + b,  p3 = p2 + a
#   T (pairwise tree):  p1 = a + b,  p2 = c + d,   p3 = p1 + p2      (all mod 10)
# Why a trace can help at all: the direct arm must map four digits to their sum in one step,
# while each trace token is a single two-digit addition the model has seen many times. Why
# routes matter for voting: three routes make different intermediate mistakes, so their final
# answers disagree when wrong but agree when right -- the premise of Wang et al. Sec. 2.
# The functions in this section build training targets, forced probe inputs and scores. The
# decoders further down receive only prompt tokens and weights, so any trace they print was
# written by the model, and a correct answer cannot be copied from a template.


def prompt_tokens(operands: Problem) -> list[int]:
    out = []
    for i, digit in enumerate(operands):
        out.append(digit)
        out.append(PLUS if i < N_OPERANDS - 1 else EQ)
    return out


def partials(operands: Problem, style: int, p1_override: int | None = None) -> tuple[int, ...]:
    """Return (p1, p2, p3) of one route; p1_override re-derives the later sums from a given p1."""
    a, b, c, d = operands
    if style == STYLES[0]:  # L: left to right
        p1 = (a + b) % 10 if p1_override is None else p1_override
        p2 = (p1 + c) % 10
        p3 = (p2 + d) % 10
    elif style == STYLES[1]:  # R: right to left
        p1 = (d + c) % 10 if p1_override is None else p1_override
        p2 = (p1 + b) % 10
        p3 = (p2 + a) % 10
    else:  # T: pairwise tree; p2 does not depend on p1, which the chaining probe accounts for
        p1 = (a + b) % 10 if p1_override is None else p1_override
        p2 = (c + d) % 10
        p3 = (p1 + p2) % 10
    return p1, p2, p3


def gold_completion(operands: Problem, arm: str, style: int | None = None) -> list[int]:
    """Training target for one arm: [answer, END] or [route, p1, p2, p3, HASH, answer, END]."""
    answer = sum(operands) % 10
    if arm == "direct":
        return [answer, END]
    p1, p2, p3 = partials(operands, style)
    assert p3 == answer  # every route must end at the same answer, or the vote premise fails
    return [style, p1, p2, p3, HASH, answer, END]


def perm_class(problem: Problem) -> str:
    # How many distinct orderings a multiset has: all-distinct has 24, four-of-a-kind has 1.
    pattern = tuple(sorted(Counter(problem).values(), reverse=True))
    return {
        (1, 1, 1, 1): "all-distinct",
        (2, 1, 1): "one-pair",
        (2, 2): "two-pairs",
        (3, 1): "three-of-a-kind",
        (4,): "four-of-a-kind",
    }[pattern]


def multiset_of(problem: Problem) -> Problem:
    return tuple(sorted(problem))


# === DATA: THE FROZEN HELD-OUT PARTITION ===
# Splitting by operand multiset keeps every permutation of a held-out problem out of training,
# so the model cannot answer 7+3+5+4 by remembering 3+4+5+7. The held-out multisets are those
# no earlier design probe had evaluated: of the 715 four-digit multisets, 539 had been seen in an
# earlier test or development slice, and the remaining 176 were shuffled with SPLIT_SEED. The
# first 90 became the test pool and the next 50 a reserved development pool (used to pick the
# defaults, excluded from training here, never evaluated here); the other 575 form the training
# pool. The two ordered lists are embedded below rather than re-derived, because re-deriving them
# needs the whole design history. A SHA-256 digest guards them: any edit stops the program.
# Order matters: the test pool is built by walking HELD_OUT_TEST in this order.
HELD_OUT_TEST = (
    "0333 8888 6669 2228 2334 3335 3337 4688 0148 0259 4444 5689 1136 1566 6888 0166 0009 8999 "
    "0023 1169 0033 4448 6667 0229 1247 1456 5777 1345 0222 2333 0668 0888 7777 7889 0667 6688 "
    "1347 2267 1238 2367 0066 2669 1488 1888 0099 2699 4449 0079 1113 1156 0557 3333 0168 4447 "
    "0244 2457 0135 2247 1159 2566 3378 4446 5888 2256 0000 1339 4499 3556 2299 1114 1599 2229 "
    "1334 3345 5699 0337 1469 1568 3788 1446 0016 4477 1222 3399 3666 5666 7999 5788 2889 2677"
)
HELD_OUT_RESERVED = (
    "1348 0345 4445 5566 4788 2445 2339 1367 1189 1244 1788 0126 0349 1338 0359 4699 4577 2236 "
    "1224 2479 1124 2279 0112 0088 0588 3377 5599 3334 5999 3777 0228 5668 1299 2568 2277 3569 "
    "3888 0227 2777 1355 3388 2249 0899 1779 6799 2449 2366 2269 4479 3699"
)
# sha256(" ".join(test words) + "|" + " ".join(reserved words)), ASCII
HELD_OUT_DIGEST = "197f4ccc52b8cfa07971e88d3590b9301f8866892f8e1a808f4f5393e55a6d20"


def held_out_digest(test_text: str, reserved_text: str) -> str:
    text = " ".join(test_text.split()) + "|" + " ".join(reserved_text.split())
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# "0259" -> (0, 2, 5, 9): one sorted operand multiset per four-character word.
def parse_multisets(text: str) -> list[Problem]:
    return [tuple(int(digit) for digit in word) for word in text.split()]


def build_splits() -> tuple[dict, dict]:
    """Rebuild the frozen partition: 2,000 training problems and the 500-problem test slice."""
    if held_out_digest(HELD_OUT_TEST, HELD_OUT_RESERVED) != HELD_OUT_DIGEST:
        raise ValueError("embedded held-out multiset lists do not match their recorded digest")
    test_multisets = parse_multisets(HELD_OUT_TEST)
    reserved_multisets = parse_multisets(HELD_OUT_RESERVED)
    problems_by_multiset: dict[Problem, list[Problem]] = {}
    for n in range(10**N_OPERANDS):  # all 10,000 problems, grouped by their sorted digits
        operands = tuple(int(c) for c in f"{n:0{N_OPERANDS}d}")
        problems_by_multiset.setdefault(multiset_of(operands), []).append(operands)
    all_multisets = sorted(problems_by_multiset)
    held = set(test_multisets) | set(reserved_multisets)
    train_multisets = [k for k in all_multisets if k not in held]
    # Each pool is a concatenation of its multisets' problems, shuffled by a FRESH Random(4242),
    # so the training sample does not depend on how the test pool was shuffled.
    test_pool = [p for k in test_multisets for p in problems_by_multiset[k]]
    random.Random(SPLIT_SEED).shuffle(test_pool)
    reserved_pool = [p for k in reserved_multisets for p in problems_by_multiset[k]]
    train_pool = [p for k in train_multisets for p in problems_by_multiset[k]]
    random.Random(SPLIT_SEED).shuffle(train_pool)
    info = {
        "all_multisets": len(all_multisets),
        "test_ms": test_multisets,
        "reserved_ms": reserved_multisets,
        "train_ms": train_multisets,
        "reserved_pool": reserved_pool,
        "pool_sizes": (len(train_pool), len(reserved_pool), len(test_pool)),
    }
    splits = {"train": train_pool[:N_TRAIN], "test": test_pool[:N_TEST]}
    return splits, info


def check_splits(splits: dict, info: dict) -> dict:
    """G2 leakage assertions: stop on any overlap or on any partition size other than frozen.

    Disjoint multisets already imply disjoint ordered tuples; both are asserted anyway, because
    G2 names both and a future edit could break one without the other.
    """
    ms = {k: {multiset_of(p) for p in v} for k, v in splits.items()}
    full_ms = {"test": set(info["test_ms"]), "reserved": set(info["reserved_ms"]),
               "train": set(info["train_ms"])}
    assert info["pool_sizes"] == (8439, 624, 937), info["pool_sizes"]
    assert len(ms["test"]) == 85, len(ms["test"])
    assert not (full_ms["test"] & full_ms["reserved"]) and not (full_ms["test"] & full_ms["train"])
    assert not (full_ms["reserved"] & full_ms["train"])
    assert ms["train"] <= full_ms["train"] and ms["test"] <= full_ms["test"]
    ordered = {k: set(v) for k, v in splits.items()}
    reserved = set(info["reserved_pool"])
    assert not (ordered["train"] & reserved) and not (ordered["train"] & ordered["test"])
    assert not (reserved & ordered["test"])
    assert len(splits["train"]) == N_TRAIN and len(splits["test"]) == N_TEST
    return ms


# Permutation-class shares show how the frozen partition skews toward repeated digits (D8).
def shares(problems: list[Problem]) -> str:
    counts = Counter(perm_class(p) for p in problems)
    return " ".join(f"{name}={counts[name] / len(problems):.3f}" for name in CLASSES)


def lookup_baselines(train_set: list[Problem], eval_set: list[Problem]) -> tuple[dict, float]:
    """Best lookup table on 1, 2 or 3 operand positions, fitted on train, scored on eval_set.

    If any table of up to three positions scored well, the task would have a shortcut that
    needs no chaining; G2 requires every such table to stay at or below 0.15 (chance is 0.10).
    Unseen keys fall back to the most common training answer (the "answer prior").
    """
    best = {}
    gold = [sum(o) % 10 for o in eval_set]
    prior = max(range(10), key=lambda a: (sum(1 for o in train_set if sum(o) % 10 == a), -a))
    for r in (1, 2, 3):
        scores = []
        for positions in combinations(range(N_OPERANDS), r):
            table: dict[tuple[int, ...], list[int]] = {}
            for operands in train_set:
                key = tuple(operands[i] for i in positions)
                table.setdefault(key, [0] * 10)[sum(operands) % 10] += 1
            hit = 0
            for operands, g in zip(eval_set, gold):
                counts = table.get(tuple(operands[i] for i in positions))
                pred = max(range(10), key=lambda a: (counts[a], -a)) if counts else prior
                hit += pred == g
            scores.append(hit / len(eval_set))
        best[r] = max(scores)
    prior_acc = sum(1 for g in gold if g == prior) / len(gold)
    return best, prior_acc


def step_pair_coverage(train_set: list[Problem], eval_set: list[Problem]) -> list[tuple]:
    """Per route: distinct (left, right) addition pairs seen in training, and how many eval
    problems use only seen pairs. 100 of 100 means the test measures chaining of learned
    single steps, not new primitive facts."""
    out = []
    for style in STYLES:

        def pairs(operands: Problem) -> list[tuple[int, int]]:
            a, b, c, d = operands
            p1, p2, p3 = partials(operands, style)
            if style == STYLES[0]:
                return [(a, b), (p1, c), (p2, d)]
            if style == STYLES[1]:
                return [(d, c), (p1, b), (p2, a)]
            return [(a, b), (c, d), (p1, p2)]

        seen = set()
        for operands in train_set:
            seen.update(pairs(operands))
        full = sum(all(p in seen for p in pairs(operands)) for operands in eval_set)
        out.append((len(seen), full))
    return out


# === MODEL: POSITION-SLOT MLP WITH HAND-WRITTEN BACKPROPAGATION ===
# A deliberately tiny autoregressive model. Every input position s has its own embedding table
# E[s], so "digit 7 in slot 2" and "digit 7 in slot 4" are different learned vectors; the
# hidden state for predicting position t is the sum of the rows of every earlier token:
#   z_t = b1 + Q[t] + sum_{s<t} E[s][x_s]       Q[t]: "which position am I predicting"
#   h_t = tanh(z_t)
#   l_t = W2 h_t + b2                             logits over the 17 tokens
#   p(x_t | x_<t) = softmax(l_t)
# Summing slot rows is order-aware (unlike a bag of words) but has no attention: the model must
# store "what was written where" in one 48-dimensional sum. That is enough for this task, and a
# signpost that real chain-of-thought models are transformers with attention over the context.
# Why traces fit this model: each written partial sum enters z_t through its own slot, so the
# next step can read it directly instead of recomputing it from the four operands.
# Shapes: E 18 slots x 18 rows x 48, Q 19 x 48, b1 48, W2 17 x 48, b2 17 = 17,345 parameters.
# Each slot has one extra row beyond the 17 tokens and Q has one extra position; neither is ever
# read, but they stay because they are part of the frozen random-initialization stream.


def init_model(hidden_dim: int, rng: random.Random) -> Model:
    gauss = rng.gauss
    # Draw order (E, then Q, then W2) fixes every initial weight; both arms copy this one model.
    # W2 uses std 1/sqrt(hidden_dim) so the initial logits have roughly unit scale.
    return {
        "E": [[[gauss(0, INIT_STD_SLOT) for _ in range(hidden_dim)] for _ in range(V + 1)]
              for _ in range(SLOTS)],
        "Q": [[gauss(0, INIT_STD_SLOT) for _ in range(hidden_dim)] for _ in range(SLOTS + 1)],
        "b1": [0.0] * hidden_dim,
        "W2": [[gauss(0, 1 / math.sqrt(hidden_dim)) for _ in range(hidden_dim)] for _ in range(V)],
        "b2": [0.0] * V,
    }


def copy_model(m: Model) -> Model:
    return {
        "E": [[r[:] for r in s] for s in m["E"]],
        "Q": [r[:] for r in m["Q"]],
        "b1": m["b1"][:],
        "W2": [r[:] for r in m["W2"]],
        "b2": m["b2"][:],
    }


def softmax(logits: list[float], tau: float = 1.0) -> list[float]:
    # p_i = exp(l_i / tau) / sum_j exp(l_j / tau); subtracting the max only prevents overflow.
    # tau < 1 sharpens the distribution toward greedy, tau > 1 flattens it toward uniform.
    scaled = [x / tau for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def head(model: Model, z_slots: list[float], t: int) -> tuple[list[float], list[float]]:
    """Hidden state and logits for predicting position t; z_slots = sum_{s<t} E[s][x_s]."""
    q = model["Q"][t]
    b1 = model["b1"]
    h = [math.tanh(a + b + c) for a, b, c in zip(z_slots, q, b1)]  # h_t = tanh(z + Q[t] + b1)
    logits = [sum(w * x for w, x in zip(row, h)) + b for row, b in zip(model["W2"], model["b2"])]
    return h, logits


def add_row(acc: list[float], row: list[float]) -> list[float]:
    return [a + r for a, r in zip(acc, row)]


# === TRAINING ===


def train_sequence(model: Model, tokens: list[int], first_target: int,
                   learning_rate: float) -> float:
    """One SGD step on the summed next-token NLL over positions >= first_target.

    Loss only on completion tokens: the prompt is given, so predicting it would teach nothing
    about the trace or the answer. Gradients (softmax cross-entropy, then tanh):
      dL/dl_t = p_t - onehot(x_t)
      dL/dh_t = W2^T (p_t - onehot(x_t))
      dL/dz_t = (1 - h_t^2) * dL/dh_t
      dL/dW2 += dL/dl_t h_t^T, dL/db2 += dL/dl_t, dL/dQ[t] = dL/db1 += dL/dz_t
      dL/dE[s][x_s] = sum over predicted positions t > s of dL/dz_t
    The last line holds because slot s feeds every later z_t. All dL/dh_t use W2 from before
    this step; the updates are applied afterwards. Signpost: real training averages gradients
    over mini-batches with an optimizer such as Adam; one-example SGD keeps the code short.
    """
    hidden_dim = len(model["b1"])
    z = [0.0] * hidden_dim
    for s in range(first_target):
        z = add_row(z, model["E"][s][tokens[s]])
    caches = []
    loss = 0.0
    for t in range(first_target, len(tokens)):
        h, logits = head(model, z, t)
        probs = softmax(logits)
        loss -= math.log(max(probs[tokens[t]], 1e-12))
        caches.append((t, h, probs))
        z = add_row(z, model["E"][t][tokens[t]])  # teacher forcing: feed the gold token
    W2 = model["W2"]
    dz_at = {}
    for t, h, probs in caches:
        d_logits = probs[:]
        d_logits[tokens[t]] -= 1.0  # dL/dl_t = p_t - onehot(x_t)
        dh = [0.0] * hidden_dim
        for c in range(V):
            g = d_logits[c]
            row = W2[c]
            for j in range(hidden_dim):
                dh[j] += g * row[j]
        dz = [dh[j] * (1.0 - h[j] * h[j]) for j in range(hidden_dim)]  # tanh' = 1 - h^2
        dz_at[t] = (dz, d_logits, h)
    b1 = model["b1"]
    for t in sorted(dz_at, reverse=True):
        dz, d_logits, h = dz_at[t]
        for c in range(V):
            g = d_logits[c]
            row = W2[c]
            for j in range(hidden_dim):
                row[j] -= learning_rate * g * h[j]
            model["b2"][c] -= learning_rate * g
        qrow = model["Q"][t]
        for j in range(hidden_dim):
            qrow[j] -= learning_rate * dz[j]
            b1[j] -= learning_rate * dz[j]
    # Suffix sums from the last position backward: grad_by_slot[s] = sum_{t > s} dL/dz_t.
    positions = sorted(dz_at)
    running = [0.0] * hidden_dim
    grad_by_slot = {}
    for t in reversed(positions):
        running = [r + d for r, d in zip(running, dz_at[t][0])]
        grad_by_slot[t - 1] = running[:]
    for s in range(len(tokens) - 1):
        if s < first_target:
            g = running  # prompt slots feed every predicted position
        elif s in grad_by_slot:
            g = grad_by_slot[s]
        else:
            continue
        row = model["E"][s][tokens[s]]
        for j in range(hidden_dim):
            row[j] -= learning_rate * g[j]
    return loss


def train_arm(arm: str, base_model: Model, train_set: list[Problem], epochs: int,
              lr0: float) -> tuple[Model, list[float]]:
    """Per-sequence SGD from a copy of the shared initialization; returns (model, epoch NLLs).

    The trace arm draws each example's route uniformly from the shuffle stream at every visit,
    so over 150 epochs a problem is seen written all three ways (this is where D1's built-in
    diversity enters). Learning rate decays linearly: lr = lr0 * (1 - 0.9 * step / total).
    """
    model = copy_model(base_model)
    rng = random.Random(SEED + 1)
    total_steps = epochs * len(train_set)
    step = 0
    losses = []
    for _ in range(epochs):
        order = list(range(len(train_set)))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_tokens = 0
        for idx in order:
            operands = train_set[idx]
            prompt = prompt_tokens(operands)
            style = rng.choice(STYLES) if arm == "trace" else None
            completion = gold_completion(operands, arm, style)
            learning_rate = lr0 * (1.0 - (1.0 - LR_FINAL_FRACTION) * step / total_steps)
            epoch_loss += train_sequence(model, prompt + completion, len(prompt), learning_rate)
            epoch_tokens += len(completion)
            step += 1
        losses.append(epoch_loss / epoch_tokens)
    return model, losses


def mean_nll(model: Model, arm: str, problems: list[Problem]) -> float:
    """Mean NLL per completion token; the trace arm is scored under all three routes.

    Scoring all three routes (not only the one sampled in training) makes G1 ask whether the
    model learned every route, not just whichever it happened to favor.
    """
    total = 0.0
    n = 0
    for operands in problems:
        prompt = prompt_tokens(operands)
        for style in STYLES if arm == "trace" else (None,):
            tokens = prompt + gold_completion(operands, arm, style)
            z = [0.0] * len(model["b1"])
            for s in range(len(prompt)):
                z = add_row(z, model["E"][s][tokens[s]])
            for t in range(len(prompt), len(tokens)):
                _, logits = head(model, z, t)
                total -= math.log(max(softmax(logits)[tokens[t]], 1e-12))
                n += 1
                z = add_row(z, model["E"][t][tokens[t]])
    return total / n


def param_distance(a: Model, b: Model) -> float:
    """L1 distance over the slot tables and W2: proof that training moved the weights."""
    d = 0.0
    for s in range(SLOTS):
        for tok in range(V + 1):
            d += sum(abs(x - y) for x, y in zip(a["E"][s][tok], b["E"][s][tok]))
    d += sum(abs(x - y) for ra, rb in zip(a["W2"], b["W2"]) for x, y in zip(ra, rb))
    return d


# Counts every allocated weight, including the never-read pad rows (17,345 in total).
def n_params(model: Model) -> int:
    return (
        sum(len(r) for s in model["E"] for r in s)
        + sum(len(r) for r in model["Q"])
        + len(model["b1"])
        + sum(len(r) for r in model["W2"])
        + len(model["b2"])
    )


# === INFERENCE: DECODING (prompt tokens and weights only) ===
# Greedy decoding takes the argmax token at every step; sampling draws from softmax(l / tau).
# A route is chosen by the very first generated token, so greedy commits to one route per
# problem, while sampling spreads over routes -- the diversity the vote below feeds on.
# Forced tokens appear only inside the labeled probes, which overwrite chosen positions to ask
# "what does the model write next if this had been written here?".


def generate(model: Model, prompt: list[int], rng: random.Random, tau: float, greedy: bool,
             max_new: int = MAX_NEW,
             forced: dict[int, int] | None = None) -> tuple[list[int], list[float]]:
    """Return (generated tokens, log-probability of each at temperature 1).

    The temperature-1 log-probabilities feed Eq. 1's path weight regardless of tau, so a hot
    sample is weighted by how likely the unscaled model finds it. forced = {offset: token}.
    """
    hidden_dim = len(model["b1"])
    z = [0.0] * hidden_dim
    for s, tok in enumerate(prompt):
        z = add_row(z, model["E"][s][tok])
    out = []
    logps = []
    for step in range(max_new):
        t = len(prompt) + step
        _, logits = head(model, z, t)
        probs1 = softmax(logits)
        if forced is not None and step in forced:
            tok = forced[step]
        elif greedy:
            tok = max(range(V), key=lambda i: (logits[i], -i))  # ties go to the lowest id
        else:
            # Inverse-CDF sampling: the first token whose cumulative probability exceeds a
            # uniform draw. If rounding leaves the total just below the draw, take the last id.
            probs = softmax(logits, tau)
            r = rng.random()
            acc = 0.0
            tok = V - 1
            for i, p in enumerate(probs):
                acc += p
                if r < acc:
                    tok = i
                    break
        out.append(tok)
        logps.append(math.log(max(probs1[tok], 1e-12)))
        z = add_row(z, model["E"][t][tok])  # the model reads what it just wrote
        if tok == END:
            break
    return out, logps


def parse_answer(generated: list[int], arm: str) -> int | None:
    """Wang et al.'s task parser (footnote 1), here: the digit right after the first '#'.

    A sample without '#', or with a non-digit after it, casts no vote. The parser reads only
    the model's own tokens; it never checks whether the written partial sums add up.
    """
    if arm == "direct":
        return generated[0] if generated and generated[0] < 10 else None
    if HASH not in generated:
        return None
    i = generated.index(HASH)
    if i + 1 < len(generated) and generated[i + 1] < 10:
        return generated[i + 1]
    return None


# The exact 7-token shape a trained trace should have; the probes skip traces that miss it,
# but the parser and the vote never require it (Wang et al. vote on any parseable answer).
def well_formed(out: list[int]) -> bool:
    return (
        len(out) == TRACE_LEN
        and out[0] in STYLES
        and all(x < 10 for x in (out[1], out[2], out[3], out[5]))
        and out[4] == HASH
        and out[6] == END
    )


# === INFERENCE: AGGREGATION (Wang et al. Sec. 2, Eq. 1, Table 1) ===
# Self-consistency marginalizes the reasoning path out: answer = argmax_a sum_i 1(a_i = a).
# Intuition: a wrong route can land on many different wrong digits, but every correct route
# lands on the same digit, so with enough samples the correct answer collects the most votes
# even when no single route is right most of the time.
# Eq. 1 gives each sampled path a length-normalized probability,
#   P(r_i, a_i) = exp( (1/K) * sum_k log P(t_k | prompt, t_1..t_{k-1}) )
# where K counts the sample's generated tokens. Table 1 aggregations, by name below:
#   majority  unweighted sum (the vote)        wsum_raw   sum of exp(sum log p)
#   wsum_norm sum of Eq. 1 weights             wavg_raw / wavg_norm  those sums / vote count
# Tie rule (ours; the paper does not specify one): among answers tied on votes, take the
# higher sum of Eq. 1 weights; if that also ties, the smallest digit. No valid sample at all
# means the vote abstains, which is scored as wrong -- never replaced by a guess.
RULES = ("majority", "wsum_raw", "wsum_norm", "wavg_raw", "wavg_norm")


def tally(samples: list[Sample]) -> tuple[dict, dict, dict]:
    """Per answer: vote count, sum of Eq. 1 weights, sum of unnormalized path probabilities."""
    votes, norm_w, raw_w = {}, {}, {}
    for ans, sum_logp, n_tok in samples:
        if ans is None:  # unparseable samples cast no vote
            continue
        votes[ans] = votes.get(ans, 0) + 1
        norm_w[ans] = norm_w.get(ans, 0.0) + math.exp(sum_logp / n_tok)  # Equation 1
        raw_w[ans] = raw_w.get(ans, 0.0) + math.exp(sum_logp)
    return votes, norm_w, raw_w


def aggregate(samples: list[Sample], rule: str) -> tuple[int | None, str]:
    """samples: (answer or None, sum_logp, n_tokens). Returns (answer or None, tie_info)."""
    votes, norm_w, raw_w = tally(samples)
    if not votes:
        return None, "abstain"
    if rule == "majority":
        best = max(votes.values())
        tied = sorted(a for a, c in votes.items() if c == best)
        if len(tied) == 1:
            return tied[0], "none"
        top_w = max(norm_w[a] for a in tied)
        tied_w = sorted(a for a in tied if norm_w[a] == top_w)
        return tied_w[0], ("eq1-weight" if len(tied_w) == 1 else "smallest-digit")
    if rule == "wsum_raw":
        score = raw_w
    elif rule == "wsum_norm":
        score = norm_w
    elif rule == "wavg_raw":
        score = {a: raw_w[a] / votes[a] for a in votes}
    elif rule == "wavg_norm":
        score = {a: norm_w[a] / votes[a] for a in votes}
    else:
        raise ValueError(rule)
    best = max(score.values())
    return sorted(a for a in score if score[a] == best)[0], "none"


def vote_unit_checks() -> float:
    """G5: aggregator arithmetic on hand-made samples (no model), then a binomial identity.

    The identity: if each of 5 independent votes is correct with p = 0.7 and a wrong vote
    always names the same other answer, the majority is correct with probability
    sum_{j=3..5} C(5, j) p^j (1 - p)^(5 - j); enumerating all 32 outcomes through aggregate()
    must give the same number.
    """
    lp = math.log(0.5)

    def s(answer: int | None) -> Sample:  # a synthetic 3-token sample with probability 0.5^3
        return (answer, lp * 3, 3)

    assert aggregate([s(3), s(3), s(5)], "majority") == (3, "none")
    assert aggregate([s(1), s(2)], "majority")[1] in ("eq1-weight", "smallest-digit")
    confident = [(7, math.log(0.9) * 3, 3), (4, math.log(0.1) * 3, 3)]
    assert aggregate(confident, "majority") == (7, "eq1-weight")  # tie broken by Eq. 1 weight
    assert aggregate([s(6), s(2)], "majority") == (2, "smallest-digit")  # equal weights too
    assert aggregate([], "majority") == (None, "abstain")
    assert aggregate([s(None), s(None)], "majority") == (None, "abstain")
    assert aggregate([s(9)], "majority") == (9, "none")  # k = 1 is just that sample
    assert aggregate([s(None), s(4), s(None)], "majority") == (4, "none")  # invalid ones skipped
    assert aggregate([s(1), s(2), s(2), s(1), s(3)], "majority")[0] in (1, 2)  # 2-way tie of 5
    # Count versus weight: two unlikely votes for 1 outvote one likely vote for 2 by count only.
    mixed = [(1, math.log(0.01) * 3, 3), (1, math.log(0.01) * 3, 3), (2, math.log(0.9) * 3, 3)]
    assert aggregate(mixed, "majority")[0] == 1 and aggregate(mixed, "wsum_norm")[0] == 2
    assert aggregate(mixed, "wavg_raw")[0] == 2
    # Averages ignore how many samples agree: one confident vote beats four average ones.
    many = [s(1)] * 4 + [(2, math.log(0.9) * 3, 3)]
    assert aggregate(many, "majority")[0] == 1 and aggregate(many, "wsum_norm")[0] == 1
    assert aggregate(many, "wavg_norm")[0] == 2 and aggregate(many, "wavg_raw")[0] == 2
    p = 0.7
    closed = sum(math.comb(5, j) * p**j * (1 - p) ** (5 - j) for j in range(3, 6))
    enum = 0.0
    for mask in range(32):
        bits = [(mask >> i) & 1 for i in range(5)]
        prob = 1.0
        for b in bits:
            prob *= p if b else 1 - p
        votes = [(1 if b else 0, lp * 3, 3) for b in bits]
        if aggregate(votes, "majority")[0] == 1:
            enum += prob
    assert abs(closed - enum) < 1e-12, (closed, enum)
    return closed


# === INFERENCE: STATISTICS ===


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n (better behaved near 0 or 1 than
    p +/- z sqrt(p(1-p)/n)): center = (p + z^2/2n) / (1 + z^2/n), half-width =
    z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - r, c + r)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test for two decoders scored on the same problems.

    Only discordant problems carry information: b = only the first decoder right, c = only the
    second. Under "no difference" b ~ Binomial(b + c, 1/2), so p = min(1, 2 P(X <= min(b, c))).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt_acc(correct: int, n: int) -> str:
    lo, hi = wilson(correct, n)
    return f"{correct / n:.3f} ({correct}/{n}; 95% Wilson {lo:.3f}-{hi:.3f})"


# Greedy accuracy scores the model's own decision: the parsed answer of its argmax trace.
# A malformed trace with no parseable answer is simply wrong, as in the vote.
def greedy_accuracy(model: Model, arm: str, problems: list[Problem], rng: random.Random) -> list:
    ok = []
    for operands in problems:
        out, _ = generate(model, prompt_tokens(operands), rng, 1.0, True)
        ok.append(parse_answer(out, arm) == sum(operands) % 10)
    return ok


# === INFERENCE: REPORT-ONLY FAITHFULNESS PROBES ===
# Does the written trace drive the answer, or is it decoration? Each probe takes the model's
# own greedy trace, overwrites some tokens, and lets the model continue greedily:
#   (a) write p3 + 1 as the last partial sum: does the answer copy the written p3?
#   (b) write p1 + 1 as the first partial sum: do the model's later sums chain from it?
#   (c) write a whole internally consistent trace from p1 + 1: does the answer follow it?
# The probes are interventions on inputs the model never saw in training (a prompt and a
# trace that disagree), so they show partial dependence, not a faithful causal chain (D6).


def faithfulness_probes(model: Model, problems: list[Problem], label: str) -> None:
    """Report-only probes (a)-(c); every classified value is the model's own greedy output."""
    skipped = 0
    a_stats: Counter[str] = Counter()
    b_stats: Counter[str] = Counter()
    c_stats: Counter[str] = Counter()
    by_style: dict[int, Counter[str]] = {s: Counter() for s in STYLES}
    rng = random.Random(0)  # unused by greedy decoding; required by the generate signature
    for operands in problems:
        prompt = prompt_tokens(operands)
        own, _ = generate(model, prompt, rng, 1.0, True)
        if not well_formed(own):  # nothing to intervene on; counted and reported
            skipped += 1
            continue
        style, p1, p2, p3 = own[0], own[1], own[2], own[3]
        true_answer = sum(operands) % 10
        # (a) last-step forcing: keep the model's own route, p1 and p2, write p3 + 1 and '#'.
        forced_a = {0: style, 1: p1, 2: p2, 3: (p3 + 1) % 10, 4: HASH}
        out_a, _ = generate(model, prompt, rng, 1.0, True, forced=forced_a)
        ans_a = parse_answer(out_a, "trace")
        key = ("equals-forced-p3" if ans_a == (p3 + 1) % 10
               else "equals-original-p3" if ans_a == p3 else "neither")
        a_stats[key] += 1
        by_style[style]["a:" + key] += 1
        # (b) first-step chaining: write p1 + 1, then compare the model's p2 and p3 with the
        # values that chain from it ("chains") and with the true route values ("prompt-true").
        t1 = (p1 + 1) % 10
        out_b, _ = generate(model, prompt, rng, 1.0, True, forced={0: style, 1: t1})
        chain = partials(operands, style, p1_override=t1)
        truth = partials(operands, style)
        if len(out_b) >= 4 and out_b[2] < 10 and out_b[3] < 10:
            if style != STYLES[2]:  # route T's p2 = c + d ignores p1, so only p3 can chain
                k2 = ("chains" if out_b[2] == chain[1]
                      else "prompt-true" if out_b[2] == truth[1] else "neither")
                b_stats["p2:" + k2] += 1
            k3 = ("chains" if out_b[3] == chain[2]
                  else "prompt-true" if out_b[3] == truth[2] else "neither")
            b_stats["p3:" + k3] += 1
            by_style[style]["b-p3:" + k3] += 1
            if style != STYLES[2] and out_b[2] == chain[1] and out_b[3] == chain[2]:
                b_stats["joint:both-chain"] += 1
            if style != STYLES[2]:
                b_stats["joint:denominator"] += 1
        else:
            b_stats["unparseable"] += 1
        # (c) internally consistent corrupted trace: every partial sum follows from p1 + 1, so
        # trace and prompt disagree only about the answer; which one does the answer token read?
        forced_c = {0: style, 1: chain[0], 2: chain[1], 3: chain[2], 4: HASH}
        out_c, _ = generate(model, prompt, rng, 1.0, True, forced=forced_c)
        ans_c = parse_answer(out_c, "trace")
        kc = ("equals-forced-p3" if ans_c == chain[2]
              else "equals-prompt-true" if ans_c == true_answer else "neither")
        c_stats[kc] += 1
        by_style[style]["c:" + kc] += 1
    used = len(problems) - skipped

    def share_list(stats: Counter[str], prefix: str, keys: tuple[str, ...], denom: int) -> str:
        return ", ".join(f"{k}={stats[prefix + k]} ({stats[prefix + k] / max(denom, 1):.3f})"
                         for k in keys)

    n_p3 = sum(v for k, v in b_stats.items() if k.startswith("p3:"))
    n_p2 = sum(v for k, v in b_stats.items() if k.startswith("p2:"))
    b_keys = ("chains", "prompt-true", "neither")
    print(f"faithfulness probes on {label} (n={len(problems)}, "
          f"malformed greedy traces skipped={skipped}, used={used})")
    print("  (a) force p3+1 then read answer: "
          + share_list(a_stats, "", ("equals-forced-p3", "equals-original-p3", "neither"), used))
    print(f"  (b) force p1+1 then read later steps: p3 over {n_p3} problems: "
          + share_list(b_stats, "p3:", b_keys, n_p3))
    print(f"      p2 (styles L,R only) over {n_p2}: " + share_list(b_stats, "p2:", b_keys, n_p2)
          + f"; joint both-chain {b_stats['joint:both-chain']}/{b_stats['joint:denominator']}"
          + f"; unparseable {b_stats['unparseable']}")
    print("  (c) internally consistent corrupted trace then read answer: "
          + share_list(c_stats, "", ("equals-forced-p3", "equals-prompt-true", "neither"), used))
    for s in STYLES:
        print(f"      style {CHARS[s]}: "
              + " ".join(f"{k}={v}" for k, v in sorted(by_style[s].items())))


def explain_vote(operands: Problem, gold_answer: int, samples: list[Sample]) -> None:
    """Print the voting arithmetic of one problem: counts, Eq. 1 weights, and the decision."""
    votes, norm_w, _ = tally(samples)
    invalid = sum(1 for ans, _, _ in samples if ans is None)
    answer, tie_info = aggregate(samples, "majority")
    counts = ", ".join(f"{a}:{votes[a]}" for a in sorted(votes))
    weights = ", ".join(f"{a}:{norm_w[a]:.4f}" for a in sorted(norm_w))
    print(f"  {decode_text(prompt_tokens(operands))} gold={gold_answer}")
    print(f"    votes per answer {{{counts}}}; samples with no parseable answer={invalid}")
    print(f"    summed Eq. 1 weights {{{weights}}}")
    print(f"    vote -> {answer} (tie rule used: {tie_info})")


# === MAIN ===
# Order of the run (and of every random draw): aggregator self-checks, data and leakage checks,
# training both arms, greedy evaluation and inspection, the self-consistency temperature sweep,
# the gain and its arithmetic, then report-only probes and the gate summary.


def main() -> None:
    t_start = time.time()
    print("microcot: 4-digit sum mod 10; direct vs three-path trace; greedy and self-consistency")
    print(f"config seed={SEED} split_seed={SPLIT_SEED} hidden={HIDDEN_DIM} "
          f"trace(lr={TRACE_LEARNING_RATE},epochs={TRACE_EPOCHS}) "
          f"direct(lr={DIRECT_LEARNING_RATE},epochs={DIRECT_EPOCHS}) samples={NUM_SAMPLES} "
          f"tau={PRIMARY_TAU} taus={list(TAU_SWEEP)}")

    print("\n=== AGGREGATOR UNIT CHECKS (synthetic, no model) ===")
    closed = vote_unit_checks()
    print("all aggregator checks passed; P(majority of 5 correct | p=0.7) closed form = "
          f"enumeration = {closed:.6f}")

    # Data checks come before any training, so a leaking split stops the run before it costs
    # minutes. The lookup baselines ask whether a few operand positions alone predict the answer.
    print("\n=== SPLITS AND LEAKAGE (frozen partition) ===")
    splits, info = build_splits()
    ms = check_splits(splits, info)
    sizes = info["pool_sizes"]
    print(f"held-out lists: {len(info['test_ms'])} test and {len(info['reserved_ms'])} reserved "
          f"development multisets (embedded; SHA-256 digest matches); train multisets="
          f"{len(info['train_ms'])} of {info['all_multisets']}")
    print(f"pools: train={sizes[0]} reserved dev={sizes[1]} (excluded, never evaluated) "
          f"test={sizes[2]} problems; used train={len(splits['train'])} "
          f"test={len(splits['test'])}; test slice covers {len(ms['test'])} multisets, "
          f"train {len(ms['train'])}")
    print("overlaps (asserted zero): multiset train&reserved/train&test/reserved&test and "
          "ordered-tuple train&reserved/train&test/reserved&test = 0")
    for name in ("train", "test"):
        print(f"permutation-class shares {name}: {shares(splits[name])}")
    all_tuples = [tuple(int(c) for c in f"{n:04d}") for n in range(10**4)]
    print("permutation-class shares all 10,000 tuples: " + shares(all_tuples))
    answers = {k: [sum(o) % 10 for o in v] for k, v in splits.items()}
    for k in ("train", "test"):
        print(f"answer histogram {k}: {[answers[k].count(a) for a in range(10)]}")
    cov = step_pair_coverage(splits["train"], splits["test"])
    print("step-pair coverage per path L,R,T (pairs seen in train of 100; test problems with "
          f"every pair seen of {N_TEST}): {[c[0] for c in cov]} ; {[c[1] for c in cov]}")
    lookup, prior_acc = lookup_baselines(splits["train"], splits["test"])
    print(f"lookup baselines on test (fitted on train): answer-prior={prior_acc:.3f} "
          f"best-1-position={lookup[1]:.3f} best-2-position={lookup[2]:.3f} "
          f"best-3-position={lookup[3]:.3f} (gate: each at most {G2_LOOKUP_MAX})")

    # Both arms start from the SAME initial weights, so differences between them come from the
    # training targets (direct answer versus trace) and their unequal budgets, not from init.
    print("\n=== TRAINING ===")
    base = init_model(HIDDEN_DIM, random.Random(SEED))
    print(f"parameters per arm: {n_params(base)}; identical initialization for both arms")
    arm_cfg = {"direct": (DIRECT_EPOCHS, DIRECT_LEARNING_RATE),
               "trace": (TRACE_EPOCHS, TRACE_LEARNING_RATE)}
    models, g1_ok = {}, {}
    for arm in ("direct", "trace"):
        n_ep, lr0 = arm_cfg[arm]
        init_nll = mean_nll(base, arm, splits["train"])
        t0 = time.time()
        model, losses = train_arm(arm, base, splits["train"], n_ep, lr0)
        train_seconds = time.time() - t0
        final_nll = mean_nll(model, arm, splits["train"])
        dist = param_distance(base, model)
        models[arm] = model
        g1_ok[arm] = final_nll <= G1_NLL_RATIO * init_nll and dist > 0
        # Supervised tokens per problem: 2 for "answer ." and 7 for a full trace; this is the
        # budget gap D4 refers to.
        style0 = STYLES[0] if arm == "trace" else None
        tok_per_problem = len(gold_completion(splits["train"][0], arm, style0))
        print(f"arm={arm} epochs={n_ep} lr={lr0}: train NLL/token init={init_nll:.4f} "
              f"final={final_nll:.4f} ratio={final_nll / init_nll:.3f} param-L1-change={dist:.1f}")
        print(f"  budget: per-sequence updates={n_ep * N_TRAIN} supervised tokens per problem per "
              f"epoch={tok_per_problem} total supervised tokens={n_ep * N_TRAIN * tok_per_problem}")
        if arm == "direct":  # the full curve shows the long plateau before the late drop (D4)
            print("  direct per-epoch NLL/token (all epochs): "
                  + " ".join(f"{x:.4f}" for x in losses))
        else:
            print("  trace per-epoch NLL/token (every 5th epoch): "
                  + " ".join(f"{x:.4f}" for x in losses[::5]))
        print(f"[timing] arm={arm} training seconds={train_seconds:.1f}")
    print("extended 270-epoch direct control: omitted by design (about 130 s extra would put the "
          "frozen runtime budget at risk)")
    print("budget note: the arms are not matched in epochs, updates, tokens or seconds; the direct "
          "arm is the cheaper one")

    # Greedy first: Wang et al.'s baseline is greedy decoding of the same chain-of-thought model.
    # Greedy consumes no random numbers, so gen_rng only advances in the 12 inspection samples.
    direct, trace = models["direct"], models["trace"]
    print("\n=== TEST SLICE (500 problems; the only evaluation split of this program) ===")
    test = splits["test"]
    n = len(test)
    gold = [sum(o) % 10 for o in test]  # scoring only; no decoder below receives these
    gen_rng = random.Random(SEED + 2)
    direct_ok = greedy_accuracy(direct, "direct", test, gen_rng)
    trace_ok = greedy_accuracy(trace, "trace", test, gen_rng)
    print(f"direct greedy accuracy: {fmt_acc(sum(direct_ok), n)}")
    print(f"trace greedy accuracy: {fmt_acc(sum(trace_ok), n)}")
    b = sum(1 for x, y in zip(trace_ok, direct_ok) if x and not y)
    c = sum(1 for x, y in zip(trace_ok, direct_ok) if y and not x)
    print(f"trace minus direct = {(sum(trace_ok) - sum(direct_ok)) / n:+.3f}; discordant "
          f"trace-only={b} direct-only={c}; exact McNemar p={mcnemar_exact(b, c):.3g}")

    print("\ngreedy trace inspection (model-generated; first 8 test problems):")
    wf = 0
    greedy_styles: Counter[int] = Counter()
    for i, operands in enumerate(test):
        out, _ = generate(trace, prompt_tokens(operands), gen_rng, 1.0, True)
        wf += well_formed(out)
        if out and out[0] in STYLES:
            greedy_styles[out[0]] += 1
        if i < 8:
            print(f"  {decode_text(prompt_tokens(operands))} -> {decode_text(out)} "
                  f"parsed={parse_answer(out, 'trace')} gold={gold[i]}")
    print(f"greedy trace well-formed: {wf}/{n}; greedy style choices L,R,T = "
          f"{[greedy_styles[s] for s in STYLES]}")
    # These 12 inspection samples come from gen_rng, never from the vote's sampling stream.
    print("sampled traces at tau 0.7 (first 3 test problems, 4 samples each):")
    for operands, g in list(zip(test, gold))[:3]:
        for _ in range(4):
            out, _ = generate(trace, prompt_tokens(operands), gen_rng, PRIMARY_TAU, False)
            print(f"  {decode_text(prompt_tokens(operands))} -> {decode_text(out)} "
                  f"parsed={parse_answer(out, 'trace')} gold={g}")

    # Self-consistency proper. One stream, consumed in a fixed order: temperatures, then
    # problems, then 40 samples, so the tau = 0.7 samples depend on the 0.3 and 0.5 sweeps
    # having run first. Each k reuses the first k of the same 40 samples (nested votes).
    print("\n=== SELF-CONSISTENCY (sampling over the test slice) ===")
    sample_rng = random.Random(SEED + 3)
    primary_samples = None
    primary_style_counts = None
    for tau in TAU_SWEEP:
        per_problem = []
        style_counts = []
        for operands in test:
            prompt = prompt_tokens(operands)
            samples = []
            sc: Counter[int] = Counter()
            for _ in range(NUM_SAMPLES):
                out, lps = generate(trace, prompt, sample_rng, tau, False)
                samples.append((parse_answer(out, "trace"), sum(lps), len(out)))
                if out and out[0] in STYLES:
                    sc[out[0]] += 1
            per_problem.append(samples)
            style_counts.append(sc)
        # Single-sample accuracy is what one sampled trace achieves on average; temperature
        # trades it away for diversity. The vote's job is to win that accuracy back and more.
        n_valid = sum(1 for s in per_problem for x in s if x[0] is not None)
        single = sum(1 for s, g in zip(per_problem, gold) for x in s if x[0] == g)
        denom = n * NUM_SAMPLES
        print(f"tau={tau}: single-sample accuracy={single / denom:.3f} "
              f"(valid-parse rate {n_valid / denom:.3f}) over {denom} samples")
        for k in KS:
            cells = []
            ties = abst = 0
            tie_levels: Counter[str] = Counter()
            for rule in RULES:
                hits = 0
                for s, g in zip(per_problem, gold):
                    ans, info_t = aggregate(s[:k], rule)
                    hits += ans == g
                    if rule == "majority":
                        abst += info_t == "abstain"
                        if info_t in ("eq1-weight", "smallest-digit"):
                            ties += 1
                            tie_levels[info_t] += 1
                cells.append(hits / n)
            # oracle@k: some sample among the first k was right -- a ceiling for any vote rule
            oracle = sum(1 for s, g in zip(per_problem, gold) if any(x[0] == g for x in s[:k])) / n
            print(f"  k={k:>2} majority={cells[0]:.3f} ties={ties} {dict(tie_levels)} "
                  f"abstain={abst} | wsum_raw={cells[1]:.3f} wsum_norm={cells[2]:.3f} "
                  f"wavg_raw={cells[3]:.3f} wavg_norm={cells[4]:.3f} | oracle@k={oracle:.3f}")
        if tau == PRIMARY_TAU:
            primary_samples = per_problem
            primary_style_counts = style_counts
    assert primary_samples is not None and primary_style_counts is not None

    # G3(b): a vote can only exploit diverse routes if a single problem's samples actually use
    # several of them; pooled shares alone could hide problems that always take one route.
    print(f"\n=== G3(b) DIVERSITY AND G4 SELF-CONSISTENCY GAIN at tau={PRIMARY_TAU} ===")
    diverse = sum(1 for sc in primary_style_counts
                  if sum(1 for s in STYLES if sc[s] >= G3_STYLE_MIN_SAMPLES) >= 2)
    pooled = [sum(sc[s] for sc in primary_style_counts) / (n * NUM_SAMPLES) for s in STYLES]
    print(f"problems with at least two style tokens each in >= {G3_STYLE_MIN_SAMPLES} of "
          f"{NUM_SAMPLES} samples: {diverse}/{n} ({diverse / n:.3f}); "
          f"pooled style shares L,R,T = {[round(x, 3) for x in pooled]}")
    # G4 compares the vote with greedy on the SAME 500 problems, so the test is paired: only
    # problems where exactly one decoder is right (SC-only, greedy-only) count in McNemar.
    # k = 1 is a single sample, usually worse than greedy; the gain must come from voting.
    greedy = trace_ok
    vote = {k: [aggregate(s[:k], "majority")[0] == g for s, g in zip(primary_samples, gold)]
            for k in KS}
    for k in KS:
        v = vote[k]
        bb = sum(1 for x, y in zip(v, greedy) if x and not y)
        cc = sum(1 for x, y in zip(v, greedy) if y and not x)
        print(f"SC k={k:>2}: {fmt_acc(sum(v), n)} vs greedy {fmt_acc(sum(greedy), n)}; "
              f"diff={(sum(v) - sum(greedy)) / n:+.3f} ({sum(v) - sum(greedy):+d} problems); "
              f"SC-only={bb} greedy-only={cc}; exact McNemar p={mcnemar_exact(bb, cc):.3g}")
    # Wrong votes by offset (vote - gold) mod 10: the pattern shows whether the vote's mistakes
    # are scattered or systematic (all samples sharing one wrong step).
    wrong: Counter[str] = Counter()
    for s, g in zip(primary_samples, gold):
        maj, _ = aggregate(s, "majority")
        if maj != g:
            wrong["abstain" if maj is None else f"off-by-{(maj - g) % 10}"] += 1
    print(f"wrong SC(k={NUM_SAMPLES}) answers by offset from gold: {dict(sorted(wrong.items()))}")

    print(f"\n=== VOTING ARITHMETIC (k={NUM_SAMPLES}, tau={PRIMARY_TAU}) ===")
    print("first test problem:")
    explain_vote(test[0], gold[0], primary_samples[0])
    # Ties are common at small k (two answers with equal counts); the Eq. 1 weight then
    # prefers the answer whose traces the model found more probable on average.
    tied = [i for i, s in enumerate(primary_samples) if aggregate(s, "majority")[1] != "none"]
    if tied:
        print(f"first test problem whose vote ties (problem #{tied[0]}):")
        explain_vote(test[tied[0]], gold[tied[0]], primary_samples[tied[0]])
    else:
        print("no test problem has a tied vote at this k")

    # Report-only: route accuracy with the route token forced, and whether routes fail on the
    # same problems. The vote gains where routes disagree and the right answer is the common one.
    print("\n=== REPORT-ONLY: forced-path accuracy and path error overlap (test slice) ===")
    forced_ok = {}
    for st in STYLES:
        ok = []
        for operands, g in zip(test, gold):
            out, _ = generate(trace, prompt_tokens(operands), gen_rng, 1.0, True, forced={0: st})
            ok.append(parse_answer(out, "trace") == g)
        forced_ok[st] = ok
        print(f"path {CHARS[st]} only (style forced, rest greedy): {fmt_acc(sum(ok), n)}")
    # Oracle: some forced route is right. Overlap: if routes failed independently, both-wrong
    # counts would match wrong1 * wrong2 / n; larger counts mean shared mistakes, capping the vote.
    any_route = sum(1 for i in range(n) if any(forced_ok[st][i] for st in STYLES)) / n
    print(f"oracle over the three forced greedy paths (any correct): {any_route:.3f}")
    overlap = {}
    for s1, s2 in combinations(STYLES, 2):
        both = sum(1 for i in range(n) if not forced_ok[s1][i] and not forced_ok[s2][i])
        w1 = sum(1 for i in range(n) if not forced_ok[s1][i])
        w2 = sum(1 for i in range(n) if not forced_ok[s2][i])
        overlap[f"{CHARS[s1]}{CHARS[s2]}"] = (both, w1, w2, round(w1 * w2 / n, 1))
    print(f"path error overlap (both wrong, wrong1, wrong2, expected-if-independent): {overlap}")
    # D3: the honest comparison for "does voting beat simply taking the best route?". The
    # best route is picked by looking at this same test slice, which flatters it.
    best_route = max(STYLES, key=lambda st: (sum(forced_ok[st]), -st))
    margin = sum(vote[NUM_SAMPLES]) - sum(forced_ok[best_route])
    print(f"post hoc: SC k={NUM_SAMPLES} minus the best single forced path ({CHARS[best_route]}, "
          f"chosen after seeing this test slice) = {margin:+d} problems ({margin / n:+.3f})")

    print()
    faithfulness_probes(trace, test, "test slice")

    # D8: the held-out slice over-represents repeated digits, so results are also split by
    # permutation class; classes under CLASS_MIN_N problems are too small to read much into.
    print("\n=== REPORT-ONLY: per permutation class (test slice) ===")
    for name in CLASSES:
        idx = [i for i, p in enumerate(test) if perm_class(p) == name]
        if not idx:
            print(f"{name}: n=0")
            continue
        m = len(idx)
        d_ok = sum(direct_ok[i] for i in idx)
        t_ok = sum(trace_ok[i] for i in idx)
        v40 = sum(vote[40][i] for i in idx)
        flag = " (descriptive only, n<%d)" % CLASS_MIN_N if m < CLASS_MIN_N else ""
        print(f"{name}: n={m} direct greedy={d_ok / m:.3f} trace greedy={t_ok / m:.3f} "
              f"SC k=40={v40 / m:.3f} SC-minus-greedy={(v40 - t_ok) / m:+.3f} "
              f"({v40 - t_ok:+d} problems){flag}")

    # Every gate below is recomputed from this run's own counts against the frozen thresholds.
    # G6 (two identical runs) and G7 (runtime on the qualification host) need runs outside this
    # one, and G8 (exact reproduction of the design-study numbers) is checked by comparison.
    print("\n=== GATE SUMMARY (computed in-program from the frozen thresholds) ===")
    t_acc = sum(trace_ok) / n
    g1 = g1_ok["direct"] and g1_ok["trace"]
    g2 = max(lookup.values()) <= G2_LOOKUP_MAX
    g3a = t_acc >= G3_FLOOR
    g3b = diverse / n >= G3_PROBLEM_FRACTION
    gain_problems = sum(vote[40]) - sum(greedy)
    b40 = sum(1 for x, y in zip(vote[40], greedy) if x and not y)
    c40 = sum(1 for x, y in zip(vote[40], greedy) if y and not x)
    p_val = mcnemar_exact(b40, c40)
    g4 = gain_problems / n >= G4_GAIN and p_val < G4_ALPHA and sum(vote[10]) >= sum(greedy)
    word = {True: "PASS", False: "FAIL"}
    print(f"G1 training occurred: direct {word[g1_ok['direct']]}, trace {word[g1_ok['trace']]} "
          f"-> {word[g1]}")
    print(f"G2 leakage blocked: overlaps asserted zero; max lookup baseline "
          f"{max(lookup.values()):.3f} <= {G2_LOOKUP_MAX} -> {word[g2]}")
    print(f"G3(a) trace greedy {t_acc:.3f} >= {G3_FLOOR} -> {word[g3a]}")
    print(f"G3(b) diverse problems {diverse}/{n} = {diverse / n:.3f} >= {G3_PROBLEM_FRACTION} "
          f"-> {word[g3b]}")
    print(f"G4 SC k=40 tau 0.7 minus greedy = {gain_problems:+d} problems "
          f"({gain_problems / n:+.3f}, need >= {G4_GAIN:+.2f}); exact McNemar p={p_val:.3g} "
          f"(need < {G4_ALPHA}); SC k=10 {sum(vote[10])} vs greedy {sum(greedy)} "
          f"(need not lower) -> {word[g4]}")
    print("G5 aggregation arithmetic: unit checks and binomial identity passed (asserted above); "
          "ties and abstentions reported above -> PASS")
    print("G6 determinism: checked outside this run (two default runs must match except "
          "[timing] lines)")
    print("G7 runtime: checked outside this run (whole default program at most 360 s under the "
          "pinned CPython 3.12.8 on the M-series qualification host)")
    print(f"[timing] total seconds={time.time() - t_start:.1f}")
    gates = (("G1", g1), ("G2", g2), ("G3(a)", g3a), ("G3(b)", g3b), ("G4", g4))
    failed = [name for name, ok in gates if not ok]
    if failed:  # fail loudly: a missed pre-registered gate is never reported as a pass
        raise RuntimeError(f"pre-registered gate(s) failed: {', '.join(failed)}")
    print("VERDICT for the in-program gates G1-G5: all pass")


if __name__ == "__main__":
    main()

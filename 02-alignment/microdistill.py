"""
Knowledge distillation from first principles: a trained, frozen teacher network passes on its
temperature-softened class probabilities -- not just its top answer -- to a much smaller
student, which learns from those soft targets and the true labels together.
"""
# Reference: Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (2015).
# https://arxiv.org/abs/1503.02531 -- Section 2, Eq. 1 (softmax with temperature T) and
# Section 2.1, Eq. 2 (the gradient of the soft-target cross-entropy with respect to a
# student logit), the T^2 scaling of soft-target gradients when mixed with hard targets,
# and inference at temperature 1. The paper distils large MNIST/speech models and
# ensembles; this script distils a 99-parameter MLP into a 27-parameter MLP on synthetic
# 2-D clusters. T = 2 and alpha = 0.9 are toy choices, not values the paper prescribes.

# === TRADEOFFS ===
# + Soft targets carry how the teacher ranks the WRONG classes, which one-hot labels lack
# + The student can be far smaller and cheaper to run than the teacher
# + Needs no new labels: the teacher labels any input it can score
# - Requires a trained teacher first, so total training cost goes up, not down
# - A student can only approximate the teacher; capacity limits what transfers
# - Temperature and the soft/hard mixing weight are extra hyperparameters
# WHEN TO USE: Deploying a cheaper model that should behave like an expensive one you
#   already trained (or an ensemble), especially when unlabeled inputs are plentiful.
# WHEN NOT TO: When no stronger teacher exists, or when the small model trained on
#   labels alone already meets the requirement.

from __future__ import annotations

import hashlib
import math
import random
import struct

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Data: three 2-D Gaussian clusters, labels = the cluster that generated the point
CENTERS = ((-1.0, 0.0), (1.0, 0.0), (0.0, 1.25))
CLUSTER_STD = 0.35
TRAIN_PER_CLASS = 80  # 240 training points
HELDOUT_PER_CLASS = 32  # 96 held-out points, drawn separately, report-only
NUM_CLASSES = len(CENTERS)

# Models: ReLU MLPs with biases. Teacher 2 -> 16 -> 3, student 2 -> 4 -> 3.
INPUT_DIM = 2
TEACHER_HIDDEN = 16  # 2*16 + 16 + 16*3 + 3 = 99 parameters
STUDENT_HIDDEN = 4  # 2*4 + 4 + 4*3 + 3 = 27 parameters

# Training: full-batch gradient descent, one update per epoch
TEACHER_EPOCHS = 400
STUDENT_EPOCHS = 400
LEARNING_RATE = 0.1
LOG_EVERY = 100

# Distillation: soft targets at temperature T, mixed with hard labels by ALPHA
TEMPERATURE = 2.0
ALPHA = 0.9  # weight on the soft-target term; 1 - ALPHA weighs the hard-label term

# Acceptance -- frozen before the first run and checked on TRAINING data only.
# Held-out points are reported, never used as a criterion, for tuning or for stopping.
MIN_RELATIVE_DECREASE = 0.05  # each tracked training loss must fall by at least 5%
MIN_TEACHER_ACCURACY = 0.80
EXPECTED_TEACHER_PARAMS = 99
EXPECTED_STUDENT_PARAMS = 27

# Signpost: Hinton et al. distil networks with millions of weights trained on MNIST and
# speech data. The mechanism here -- match the student's softened distribution to a
# frozen teacher's, with a T^2-scaled soft term plus a small hard-label term -- is the
# same; the scale and the task are toys.


# === DATA GENERATION ===


def make_clusters(per_class: int) -> list[tuple[float, float, int]]:
    """Draw per_class points around each center; the label is the generating cluster.

    Labels come from the data-generating process, never from a model: the teacher is
    trained on them, and the student's hard-label term uses them too.
    """
    points = []
    for label, (cx, cy) in enumerate(CENTERS):
        for _ in range(per_class):
            points.append(
                (random.gauss(cx, CLUSTER_STD), random.gauss(cy, CLUSTER_STD), label)
            )
    return points


# === MODEL DEFINITION ===

# A model is a dict of matrices: w1 [hidden][2], b1 [1][hidden], w2 [3][hidden],
# b2 [1][3]. Biases are stored as one-row matrices so every entry is a list[list[float]].
Model = dict[str, list[list[float]]]


def init_mlp(hidden: int) -> Model:
    """He-style Gaussian weights (std sqrt(2 / fan_in), suited to ReLU), zero biases."""
    return {
        "w1": [
            [random.gauss(0, math.sqrt(2 / INPUT_DIM)) for _ in range(INPUT_DIM)]
            for _ in range(hidden)
        ],
        "b1": [[0.0] * hidden],
        "w2": [
            [random.gauss(0, math.sqrt(2 / hidden)) for _ in range(hidden)]
            for _ in range(NUM_CLASSES)
        ],
        "b2": [[0.0] * NUM_CLASSES],
    }


def count_params(model: Model) -> int:
    return sum(len(row) for matrix in model.values() for row in matrix)


def weight_digest(model: Model) -> str:
    """SHA-256 of every weight's exact 8-byte IEEE-754 encoding, in a fixed order."""
    packed = b"".join(
        struct.pack("<d", v) for key in sorted(model) for row in model[key] for v in row
    )
    return hashlib.sha256(packed).hexdigest()


def forward(model: Model, x: tuple[float, float]) -> tuple[list[float], list[float]]:
    """Return (hidden activations after ReLU, output logits) for one input point."""
    hidden = [
        max(0.0, w[0] * x[0] + w[1] * x[1] + b)
        for w, b in zip(model["w1"], model["b1"][0])
    ]
    logits = [
        sum(wj * hj for wj, hj in zip(w, hidden)) + b
        for w, b in zip(model["w2"], model["b2"][0])
    ]
    return hidden, logits


def log_softmax(logits: list[float], temperature: float) -> list[float]:
    """log softmax(z / T), stabilized by subtracting the max scaled logit.

    log p_i = z_i/T - m - log(sum_j exp(z_j/T - m)), m = max_j z_j/T. The largest term in
    the sum is exp(0) = 1, so nothing overflows and every log-probability is finite.
    """
    if not (temperature > 0 and math.isfinite(temperature)):
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    scaled = [z / temperature for z in logits]
    shift = max(scaled)
    log_norm = shift + math.log(sum(math.exp(s - shift) for s in scaled))
    return [s - log_norm for s in scaled]


def softmax(logits: list[float], temperature: float) -> list[float]:
    """Eq. 1 of Hinton et al.: q_i = exp(z_i/T) / sum_j exp(z_j/T)."""
    return [math.exp(lp) for lp in log_softmax(logits, temperature)]


def accumulate_mlp_gradient(
    model: Model,
    grads: Model,
    x: tuple[float, float],
    hidden: list[float],
    dlogits: list[float],
) -> None:
    """Backpropagate dLoss/dlogits for one point through the MLP into grads (+=).

    logits = W2 h + b2  ->  dW2 = dlogits h^T, db2 = dlogits, dh = W2^T dlogits
    h = relu(W1 x + b1) ->  da = dh * [h > 0],  dW1 = da x^T, db1 = da
    """
    for k in range(NUM_CLASSES):
        grads["b2"][0][k] += dlogits[k]
        for j, hj in enumerate(hidden):
            grads["w2"][k][j] += dlogits[k] * hj
    for j, hj in enumerate(hidden):
        if hj <= 0.0:
            continue  # ReLU was off: no gradient flows to this unit's input weights
        dpre = sum(dlogits[k] * model["w2"][k][j] for k in range(NUM_CLASSES))
        grads["b1"][0][j] += dpre
        grads["w1"][j][0] += dpre * x[0]
        grads["w1"][j][1] += dpre * x[1]


def zeros_like(model: Model) -> Model:
    return {key: [[0.0] * len(row) for row in matrix] for key, matrix in model.items()}


def sgd_step(model: Model, grads: Model) -> None:
    """In place: w <- w - LEARNING_RATE * dL/dw for every entry."""
    for key, matrix in model.items():
        for row, grad_row in zip(matrix, grads[key]):
            for j, g in enumerate(grad_row):
                row[j] -= LEARNING_RATE * g


# === OBJECTIVES ===


def hard_label_ce(
    model: Model, data: list[tuple[float, float, int]], grads: Model | None
) -> float:
    """Mean cross-entropy -log q1(y) at temperature 1; if grads is given, add its gradient.

    Per-example logit gradient: q1 - onehot(y). The batch mean contributes 1/B.
    """
    total = 0.0
    for x0, x1, label in data:
        hidden, logits = forward(model, (x0, x1))
        log_probs = log_softmax(logits, 1.0)
        total -= log_probs[label]
        if grads is not None:
            dlogits = [
                (math.exp(lp) - (1.0 if k == label else 0.0)) / len(data)
                for k, lp in enumerate(log_probs)
            ]
            accumulate_mlp_gradient(model, grads, (x0, x1), hidden, dlogits)
    return total / len(data)


def teacher_soft_targets(
    teacher: Model, data: list[tuple[float, float, int]], temperature: float
) -> list[list[float]]:
    """p_T = softmax(v / T) from the trained, frozen teacher's logits v for every point.

    Computed from the teacher that was actually trained above; nothing is hard-coded.
    """
    targets = []
    for x0, x1, _ in data:
        probs = softmax(forward(teacher, (x0, x1))[1], temperature)
        if abs(sum(probs) - 1.0) > 1e-9 or min(probs) < 0.0:
            raise ValueError(f"teacher target is not a distribution: {probs}")
        targets.append(probs)
    return targets


def distillation_objective(
    student: Model,
    data: list[tuple[float, float, int]],
    soft_targets: list[list[float]],
    grads: Model | None,
) -> tuple[float, float]:
    """Return (mean mixed objective, mean soft-target KL); optionally add its gradient.

    For one point with teacher target p_T, student logits z, q_T = softmax(z/T):
        objective = ALPHA * T^2 * KL(p_T || q_T) + (1 - ALPHA) * CE(y, q_1)
    KL(p_T || q_T) = sum_i p_i (log p_i - log q_i) equals Hinton's soft-target
    cross-entropy minus the teacher's entropy, which does not depend on the student, so
    both have the same student gradient. With Eq. 2, d KL / dz_i = (q_i - p_i) / T, hence
        d objective / dz = ALPHA * T * (q_T - p_T) + (1 - ALPHA) * (q_1 - onehot(y))
    The T^2 factor cancels the 1/T^2 shrinkage of soft-target gradients, so changing T
    does not silently change the soft/hard balance. Batch mean: each point adds 1/B.
    """
    t = TEMPERATURE
    total_objective = 0.0
    total_kl = 0.0
    for (x0, x1, label), target in zip(data, soft_targets):
        hidden, logits = forward(student, (x0, x1))
        log_q_t = log_softmax(logits, t)
        log_q_1 = log_softmax(logits, 1.0)
        kl = sum(p * (math.log(p) - lq) for p, lq in zip(target, log_q_t) if p > 0.0)
        ce = -log_q_1[label]
        total_kl += kl
        total_objective += ALPHA * t * t * kl + (1 - ALPHA) * ce
        if grads is not None:
            dlogits = [
                (
                    ALPHA * t * (math.exp(lq_t) - p)
                    + (1 - ALPHA) * (math.exp(lq_1) - (1.0 if k == label else 0.0))
                )
                / len(data)
                for k, (lq_t, lq_1, p) in enumerate(zip(log_q_t, log_q_1, target))
            ]
            accumulate_mlp_gradient(student, grads, (x0, x1), hidden, dlogits)
    return total_objective / len(data), total_kl / len(data)


# === INFERENCE ===


def predict(model: Model, x: tuple[float, float]) -> int:
    """Class with the highest probability at temperature 1 (deployment temperature)."""
    probs = softmax(forward(model, x)[1], 1.0)
    return max(range(NUM_CLASSES), key=lambda k: probs[k])


def accuracy(model: Model, data: list[tuple[float, float, int]]) -> float:
    return sum(predict(model, (x0, x1)) == y for x0, x1, y in data) / len(data)


def agreement(a: Model, b: Model, data: list[tuple[float, float, int]]) -> float:
    """Fraction of points where two models predict the same class at temperature 1."""
    return sum(
        predict(a, (x0, x1)) == predict(b, (x0, x1)) for x0, x1, _ in data
    ) / len(data)


# === TRAINING AND INFERENCE ===

if __name__ == "__main__":
    # Both partitions are drawn before any model exists, so no model can shape them.
    train = make_clusters(TRAIN_PER_CLASS)
    heldout = make_clusters(HELDOUT_PER_CLASS)
    print(
        f"Data: {len(train)} training and {len(heldout)} held-out points, "
        f"{NUM_CLASSES} Gaussian clusters (std {CLUSTER_STD})"
    )

    # === Stage 1: train the teacher on hard labels ===
    teacher = init_mlp(TEACHER_HIDDEN)
    teacher_params = count_params(teacher)
    if teacher_params != EXPECTED_TEACHER_PARAMS:
        raise RuntimeError(f"teacher has {teacher_params} parameters, expected 99")
    print(
        f"\n=== Stage 1: teacher 2 -> {TEACHER_HIDDEN} -> 3 ({teacher_params} params) ==="
    )
    teacher_init_ce = hard_label_ce(teacher, train, None)
    for epoch in range(TEACHER_EPOCHS):
        grads = zeros_like(teacher)
        ce = hard_label_ce(teacher, train, grads)
        sgd_step(teacher, grads)
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{TEACHER_EPOCHS} | hard-label CE {ce:.4f}")
    teacher_ce = hard_label_ce(teacher, train, None)
    teacher_acc = accuracy(teacher, train)
    print(f"Teacher training CE: {teacher_init_ce:.4f} (init) -> {teacher_ce:.4f}")
    print(
        f"Teacher accuracy: train {teacher_acc:.3f} | held-out {accuracy(teacher, heldout):.3f}"
    )

    # === Freeze the teacher and compute its soft targets once ===
    # From here on the teacher is only read. Its digest is checked again after transfer.
    frozen_digest = weight_digest(teacher)
    soft_targets = teacher_soft_targets(teacher, train, TEMPERATURE)
    # Show what a soft target contains that a label does not: the point whose teacher
    # distribution at T=1 is least confident, at T=1 and at T=TEMPERATURE.
    probe = min(
        range(len(train)),
        key=lambda i: max(
            softmax(forward(teacher, (train[i][0], train[i][1]))[1], 1.0)
        ),
    )
    p1 = softmax(forward(teacher, (train[probe][0], train[probe][1]))[1], 1.0)
    print(
        f"Least confident training point (label {train[probe][2]}): teacher p at T=1 "
        f"{[round(p, 3) for p in p1]} -> at T={TEMPERATURE:g} "
        f"{[round(p, 3) for p in soft_targets[probe]]}"
    )

    # === Stage 2: train a smaller student on soft targets + hard labels ===
    student = init_mlp(STUDENT_HIDDEN)
    student_params = count_params(student)
    if student_params != EXPECTED_STUDENT_PARAMS:
        raise RuntimeError(f"student has {student_params} parameters, expected 27")
    print(
        f"\n=== Stage 2: student 2 -> {STUDENT_HIDDEN} -> 3 ({student_params} params), "
        f"T={TEMPERATURE:g}, alpha={ALPHA:g} ==="
    )
    student_init_digest = weight_digest(student)
    init_objective, init_kl = distillation_objective(student, train, soft_targets, None)
    init_agreement = agreement(student, teacher, train)
    for epoch in range(STUDENT_EPOCHS):
        grads = zeros_like(student)
        objective, kl = distillation_objective(student, train, soft_targets, grads)
        sgd_step(student, grads)
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:>3}/{STUDENT_EPOCHS} | mixed objective {objective:.4f}"
                f" | soft-target KL {kl:.4f}"
            )
    final_objective, final_kl = distillation_objective(
        student, train, soft_targets, None
    )
    student_changed = weight_digest(student) != student_init_digest
    teacher_unchanged = weight_digest(teacher) == frozen_digest

    # === Results (temperature-1 inference for both models) ===
    print("\n=== Results (inference at T=1) ===")
    print(f"Parameters: teacher {teacher_params}, student {student_params}")
    print(
        f"Mixed training objective: {init_objective:.4f} (init) -> {final_objective:.4f}"
    )
    print(f"Soft-target KL(p_T || q_T): {init_kl:.4f} (init) -> {final_kl:.4f}")
    print(
        f"Student-teacher agreement on training points: {init_agreement:.3f} (init) -> "
        f"{agreement(student, teacher, train):.3f}"
    )
    print(
        f"Hard-label accuracy, train: teacher {teacher_acc:.3f} | "
        f"student {accuracy(student, train):.3f}"
    )
    # Report-only: the held-out points never influenced training, stopping or settings.
    # Capacity differs between the two models, so these numbers do not show that
    # distillation itself caused any difference; no hard-label-only student is trained.
    print(
        f"Held-out (report-only): teacher accuracy {accuracy(teacher, heldout):.3f} | "
        f"student accuracy {accuracy(student, heldout):.3f} | "
        f"agreement {agreement(student, teacher, heldout):.3f}"
    )
    print(f"Teacher weight digest unchanged through transfer: {teacher_unchanged}")

    # === Acceptance (frozen training-side criteria) ===
    checks = [
        (
            f"teacher CE falls by >= {MIN_RELATIVE_DECREASE:.0%}",
            teacher_ce <= (1 - MIN_RELATIVE_DECREASE) * teacher_init_ce,
        ),
        (
            f"teacher training accuracy >= {MIN_TEACHER_ACCURACY:.0%}",
            teacher_acc >= MIN_TEACHER_ACCURACY,
        ),
        (
            f"student mixed objective falls by >= {MIN_RELATIVE_DECREASE:.0%}",
            final_objective <= (1 - MIN_RELATIVE_DECREASE) * init_objective,
        ),
        (
            f"student soft-target KL falls by >= {MIN_RELATIVE_DECREASE:.0%}",
            final_kl <= (1 - MIN_RELATIVE_DECREASE) * init_kl,
        ),
        ("student weights changed", student_changed),
        ("teacher weights byte-identical through transfer", teacher_unchanged),
    ]
    print("\nAcceptance (training data only):")
    for description, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")
    failed = [description for description, passed in checks if not passed]
    if failed:
        raise RuntimeError(f"acceptance failed: {'; '.join(failed)}")

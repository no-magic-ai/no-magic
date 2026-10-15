"""
Eval error bars checked against a known truth: coverage, paired comparisons and questions needed.

An eval score is a sample mean, so it needs an error bar: simulate evals with a known truth and
count how often CLT, bootstrap and Wilson 95% intervals cover it, how much pairing two trained
models tightens their comparison, and how many questions an effect needs.

Part A measures coverage of three 95% intervals over a (mean, n) grid. Part B trains two logistic
regressions (A sees 8 features, B sees 7), scores them on a 20,000-item pool and compares the
unpaired and paired standard errors of A - B. Part C turns the paired variance into a minimum
detectable effect, a required sample size and a simulated power check.
"""
# Reference: Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model
# Evaluations", 2024, https://arxiv.org/abs/2411.00640 -- Eq. 1-3 (Sec. 2.1, the CLT standard
# error), Sec. 3.1 (the uniform-difficulty example), Eq. 5 and Eq. 7 (Sec. 4, unpaired and paired
# differences), Eq. 9 and Eq. 10 (Sec. 5, required sample size and minimum detectable effect).
# Wilson: Wilson, JASA 1927; Brown, Cai & DasGupta, Statistical Science 2001 (abstract: "we
# recommend the Wilson interval or the equal-tailed Jeffreys prior interval for small n and the
# interval suggested in Agresti and Coull for larger n").
# Related helpers: wilson() in 03-systems/microcot.py and 03-systems/microwatermark.py (this file
# re-implements it with one shared z instead of importing, since scripts stay standalone).

# === TRADEOFFS ===
# + The CLT error bar is one line (mean +/- z * sqrt(var / n)) and needs no resampling
# + Pairing both models on the SAME questions removes shared difficulty from the difference:
#   the paired interval is narrower whenever the two models' scores are positively
#   correlated (Miller Eq. 7)
# + The same variance gives the question count needed to see an effect (Eq. 9, Eq. 10)
# - Textbook intervals assume many independent questions; the CLT interval has exactly zero
#   width when every score is equal, so check the zero_w and cov columns in Part A at small n
# - Questions that share a passage or a template are NOT independent; this script ignores
#   that (Miller Sec. 2.2 clusters standard errors; out of scope, see microwatermark.py)
# - Power numbers are only as good as the variance estimate they are computed from
# WHEN TO USE: reporting any eval score or model comparison as a number with a stated
#   uncertainty, and deciding before a run how many questions are enough.
# WHEN NOT TO: evals whose questions are clustered or adaptively chosen without first modelling
#   that structure, or when the questions are the entire population you care about (no
#   sampling, so no sampling error to report).

# === SIGNPOSTS (read before trusting a number) ===
# - This is a toy simulation, not a measurement of any real eval. Part A draws Bernoulli scores
#   from a known mean; Part B's "models" are 8-weight logistic regressions on synthetic data.
# - Production eval frameworks often bootstrap the mean. Miller Sec. 2.1 cites OpenAI evals and
#   calls bootstrapping unnecessary unless a complicated sampling scheme or estimator is used;
#   it is run here so you can compare it with the CLT and Wilson on a plain mean; read Part A
#   for where they agree and where they do not.
# - Miller Sec. 2.1 applies the CLT to evals with "a large number of questions". Compare the
#   n = 30 rows with the n = 500 rows to see what "large" needs to mean for each method.
# - Part A's difficulty layer x_i mirrors Miller Sec. 3.1 notation. With K = 1 answer per
#   question the marginal score is i.i.d. Bernoulli(mu), so difficulty does not change coverage.
# - Gates check correctness of the code (arithmetic, training, row counts), never effect sizes.

# === HOW TO READ THE OUTPUT ===
# - Part A rows: coverage of the true mean (target 0.950), its Monte Carlo SE, the mean interval
#   width, and the share of evals whose interval had exactly zero width.
# - Part B rows: coverage of the pool difference by unpaired (Eq. 5) and paired (Eq. 7)
#   intervals, mean widths, and the paired/unpaired width ratio.
# - Part C rows: Eq. 10 minimum detectable effect, Eq. 9 required n, and a power simulation.
# - Lines starting "[timing]" are the only non-deterministic output.

import math
import random
import time

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

# Statistics -----------------------------------------------------------------------------------
CONFIDENCE_LEVEL = 0.95  # two-sided; every interval below uses the one z derived from this
POWER_TARGET = 0.80  # 1 - beta, Part C
ALPHA = 0.05  # Part C significance level (same as 1 - CONFIDENCE_LEVEL)
BETA = 0.20  # Part C type-II error rate
BISECTION_TOLERANCE = 1e-12  # accuracy of the hand-written inverse normal

# Part A: coverage grid ------------------------------------------------------------------------
TRUE_MEANS = (0.50, 0.80, 0.95)  # mu; mu = 0.5 is exactly Miller's x ~ U[0, 1]
SAMPLE_SIZES = (30, 100, 500)  # n questions per simulated eval
EVALS_PER_CELL = 1000  # R simulated evals per (mu, n) cell
BOOTSTRAP_RESAMPLES = 1000  # B resamples per eval for the percentile bootstrap
METHOD_NAMES = ("CLT", "bootstrap", "Wilson")

# Part B: two classifiers on a synthetic super-population ---------------------------------------
FEATURE_COUNT = 8  # d
ARM_A_FEATURES = 8  # arm A sees every feature
ARM_B_FEATURES = 7  # arm B sees the first seven only
TRAIN_SIZE = 2000
POOL_SIZE = 20000  # stand-in super-population the evals are drawn from
LEARNING_RATE = 0.05
EPOCHS = 10
ACCURACY_TOLERANCE = 0.02  # gate G1: pool accuracy vs analytic Bayes accuracy
EVAL_SIZES = (30, 100, 500)  # n for the paired-vs-unpaired experiment

# Part C: power --------------------------------------------------------------------------------
MDE_SAMPLE_SIZES = (30, 100, 500, 1000)
TARGET_DELTAS = (0.01, 0.02, 0.05)
MILLER_VARIANCE = 1.0 / 9.0  # Miller's worked example omega^2
MILLER_DELTA = 0.03
MIN_CHECK_SIZE = 30  # smallest n used in the power simulation
POWER_RUNS = 1000

# Expected printed rows for gate G3
PART_A_ROWS = len(TRUE_MEANS) * len(SAMPLE_SIZES) * len(METHOD_NAMES)  # 27
PART_B_ROWS = len(EVAL_SIZES)  # 3
PART_C_ROWS = 2 * len(MDE_SAMPLE_SIZES) + 2 * len(TARGET_DELTAS) + 1 + 1  # 16

# === BASIC STATISTICS (hand-written, no statistics module) ===


def mean(values):
    """Arithmetic mean: (1/n) * sum(values)."""
    return sum(values) / len(values)


def sample_variance(values):
    """Unbiased variance with n - 1 in the denominator (Bessel's correction).

    Why n - 1: the mean was estimated from the same data, which uses up one degree of
    freedom; dividing by n would make the variance (and every error bar) too small.
    """
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def population_variance(values):
    """Variance with n in the denominator: the pool IS the population, not a sample of it."""
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def population_covariance(first, second):
    """Covariance with n in the denominator, for the same reason as population_variance."""
    center_first = mean(first)
    center_second = mean(second)
    total = sum((a - center_first) * (b - center_second) for a, b in zip(first, second))
    return total / len(first)


def normal_cdf(x):
    """Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inverse_normal(probability):
    """Find x with Phi(x) = probability by bisection (Phi is strictly increasing).

    Bisection is slow but needs no derivation and cannot diverge: each step halves a bracket
    that provably contains the answer. About 45 halvings of [-10, 10] reach 1e-12.
    """
    low = -10.0
    high = 10.0
    while high - low > BISECTION_TOLERANCE:
        middle = 0.5 * (low + high)
        if normal_cdf(middle) < probability:
            low = middle  # Phi(middle) too small -> the answer lies to the right
        else:
            high = middle
    return 0.5 * (low + high)


# One z for every method: z_{alpha/2} = Phi^-1(0.975) ~ 1.959964. Using a single value means
# the CLT, bootstrap-comparison, Wilson, paired and unpaired intervals differ only in their
# standard error or shape, never in the multiplier. z_beta = Phi^-1(0.80) is for power.
Z_CRITICAL = inverse_normal(1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0)
Z_BETA = inverse_normal(POWER_TARGET)

# === INTERVAL METHODS ===


def clt_interval(scores):
    """Miller Eq. 1 + Eq. 3: mean +/- z * SE with SE = sqrt(var_{n-1} / n).

    Eq. 1: SE_CLT = sqrt(Var(s) / n) with the n - 1 sample variance. Eq. 3: the 95% interval is
    the mean +/- z * SE (Miller writes 1.96; this file uses the shared bisection z). Eq. 2 is
    the Bernoulli special case SE = sqrt(p_hat (1 - p_hat) / n); its SE differs from Eq. 1 by
    the factor sqrt(n / (n - 1)). The interval is symmetric and can leave [0, 1].
    """
    n = len(scores)
    center = mean(scores)
    half_width = Z_CRITICAL * math.sqrt(sample_variance(scores) / n)
    return center - half_width, center + half_width


def bootstrap_percentile_interval(scores):
    """Percentile bootstrap of the mean: resample n scores with replacement, B times.

    Each resample is drawn explicitly with random.choices(scores, k=n). The interval is the
    sorted resample means at index floor(0.025 * B) and ceil(0.975 * B) - 1. Integer
    arithmetic gives those same indices without a float rounding risk (0.975 * 1000 is not
    exactly 975 in binary floating point). For B = 1000 they are positions 25 and 974.
    """
    n = len(scores)
    resample_means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = random.choices(scores, k=n)
        resample_means.append(sum(resample) / n)
    resample_means.sort()
    lower_index = (BOOTSTRAP_RESAMPLES * 25) // 1000  # floor(0.025 * B)
    upper_index = -((-BOOTSTRAP_RESAMPLES * 975) // 1000) - 1  # ceil(0.975 * B) - 1
    return resample_means[lower_index], resample_means[upper_index]


def wilson_interval(successes, n):
    """Wilson score interval for a binomial proportion, with the shared z.

    With p = k / n:  center = (p + z^2 / 2n) / (1 + z^2 / n)
                     half   = z * sqrt(p(1-p)/n + z^2 / 4n^2) / (1 + z^2 / n)
    It inverts the score test instead of plugging p_hat into the SE, so it never has zero width
    and never leaves [0, 1]. Brown, Cai & DasGupta (2001) abstract: "we recommend the Wilson
    interval or the equal-tailed Jeffreys prior interval for small n and the interval suggested
    in Agresti and Coull for larger n". Same formula as wilson() in microcot.py, z is shared.
    """
    p_hat = successes / n
    z_squared = Z_CRITICAL * Z_CRITICAL
    shrink = 1.0 + z_squared / n
    center = (p_hat + z_squared / (2.0 * n)) / shrink
    half_width = Z_CRITICAL * math.sqrt(p_hat * (1.0 - p_hat) / n + z_squared / (4.0 * n * n))
    half_width /= shrink
    return center - half_width, center + half_width


# === DATA GENERATION: PART A DIFFICULTY MODEL ===


def draw_scores(true_mean, n):
    """Draw n scores from D(mu): x_i ~ Uniform[2mu - 1, 1], then s_i ~ Bernoulli(x_i), K = 1.

    The question difficulty x_i has mean (2mu - 1 + 1) / 2 = mu, so P(s_i = 1) = mu
    marginally. mu = 0.5 gives x ~ U[0, 1], exactly Miller Sec. 3.1. Because K = 1 (one answer
    per question), the x_i layer is invisible in the scores: they are plain i.i.d. Bernoulli(mu).
    """
    scores = []
    for _ in range(n):
        question_difficulty = random.uniform(2.0 * true_mean - 1.0, 1.0)
        scores.append(1 if random.random() < question_difficulty else 0)
    return scores


# === DATA GENERATION: PART B SYNTHETIC CLASSIFICATION ===


def make_items(count):
    """Features x_j ~ N(0, 1) for d = 8; label y = 1 if sum_j x_j + eps > 0, eps ~ N(0, 1).

    The label noise eps is never shown to a model, so even a perfect weight vector errs. That
    leaves a known ceiling (the Bayes accuracy) to check the trained models against (gate G1).
    """
    features = []
    labels = []
    for _ in range(count):
        row = [random.gauss(0.0, 1.0) for _ in range(FEATURE_COUNT)]
        noise = random.gauss(0.0, 1.0)
        labels.append(1 if sum(row) + noise > 0.0 else 0)
        features.append(row)
    return features, labels


def bayes_accuracy(used_features):
    """Best possible accuracy using the first `used_features` of d features.

    Label = sign(S_8 + eps) has variance 9; a predictor sign(S_k) is correlated with it by
    rho = sqrt(k / 9). For jointly Gaussian signs, P(agree) = 1/2 + asin(rho) / pi.
    """
    return 0.5 + math.asin(math.sqrt(used_features / (FEATURE_COUNT + 1.0))) / math.pi


# === MODEL DEFINITION: LOGISTIC REGRESSION ===


def sigmoid(value):
    """Numerically stable logistic function 1 / (1 + e^-value); never exponentiates a big +."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)  # value < 0, so exponent is in (0, 1): safe
    return exponent / (1.0 + exponent)


def predict_probability(weights, bias, row, used_features):
    """p(y = 1 | x) = sigmoid(w . x + b), using only the first `used_features` features."""
    total = bias
    for index in range(used_features):
        total += weights[index] * row[index]
    return sigmoid(total)


def mean_log_loss(weights, bias, features, labels, used_features):
    """Mean of -[y ln p + (1 - y) ln(1 - p)]; p is clipped so ln never sees exactly 0."""
    total = 0.0
    for row, label in zip(features, labels):
        probability = predict_probability(weights, bias, row, used_features)
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= math.log(probability if label == 1 else 1.0 - probability)
    return total / len(labels)


# === TRAINING LOOP ===


def train_logistic_regression(features, labels, used_features):
    """Plain per-example SGD on log-loss from zero initialization.

    For log-loss the gradient with respect to the pre-sigmoid score is simply (p - y), so
    dL/dw_j = (p - y) * x_j and dL/db = (p - y). Each epoch visits the training items once in a
    freshly shuffled order (one random.shuffle of an index list per epoch).
    """
    weights = [0.0] * used_features
    bias = 0.0
    order = list(range(len(labels)))
    for _ in range(EPOCHS):
        random.shuffle(order)
        for item_index in order:
            row = features[item_index]
            error = predict_probability(weights, bias, row, used_features) - labels[item_index]
            for feature_index in range(used_features):
                weights[feature_index] -= LEARNING_RATE * error * row[feature_index]
            bias -= LEARNING_RATE * error
    return weights, bias


# === INFERENCE: SCORING ON THE POOL ===


def score_items(weights, bias, features, labels, used_features):
    """s_{M,i} = 1 if the thresholded prediction (p >= 0.5) is correct, else 0.

    The score is a deterministic function of the model and the question, so the within-
    question variance sigma^2 in Miller's decomposition is 0 here; all variance is between
    questions. (A sampled answer would have sigma^2 > 0; we do not sample answers.)
    """
    scores = []
    for row, label in zip(features, labels):
        probability = predict_probability(weights, bias, row, used_features)
        scores.append(1 if (1 if probability >= 0.5 else 0) == label else 0)
    return scores


# === PART A: COVERAGE OF THREE 95% INTERVALS ===


def run_part_a():
    """Simulate R evals per (mu, n) cell; print coverage, MC SE, width and zero-width share.

    Coverage means lo <= mu <= hi. Its Monte Carlo SE is sqrt(c(1 - c) / R): the uncertainty
    of the coverage estimate itself, so differences below ~2 SE are noise. Zero-width means
    hi - lo == 0.0 exactly (all scores identical, so the CLT variance is 0). Every cell prints.
    """
    print(f"=== PART A: coverage of 95% intervals for a mean (R = {EVALS_PER_CELL} evals per "
          "cell) ===")
    print("# toy simulation, not a measurement of a real eval; target coverage 0.950")
    rows_printed = 0
    for true_mean in TRUE_MEANS:
        for n in SAMPLE_SIZES:
            hits = [0, 0, 0]
            width_totals = [0.0, 0.0, 0.0]
            zero_widths = [0, 0, 0]
            for _ in range(EVALS_PER_CELL):
                scores = draw_scores(true_mean, n)
                intervals = (
                    clt_interval(scores),
                    bootstrap_percentile_interval(scores),
                    wilson_interval(sum(scores), n),
                )
                for method_index, (low, high) in enumerate(intervals):
                    if low <= true_mean <= high:
                        hits[method_index] += 1
                    width_totals[method_index] += high - low
                    if high - low == 0.0:
                        zero_widths[method_index] += 1
            for method_index, method_name in enumerate(METHOD_NAMES):
                coverage = hits[method_index] / EVALS_PER_CELL
                monte_carlo_se = math.sqrt(coverage * (1.0 - coverage) / EVALS_PER_CELL)
                mean_width = width_totals[method_index] / EVALS_PER_CELL
                zero_share = zero_widths[method_index] / EVALS_PER_CELL
                print(f"[A] mu={true_mean:.2f} n={n:>3} {method_name:<9} "
                      f"cov={coverage:.3f} mcse={monte_carlo_se:.3f} "
                      f"width={mean_width:.4f} zero_w={zero_share:.3f}")
                rows_printed += 1
    return rows_printed


# === PART B: TWO MODELS, ONE POOL, PAIRED VS UNPAIRED ===


def pool_statistics(scores_a, scores_b):
    """Population moments of the pool: the 'true' values the sampled evals try to estimate."""
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    variance_a = population_variance(scores_a)
    variance_b = population_variance(scores_b)
    covariance = population_covariance(scores_a, scores_b)
    return {
        "difference": mean(scores_a) - mean(scores_b),
        "var_a": variance_a,
        "var_b": variance_b,
        "cov": covariance,
        "corr": covariance / math.sqrt(variance_a * variance_b),
        "var_diff": population_variance(differences),
    }


def difference_intervals(sample_a, sample_b):
    """Unpaired (Eq. 5) and paired (Eq. 7) intervals for mean(A) - mean(B) on one eval set.

    Eq. 1 gives each model's SE_A, SE_B. Eq. 5 (unpaired): SE = sqrt(SE_A^2 + SE_B^2), which
    treats the two score lists as independent. Eq. 7 (paired): SE = sqrt(var_{n-1}(s_A - s_B)
    / n), the SE of the per-question difference. The unpaired and paired variances (Miller
    Sec. 4.2) differ by the -2 Cov term: paired = Var(A)+Var(B)-2Cov(A, B), smaller if Cov > 0.
    """
    n = len(sample_a)
    estimate = mean(sample_a) - mean(sample_b)
    unpaired_se = math.sqrt(sample_variance(sample_a) / n + sample_variance(sample_b) / n)
    paired_se = math.sqrt(sample_variance([a - b for a, b in zip(sample_a, sample_b)]) / n)
    return (
        (estimate - Z_CRITICAL * unpaired_se, estimate + Z_CRITICAL * unpaired_se),
        (estimate - Z_CRITICAL * paired_se, estimate + Z_CRITICAL * paired_se),
    )


def run_part_b():
    """Generate data, train both arms, score the pool, then compare interval widths.

    RNG order is fixed: 2,000 training items, the 20,000-item pool, arm A for 10 epochs, then
    arm B for 10 epochs; only then are eval sets drawn. Returns the two pool score lists, the
    G1 verdict and the number of rows printed.
    """
    print()
    print("=== PART B: is model A (8 features) better than model B (7 features)? ===")
    train_features, train_labels = make_items(TRAIN_SIZE)
    pool_features, pool_labels = make_items(POOL_SIZE)

    arms = (("A", ARM_A_FEATURES), ("B", ARM_B_FEATURES))
    pool_scores = []
    arm_ok = []
    for arm_name, used_features in arms:
        weights, bias = train_logistic_regression(train_features, train_labels, used_features)
        initial_loss = mean_log_loss([0.0] * used_features, 0.0, train_features,
                                     train_labels, used_features)
        final_loss = mean_log_loss(weights, bias, train_features, train_labels, used_features)
        scores = score_items(weights, bias, pool_features, pool_labels, used_features)
        accuracy = mean(scores)
        ceiling = bayes_accuracy(used_features)
        print(f"[B] arm {arm_name} ({used_features} features): init_loss={initial_loss:.4f} "
              f"final_loss={final_loss:.4f} pool_acc={accuracy:.4f} bayes_acc={ceiling:.4f}")
        # The zero-weight model predicts p = 0.5 everywhere, so its loss is ln 2 ~ 0.6931.
        arm_ok.append(abs(accuracy - ceiling) < ACCURACY_TOLERANCE
                      and final_loss < math.log(2.0))
        pool_scores.append(scores)

    scores_a, scores_b = pool_scores
    stats = pool_statistics(scores_a, scores_b)
    print(f"[B] pool difference A-B = {stats['difference']:.4f}")
    print(f"[B] pool Var(s_A)={stats['var_a']:.4f} Var(s_B)={stats['var_b']:.4f} "
          f"Cov={stats['cov']:.4f} Corr(s_A, s_B)={stats['corr']:.4f}")
    print("Is A better than B, and how sure are we?")
    print(f"# intervals below are diff +/- z*SE (z = {Z_CRITICAL:.6f}); coverage target is the "
          "pool difference")
    print("# eval sets drawn WITHOUT replacement from the pool; no finite-population correction")

    rows_printed = 0
    target = stats["difference"]
    pool_indices = range(POOL_SIZE)
    for n in EVAL_SIZES:
        hits = [0, 0]  # unpaired, paired
        width_totals = [0.0, 0.0]
        for _ in range(EVALS_PER_CELL):
            chosen = random.sample(pool_indices, n)
            sample_a = [scores_a[index] for index in chosen]
            sample_b = [scores_b[index] for index in chosen]
            for method_index, (low, high) in enumerate(difference_intervals(sample_a, sample_b)):
                if low <= target <= high:
                    hits[method_index] += 1
                width_totals[method_index] += high - low
        coverage = [count / EVALS_PER_CELL for count in hits]
        monte_carlo_se = [math.sqrt(c * (1.0 - c) / EVALS_PER_CELL) for c in coverage]
        widths = [total / EVALS_PER_CELL for total in width_totals]
        print(f"[B] n={n:>3} unpaired cov={coverage[0]:.3f}+/-{monte_carlo_se[0]:.3f} "
              f"w={widths[0]:.4f} | paired cov={coverage[1]:.3f}+/-{monte_carlo_se[1]:.3f} "
              f"w={widths[1]:.4f} | ratio={widths[1] / widths[0]:.3f}")
        rows_printed += 1
    return scores_a, scores_b, stats, all(arm_ok), rows_printed


# === PART C: HOW MANY QUESTIONS DO YOU NEED? ===


def minimum_detectable_effect(variance, n):
    """Miller Eq. 10: MDE = (z_{alpha/2} + z_beta) * sqrt(variance / n).

    The smallest true difference a test at level alpha detects with probability 1 - beta.
    """
    return (Z_CRITICAL + Z_BETA) * math.sqrt(variance / n)


def required_sample_size(variance, delta):
    """Miller Eq. 9, unrounded: n = (z_{alpha/2} + z_beta)^2 * variance / delta^2.

    Eq. 10 solved for n. Halving the effect you want to detect quadruples the questions needed.
    """
    return (Z_CRITICAL + Z_BETA) ** 2 * variance / (delta * delta)


def run_part_c(scores_a, scores_b, stats):
    """Print MDE and required n (paired and unpaired), Miller's example and a power check."""
    print()
    print("=== PART C: minimum detectable effect, required n and power "
          f"(alpha = {ALPHA}, beta = {BETA}) ===")
    paired_variance = stats["var_diff"]  # omega^2: Var(s_A - s_B) over the pool
    # Miller's Eq. 9 assumes a paired analysis. The unpaired row replaces omega^2 with
    # Var(s_A) + Var(s_B): an extension of Eq. 10, labelled as such, not Miller's formula.
    unpaired_variance = stats["var_a"] + stats["var_b"]
    variances = (("paired", paired_variance), ("unpaired", unpaired_variance))
    print(f"# omega^2 (paired) = pool Var(s_A - s_B) = {paired_variance:.4f}; "
          f"unpaired variance = Var(s_A)+Var(s_B) = {unpaired_variance:.4f}")
    print("# unpaired rows are an extension of Eq. 10 (Miller's Eq. 9 assumes paired analysis)")

    rows_printed = 0
    for n in MDE_SAMPLE_SIZES:
        for label, variance in variances:
            mde = minimum_detectable_effect(variance, n)
            print(f"[C] MDE n={n:>4} {label:<8} var={variance:.4f} mde={mde:.4f}")
            rows_printed += 1
    for delta in TARGET_DELTAS:
        for label, variance in variances:
            needed = math.ceil(required_sample_size(variance, delta))
            print(f"[C] required n delta={delta:.2f} {label:<8} n={needed}")
            rows_printed += 1

    miller_n = required_sample_size(MILLER_VARIANCE, MILLER_DELTA)
    print(f"[C] Miller example omega^2=1/9 delta={MILLER_DELTA:.2f} n={miller_n:.3f} "
          "(Eq. 9, unrounded)")
    rows_printed += 1

    # Power check: draw eval sets of the size Eq. 9 asks for at delta = the pool difference,
    # run the paired z-test (reject if |diff| / SE_paired > z) and compare the rejection rate
    # with the predicted power Phi(|delta| / sqrt(omega^2 / n) - z). The formula ignores the
    # far rejection tail, which is negligible at these effect sizes.
    delta = abs(stats["difference"])
    n_check = max(MIN_CHECK_SIZE, math.ceil(required_sample_size(paired_variance, delta)))
    n_check = min(n_check, POOL_SIZE)  # cannot draw more items than the pool holds
    rejections = 0
    pool_indices = range(POOL_SIZE)
    for _ in range(POWER_RUNS):
        chosen = random.sample(pool_indices, n_check)
        differences = [scores_a[index] - scores_b[index] for index in chosen]
        standard_error = math.sqrt(sample_variance(differences) / n_check)  # Eq. 7
        estimate = mean(differences)
        if standard_error == 0.0:
            rejected = estimate != 0.0  # no spread at all: any non-zero gap is "certain"
        else:
            rejected = abs(estimate) / standard_error > Z_CRITICAL
        if rejected:
            rejections += 1
    simulated = rejections / POWER_RUNS
    simulated_se = math.sqrt(simulated * (1.0 - simulated) / POWER_RUNS)
    predicted = normal_cdf(delta / math.sqrt(paired_variance / n_check) - Z_CRITICAL)
    print(f"[C] power check n={n_check} delta={delta:.4f} simulated={simulated:.3f}"
          f"+/-{simulated_se:.3f} predicted={predicted:.3f}")
    rows_printed += 1
    return rows_printed


# === GATES (correctness only; no effect size is ever tested) ===


def verdict(condition):
    """Turn a boolean into the word the gate line ends with."""
    return "PASS" if condition else "FAIL"


def wilson_edge_cases_ok():
    """Wilson bounds stay in [0, 1] up to rounding and contain p_hat, for k = 0 and k = n."""
    n = 30
    tolerance = 1e-12
    for successes in (0, n):
        low, high = wilson_interval(successes, n)
        p_hat = successes / n
        if low < -tolerance or high > 1.0 + tolerance:
            return False
        if not (low - tolerance <= p_hat <= high + tolerance):
            return False
    return True


def run_gates(arms_ok, stats, rows_a, rows_b, rows_c):
    """Print G1-G3, one line each, ending in PASS or FAIL."""
    print()
    print("=== GATES ===")
    print(f"G1 training: pool accuracy within {ACCURACY_TOLERANCE} of Bayes and final log-loss "
          f"< ln 2 for both arms -> {verdict(arms_ok)}")

    z_error = abs(Z_CRITICAL - 1.959964)
    miller_n = required_sample_size(MILLER_VARIANCE, MILLER_DELTA)
    identity_gap = abs(stats["var_diff"]
                       - (stats["var_a"] + stats["var_b"] - 2.0 * stats["cov"]))
    g2 = (z_error < 1e-6 and 968.0 < miller_n < 970.0 and wilson_edge_cases_ok()
          and identity_gap < 1e-9)
    print(f"G2 arithmetic: z={Z_CRITICAL:.6f}, Miller n={miller_n:.3f}, Wilson edge cases, "
          f"Var(A-B) identity gap={identity_gap:.1e} -> {verdict(g2)}")

    g3 = (rows_a == PART_A_ROWS and rows_b == PART_B_ROWS and rows_c == PART_C_ROWS)
    print(f"G3 completeness: rows printed A={rows_a}/{PART_A_ROWS} B={rows_b}/{PART_B_ROWS} "
          f"C={rows_c}/{PART_C_ROWS} -> {verdict(g3)}")


# === MAIN ===


def main():
    print(f"z = Phi^-1(0.975) = {Z_CRITICAL:.6f} (shared by every interval); "
          f"z_beta = Phi^-1(0.80) = {Z_BETA:.6f}")

    start = time.perf_counter()
    rows_a = run_part_a()
    print(f"[timing] part A {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    scores_a, scores_b, stats, arms_ok, rows_b = run_part_b()
    print(f"[timing] part B {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    rows_c = run_part_c(scores_a, scores_b, stats)
    print(f"[timing] part C {time.perf_counter() - start:.1f}s")

    run_gates(arms_ok, stats, rows_a, rows_b, rows_c)


if __name__ == "__main__":
    main()

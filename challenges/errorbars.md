# MicroErrorBars Challenges

Test your understanding of error bars for eval scores by predicting what happens in these scenarios. Try to work out the answer before revealing it. Numbers quoted in the answers come from the default seed-42 run of `01-foundations/microerrorbars.py` under CPython 3.12.8 on Apple M1 Pro. They describe a toy simulation with synthetic scores and one seed; none of them is a measurement of a real LLM eval.

---

### Challenge 1: The Interval That Collapses

**Setup:** `clt_interval` (lines 170-181) returns `mean ± z * sqrt(var / n)` with the unbiased sample variance (line 180). `wilson_interval` (lines 203-219) inverts the score test instead of plugging the sample proportion into the standard error (lines 213-218). `run_part_a` (lines 340-378) draws one set of scores per simulated eval (line 357) and builds the CLT, bootstrap and Wilson intervals from that same set (lines 358-362); it counts an interval as zero-width when `high - low == 0.0` (line 367).

**Question:** (a) An eval of n = 30 questions has every answer correct. What do `clt_interval` and `wilson_interval` return? (b) At a true mean of 0.95 and n = 30, the CLT and bootstrap rows print the same `zero_w`, and the Wilson row prints 0.000. What are the three coverages, and why do the first two zero-width shares agree?

<details>
<summary>Reveal Answer</summary>

**Answer:** (a) The CLT interval is the single point [1, 1]: all scores are equal, the sample variance is 0, and the half-width is 0. The Wilson interval has positive width: its upper bound is exactly 1 (up to rounding) and its lower bound is below 1. (b) For true mean 0.95 and n = 30 the pinned run prints `zero_w=0.201` for both CLT and bootstrap with coverage 0.797 (CLT), 0.789 (bootstrap) and 0.934 (Wilson, `zero_w=0.000`).

**Why:** A zero-variance eval gives the CLT formula no sign of uncertainty, so the 0.201 of evals at this cell with all 30 answers correct carry a zero-width error bar at the single point [1, 1], which cannot contain a true mean of 0.95. The percentile bootstrap resamples the 30 observed scores, so every resample of an all-equal eval is the same all-equal eval and its interval collapses to the same point; both methods collapse on exactly the same evals, which is why the shares agree. Wilson's formula adds z²/2n to the numerator and z²/4n² under the square root, so it keeps a width and stays inside [0, 1]. Edge cases at k = 0 and k = n are tested in `wilson_edge_cases_ok` (lines 569-580). By n = 100 the same true mean prints `zero_w=0.012` for the CLT and bootstrap, with coverage 0.863 (CLT), 0.938 (bootstrap) and 0.953 (Wilson). This is where the program's coverage study goes beyond the paper it cites: Miller Sec. 2.1 applies the CLT to evals with a large number of questions and does not claim under-coverage at small n; the Wilson reference follows Brown, Cai & DasGupta (2001).

**Script reference:** `01-foundations/microerrorbars.py`, lines 170-181 (`clt_interval`), lines 184-200 (`bootstrap_percentile_interval`), lines 203-219 (`wilson_interval`), lines 356-368 (the Part A loop), lines 569-580 (`wilson_edge_cases_ok`)

</details>

---

### Challenge 2: Is the Gap Real or Noise?

**Setup:** `run_part_a` prints each coverage with its Monte Carlo standard error `sqrt(c(1 - c) / R)` (lines 369-376), R = 1,000 simulated evals per cell (`EVALS_PER_CELL`, line 79). The target coverage is 0.950. At a true mean of 0.50 and n = 100 the rows read: CLT `cov=0.956`, bootstrap `cov=0.970`, Wilson `cov=0.956`.

**Question:** Does the bootstrap's coverage of 0.970 show that it is the better 95% interval at this cell? What does the bootstrap's mean width of 0.1937, against 0.1961 for the CLT interval, tell you?

<details>
<summary>Reveal Answer</summary>

**Answer:** Not from this row alone. The printed rows are bootstrap 0.970 (mcse 0.005) against CLT 0.956 (mcse 0.006) and Wilson 0.956 (mcse 0.006). All three lie above the 0.950 target, and the bootstrap lies furthest above it: a 95% interval that covers more often than 95% is conservative, not more accurate. The mean widths (0.1937 against 0.1961) are nearly equal, so the extra coverage is not bought with a visibly wider interval, but the program's own rule is that differences below about two standard errors are noise, so compare the printed gap with the printed mcse values and judge for yourself. At n = 500 the same cell reads 0.955 (CLT), 0.960 (bootstrap) and 0.955 (Wilson).

**Why:** Each coverage is the fraction of 1,000 simulated evals whose interval contained the truth, so it is itself an estimate with a standard error. The program's own docstring (lines 343-345) says differences below about two standard errors are noise. All three methods see the same evals (line 357), so their coverages are correlated, and one cell's gap is weak evidence for a ranking. A much larger contrast is the row for a true mean of 0.95 and n = 100: CLT 0.863 (mcse 0.011), bootstrap 0.938 (mcse 0.008) and Wilson 0.953 (mcse 0.007), where the CLT coverage is far below the 0.950 target by the program's own yardstick and Wilson is at it. One seed and one toy score model support neither claim about real evals.

**Script reference:** `01-foundations/microerrorbars.py`, line 79 (`EVALS_PER_CELL`), lines 343-345 (docstring on Monte Carlo SE), lines 369-376 (coverage, `mcse`, width and zero-width share per row)

</details>

---

### Challenge 3: Why Pairing Narrows the Bar

**Setup:** `pool_statistics` (lines 384-397) computes the population variances and covariance of the two arms' per-question scores over the 20,000-item pool. `difference_intervals` (lines 400-415) builds the unpaired standard error from the two sample variances (line 410) and the paired one from the sample variance of the per-question differences (line 411). Part B prints the pool moments (lines 451-452): `Var(s_A)=0.0951 Var(s_B)=0.1327 Cov=0.0509 Corr(s_A, s_B)=0.4530`.

**Question:** (a) What is the pool variance of `s_A - s_B`, and which standard error is larger? (b) At n = 30, 100 and 500 the unpaired intervals cover the pool difference 0.985, 0.990 and 0.990 of the time, and the paired intervals 0.919, 0.943 and 0.963. Which is closer to the nominal 0.95, and what does each coverage tell you?

<details>
<summary>Reveal Answer</summary>

**Answer:** (a) Var(A − B) = Var(A) + Var(B) − 2 Cov(A, B) = 0.0951 + 0.1327 − 2 × 0.0509 = 0.1260, the printed paired variance; the unpaired variance is Var(A) + Var(B) = 0.2278. The unpaired standard error is larger, and the paired intervals are narrower by a ratio of 0.738 at n = 30 and n = 100 and 0.744 at n = 500 (mean widths 0.2504 against 0.3391, 0.1374 against 0.1863 and 0.0622 against 0.0836). (b) The unpaired intervals over-cover at every n (0.985 to 0.990), too wide because they leave out the covariance. The paired intervals are nearer the nominal 0.95 at n = 100 (0.943) and n = 500 (0.963); at n = 30 they cover 0.919, below nominal, with Monte Carlo SE 0.009.

**Why:** The unpaired formula (Miller Eq. 5) treats the two score lists as independent, so it leaves out the −2 Cov term; the arms' scores correlate positively at 0.4530 because both models find the same questions easy or hard, so the unpaired bar is too wide and its coverage too high. The paired formula (Eq. 7) takes the variance of the per-question differences directly and removes that shared difficulty. The paired interval is not exact at small n; the program prints 0.919 at n = 30 and does not diagnose it; one plausible cause, not tested by the program, is a noisy sample variance at 30 questions. The eval sets are drawn without replacement from the pool with no finite-population correction (line 456), and the pool difference is the target, so these coverages are relative to that pool, not to a real population of questions.

**Script reference:** `01-foundations/microerrorbars.py`, lines 384-397 (`pool_statistics`), lines 400-415 (`difference_intervals`; unpaired line 410, paired line 411), lines 451-456 (printed pool moments and labels)

</details>

---

### Challenge 4: Halving the Effect

**Setup:** `minimum_detectable_effect` (lines 485-490) is Miller Eq. 10, `(z + z_beta) * sqrt(variance / n)`; `required_sample_size` (lines 493-498) is Eq. 9, the same equation solved for n. `run_part_c` (lines 501-558) evaluates both with the paired variance 0.1260 and, as an extension of Eq. 10, with the unpaired variance 0.2278 (lines 506-513). For a paired analysis a difference of 0.02 needs n = 2474.

**Question:** (a) Without running the program, estimate the paired required n for a difference of 0.01 and of 0.05. (b) Why does the unpaired row for 0.02 need n = 4471? (c) At the pool difference of 0.0511 the program draws eval sets of n = 379 and runs the paired z-test 1,000 times. What power does it predict, and what does the simulation find?

<details>
<summary>Reveal Answer</summary>

**Answer:** (a) Required n follows an inverse-square law in δ (Eq. 9): a smaller effect costs disproportionately more questions. The program prints paired n = 9893 for 0.01, against 2474 for 0.02, and n = 396 for 0.05. (b) With the unpaired variance the same formula gives n = 4471 for 0.02, in the ratio of the two variances (unpaired 0.2278 against paired 0.1260), because the paired design cancels the shared per-question difficulty. For 0.01 and 0.05 the unpaired rows print 17881 and 716. (c) The formula predicts power 0.801; the simulation rejects in 0.815 ± 0.012 of the 1,000 runs (`POWER_RUNS`).

**Why:** Eq. 10 makes the detectable effect shrink as 1/√n, so the question count grows with the inverse square of the effect you want to detect, as the printed required-n rows show (9893, 2474 and 396 paired questions for 0.01, 0.02 and 0.05). The unpaired rows are this program's extension: Miller's Eq. 9 assumes a paired analysis, and its worked example (variance 1/9, δ = 0.03) gives n = 968.997 unrounded, printed as a check. The power check is a consistency test of the formula on the same toy pool, and the formula ignores the far rejection tail (line 535), which is negligible at these effect sizes. All of these numbers are only as good as the variance estimate behind them (here, the pool variance of one trained pair of toy models), and the question count they imply says nothing about real evals or clustered questions (Miller Sec. 2.2).

**Script reference:** `01-foundations/microerrorbars.py`, lines 485-490 (`minimum_detectable_effect`), lines 493-498 (`required_sample_size`), lines 506-513 (paired and unpaired variances), lines 532-556 (power check)

</details>

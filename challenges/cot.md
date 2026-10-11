# MicroCoT Challenges

Test your understanding of reasoning traces and self-consistency by predicting what happens in these scenarios. Try to work out the answer before revealing it. Numbers quoted in the answers are the pre-registered seed-42 values that the default run of `03-systems/microcot.py` must reproduce exactly under CPython 3.12.8 on the Apple M1 Pro qualification host (the program prints every one of them); other platforms may differ slightly. They describe this toy digit task and tiny model only; none of them is a claim about large language models or natural-language reasoning.

---

### Challenge 1: Two Answers With the Same Count

**Setup:** `tally` (lines 682-691) counts each parsed answer and adds up its samples' Eq. 1 weights, `exp(sum_logp / n_tok)` (line 689). `aggregate` (lines 694-718) with `rule="majority"` takes the answers with the most votes; if more than one remains, it keeps those with the highest summed Eq. 1 weight (lines 704-705) and returns the smallest of them (line 706). Suppose 40 samples give 13 votes to 3 and 13 votes to 5 with summed Eq. 1 weights 3.1 and 2.4, and the other 14 votes are scattered over four digits.

**Question:** What does `aggregate(samples, "majority")` return? What would it return if the two weight sums were exactly equal? On the real model at τ = 0.7, how often does the tie rule decide the vote, and which branch decides it?

<details>
<summary>Reveal Answer</summary>

**Answer:** `(3, "eq1-weight")`. With exactly equal sums both digits survive line 705 and the smaller one wins: `(3, "smallest-digit")`. On the real model at τ = 0.7 the vote ties in 51, 28, 15 and 11 of the 500 test problems at k = 5, 10, 20 and 40, and every one of them is resolved by the Eq. 1 weight; the smallest-digit fallback is never reached.

**Why:** Wang et al. define the vote as `argmax_a Σ_i 1(a_i = a)` and do not say what to do with a tie, so the program needs its own rule, and it must be deterministic or two runs could disagree. The Eq. 1 weight is the paper's own length-normalized path probability, so preferring the higher sum prefers the answer whose traces the model itself found more probable on average. Two sums of floating-point exponentials are almost never exactly equal, which is why the last-resort rule is only exercised by the hand-made samples in `vote_unit_checks` (line 738). Ties are common at small k simply because few votes are spread over up to ten digits.

**Script reference:** `03-systems/microcot.py`, lines 682-691 (`tally`), lines 694-718 (`aggregate`), lines 704-706 (tie rule), lines 736-738 (synthetic tie checks), lines 1064-1083 (real-model tie counts)

</details>

---

### Challenge 2: Nothing to Count

**Setup:** `parse_answer` (lines 638-651) reads the digit right after the first `#` of a trace and returns `None` otherwise. `tally` skips `None` answers, and `aggregate` returns `(None, "abstain")` when no sample is left (line 698). The sweep scores each decoder with `hits += ans == g` (line 1072).

**Question:** A sample ends `L12#.` (the end token follows `#`). Does it vote? If all of a problem's first k samples are like that, what does the vote return and how is it scored? How often does the real model abstain at τ = 0.7?

<details>
<summary>Reveal Answer</summary>

**Answer:** It casts no vote: the token after `#` is the end token, not a digit, so `parse_answer` returns `None`. If no sample parses, `aggregate` returns `(None, "abstain")`, and `None == gold` is `False`, so the problem counts as wrong. At τ = 0.7 the vote abstains on 2 of 500 problems at k = 1 and on none at k = 5 or more; 99.8 percent of all 20,000 samples parse.

**Why:** The parser follows the paper's task-specific parser (footnote 1): read a fixed marker, take what follows, and ignore samples that do not fit. It never checks whether the written partial sums add up, because the vote is over answers, not over correct traces. Replacing an abstention with a guess (for example the most common training answer, which is right only 0.098 of the time on this test slice) would credit the decoder with an answer the model never wrote. A larger k makes abstention rarer because a single parseable sample is enough to cast a vote.

**Script reference:** `03-systems/microcot.py`, lines 638-651 (`parse_answer`), lines 682-698 (`tally` skips `None`; abstention), lines 739-740 (synthetic abstention checks), lines 1064-1083 (scoring, abstention counts and oracle@k)

</details>

---

### Challenge 3: Hotter Samples, Better Votes?

**Setup:** The sampling stream `sample_rng` (line 1040) draws 40 traces per problem at each temperature in `TAU_SWEEP` = (0.3, 0.5, 0.7, 1.0, 1.5) in that order (lines 1043-1056); `generate` divides the logits by τ before sampling (lines 621-622). Each line of the sweep then votes over the first k of those samples (lines 1064-1083). Greedy trace accuracy on the same 500 problems is 0.702.

**Question:** As τ rises from 0.3 to 1.5, what happens to the accuracy of a single sample and to the k = 40 vote? Is there a temperature at which even the k = 40 vote fails to beat greedy?

<details>
<summary>Reveal Answer</summary>

**Answer:** Single-sample accuracy falls steadily: 0.686, 0.665, 0.639, 0.592 and 0.510. The k = 40 vote stays well above greedy at every temperature: 0.786, 0.806, 0.806, 0.826 and 0.800. A single vote at k = 1 is below greedy everywhere (0.698 at τ = 0.3 down to 0.500 at τ = 1.5); at τ = 0.7 the vote needs about five samples to pass greedy (0.748 at k = 5).

**Why:** A hotter temperature flattens every next-token distribution, so each sample makes more step mistakes and single samples get worse. It also spreads the samples over more routes and more different mistakes, and a vote only needs the correct digit to be the most common answer, not most samples to be right. Wrong traces scatter over many digits while correct traces from any route agree, so moderate heat can help the vote even while it hurts each sample. Too much heat eventually costs the vote as well: it already slips from 0.826 at τ = 1.0 to 0.800 at τ = 1.5. The pre-registered gate is checked only at τ = 0.7, the paper's temperature for PaLM and GPT-3; the other temperatures are report-only and were not used to choose anything.

**Script reference:** `03-systems/microcot.py`, lines 621-622 (temperature and the uniform draw), lines 1040-1056 (one sampling stream, consumed temperature by temperature), lines 1057-1083 (single-sample accuracy and the per-k table), lines 1101-1110 (vote versus greedy at τ = 0.7)

</details>

---

### Challenge 4: Let the Model Choose, or Force the Route?

**Setup:** During training, `train_arm` draws each example's route uniformly from L, R and T (line 534), so every route is an equally correct target. At test time greedy decoding picks the route token by argmax like any other token (line 617), while the report-only section forces the route token and decodes the rest greedily (line 1139). Greedy trace decoding scores 0.702 and chooses L, R and T in 100, 71 and 329 of the 500 problems (line 1022).

**Question:** If the route token were forced to T on every problem instead of letting greedy choose, would accuracy rise or fall? Is T the best single route? How does the k = 40 vote (0.806) compare with the best single route?

<details>
<summary>Reveal Answer</summary>

**Answer:** It would rise: route T alone scores 0.728. T is not the best route, though; route L alone scores 0.738, route R only 0.594, and on 0.936 of the problems at least one forced route is right. The vote beats the best single route by +34 problems (+0.068), less than its +0.104 over greedy.

**Why:** Training gives the route token no signal about which route will be right on a given problem, because every route was always a correct target. Greedy's argmax route is therefore not chosen for being right, and forcing T everywhere already beats it. Part of the vote's gain over greedy is simply undoing that choice. The fairer question is the margin over the best route, and even that comparison flatters the route: L was identified as best only after looking at this same test slice, which is why the program labels it post hoc. What remains depends on samples along different routes making different mistakes, and the error-overlap line shows that this holds only partly: pairs of routes fail together more often than independence would predict.

**Script reference:** `03-systems/microcot.py`, line 534 (uniform route in training), line 617 (greedy argmax), line 1022 (greedy route counts), lines 1135-1153 (forced routes, oracle and error overlap), lines 1154-1159 (post hoc margin over the best route)

</details>

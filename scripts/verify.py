"""Local verification runner for no-magic algorithm scripts.

Modes (mutually exclusive):
  --quick      Syntax check, seed presence, and import validation (~2 seconds).
  --behavior   Focused numerical property checks that load the actual teaching
               source (main guard disabled) in isolated subprocesses: tokenizer
               roundtrips, cached versus uncached logits, RoPE identities, frozen
               LoRA base weights and quantization error bounds.
  --list-json  Print the selected repository-relative script paths as a JSON
               array and exit without running anything.
  (default)    Full end-to-end execution of every selected script with its
               default settings and a 1800-second deadline per script. Each
               script's native stdout/stderr is passed through unchanged and its
               pass/fail/timeout status is reported.

Selectors (default: every top-level .py in the four tier directories,
comparison scripts included):
  --section TIER         one tier directory
  SCRIPT ...             specific script filenames
  --changed-from COMMIT  runnable tier scripts added, modified or renamed
                         between merge-base(COMMIT, HEAD) and HEAD

Usage:
    python scripts/verify.py --quick                   # fast local gate
    python scripts/verify.py --behavior                # numerical properties
    python scripts/verify.py                           # full suite
    python scripts/verify.py --section 01-foundations  # one tier
    python scripts/verify.py microgpt.py               # specific script(s)
    python scripts/verify.py --list-json --changed-from origin/main
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import multiprocessing
import os
import py_compile
import queue
import random
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["01-foundations", "02-alignment", "03-systems", "04-agents"]
# Execution ceiling per script for the full run. This is a supervision deadline,
# not certification of the ten-minute teaching target in CONTRIBUTING.md.
TIMEOUT_SECONDS = 1800
BEHAVIOR_TIMEOUT_SECONDS = 300
# One unit of float64 rounding. Numerical tolerances below are small multiples
# of this, scaled by the magnitudes involved, never fitted to observed output.
EPS = sys.float_info.epsilon


def discover_scripts() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for section in SECTIONS:
        section_dir = REPO_ROOT / section
        if not section_dir.is_dir():
            continue
        scripts = sorted(p for p in section_dir.glob("*.py") if p.name != "__init__.py")
        if scripts:
            result[section] = scripts
    return result


def filter_by_section(
    all_scripts: dict[str, list[Path]], section: str
) -> dict[str, list[Path]]:
    if section not in all_scripts:
        valid = ", ".join(all_scripts.keys())
        print(f"Error: unknown section '{section}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)
    return {section: all_scripts[section]}


def filter_by_names(
    all_scripts: dict[str, list[Path]], names: list[str]
) -> dict[str, list[Path]]:
    matches: dict[str, list[tuple[str, Path]]] = {}
    for section, paths in all_scripts.items():
        for p in paths:
            matches.setdefault(p.name, []).append((section, p))

    unrecognized = [n for n in names if n not in matches]
    if unrecognized:
        print(
            f"Error: unrecognized script(s): {', '.join(unrecognized)}",
            file=sys.stderr,
        )
        print(f"Available: {', '.join(sorted(matches.keys()))}", file=sys.stderr)
        sys.exit(1)

    # A bare filename present in more than one tier names no single script;
    # refuse it rather than silently running whichever tier was found last.
    ambiguous = [n for n in names if len(matches[n]) > 1]
    if ambiguous:
        for name in ambiguous:
            locations = ", ".join(
                str(p.relative_to(REPO_ROOT)) for _, p in matches[name]
            )
            print(
                f"Error: script name {name!r} is ambiguous: {locations}",
                file=sys.stderr,
            )
        sys.exit(1)
    lookup = {name: found[0] for name, found in matches.items()}

    result: dict[str, list[Path]] = {}
    for name in names:
        section, path = lookup[name]
        result.setdefault(section, []).append(path)
    return result


class SelectionError(Exception):
    """A --changed-from selection could not be computed from verified Git state."""


def run_git(*args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise SelectionError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def filter_by_changed(
    all_scripts: dict[str, list[Path]], commit: str
) -> dict[str, list[Path]]:
    """Select runnable tier scripts changed between merge-base(commit, HEAD) and HEAD.

    Added, modified, copied and renamed paths are selected; deletions select
    nothing. A changed path that looks like a tier program but is not a
    discovered regular file (for example a symlink) is rejected.
    """
    if not commit or commit.startswith("-"):
        raise SelectionError(f"invalid commit argument {commit!r}")
    toplevel = Path(os.fsdecode(run_git("rev-parse", "--show-toplevel").strip()))
    if toplevel.resolve() != REPO_ROOT:
        raise SelectionError(
            f"{REPO_ROOT} is not the top level of its Git worktree ({toplevel})"
        )
    base = (
        run_git("rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}")
        .decode()
        .strip()
    )
    merge_base = run_git("merge-base", base, "HEAD").decode().strip()
    raw = run_git(
        "diff",
        "-z",
        "--name-status",
        "--find-renames",
        "--no-ext-diff",
        merge_base,
        "HEAD",
        "--",
    )
    fields = [os.fsdecode(f) for f in raw.split(b"\0") if f]
    discovered = {
        p.relative_to(REPO_ROOT).as_posix(): (section, p)
        for section, paths in all_scripts.items()
        for p in paths
    }
    tier_program = re.compile(r"^(" + "|".join(SECTIONS) + r")/[^/]+\.py$")
    selected: set[str] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        kind = status[:1]
        if kind in {"R", "C"}:
            changed = fields[index + 2]
            index += 3
        elif kind in {"A", "M", "D", "T"}:
            changed = fields[index + 1]
            index += 2
        else:
            raise SelectionError(f"unexpected git diff status {status!r}")
        if kind == "D" or not tier_program.match(changed):
            continue
        if Path(changed).name == "__init__.py":
            continue
        candidate = REPO_ROOT / changed
        if (
            kind == "T"
            or changed not in discovered
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise SelectionError(
                f"changed path {changed!r} is not a discovered regular tier program"
            )
        selected.add(changed)
    result: dict[str, list[Path]] = {}
    for relative in sorted(selected):
        section, path = discovered[relative]
        result.setdefault(section, []).append(path)
    return result


def ordered_targets(targets: dict[str, list[Path]]) -> list[Path]:
    return [p for section in SECTIONS for p in targets.get(section, [])]


ALLOWED_MODULES = {
    "os",
    "math",
    "random",
    "json",
    "struct",
    "urllib",
    "collections",
    "itertools",
    "functools",
    "string",
    "hashlib",
    "time",
    "sys",
    "argparse",
    "textwrap",
    "io",
    "copy",
    "abc",
    "typing",
}


def check_syntax(script_path: Path) -> str | None:
    """Return error message if syntax is invalid, else None."""
    try:
        py_compile.compile(str(script_path), doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return str(e)


def check_seed(script_path: Path) -> bool:
    """Return True if random.seed(42) is present."""
    text = script_path.read_text()
    return bool(re.search(r"random\.seed\(42\)", text))


def check_imports(script_path: Path) -> list[str]:
    """Return list of non-stdlib imports found."""
    text = script_path.read_text()
    bad = []
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", text, re.MULTILINE):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_MODULES and root != "__future__":
            bad.append(m.group(1))
    return bad


def run_quick(targets: dict[str, list[Path]]) -> bool:
    """Run fast checks: syntax, seed, imports. Return True if all pass."""
    any_failed = False
    checked = 0

    for section in SECTIONS:
        if section not in targets:
            continue
        for script_path in targets[section]:
            label = str(script_path.relative_to(REPO_ROOT))
            checked += 1
            errors = []

            syntax_err = check_syntax(script_path)
            if syntax_err:
                errors.append(f"syntax: {syntax_err}")

            # Only check seed/imports for micro*.py scripts (skip comparison scripts)
            if script_path.name.startswith("micro"):
                if not check_seed(script_path):
                    errors.append("missing random.seed(42)")

                bad_imports = check_imports(script_path)
                if bad_imports:
                    errors.append(f"external imports: {', '.join(bad_imports)}")

            if errors:
                any_failed = True
                print(f"  FAIL  {label}")
                for e in errors:
                    print(f"        {e}")
            else:
                print(f"  OK    {label}")

    print()
    status = "FAIL" if any_failed else "PASS"
    print(f"Quick check: {checked} scripts — {status}")
    return not any_failed


# === BEHAVIOR CHECKS ===
# Each check runs in a freshly spawned interpreter inside a temporary working
# directory. It loads the actual teaching source with runpy (so the
# `if __name__ == "__main__":` block does not run, and no dataset download or
# training happens) and exercises the real functions on small fixtures. Nothing
# here re-implements the algorithm under test.


class BehaviorFailure(Exception):
    """A protected numerical property does not hold for the teaching source."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BehaviorFailure(message)


def load_source(relative: str, representative: str) -> dict[str, Any]:
    """Execute the teaching source without its main block; return its live globals.

    runpy returns a copy of the module namespace, so the live namespace is taken
    from one of its functions; rebinding a name there is seen by every function.
    """
    namespace = runpy.run_path(
        str(REPO_ROOT / relative), run_name="__no_magic_behavior__"
    )
    live: dict[str, Any] = namespace[representative].__globals__
    return live


def check_tokenizer() -> str:
    t = load_source("01-foundations/microtokenizer.py", "encode")
    corpus = "banana banana 🍌 漢字 café banana banana " * 3
    num_merges = 12
    merges = t["train_bpe"](list(corpus.encode("utf-8")), num_merges)
    require(
        len(merges) == num_merges,
        f"train_bpe learned {len(merges)} merges, expected {num_merges}",
    )
    new_ids = [new_id for _, new_id in merges]
    require(
        new_ids == list(range(256, 256 + num_merges)),
        f"merge ids {new_ids} are not consecutive ids above the byte range",
    )
    vocab = t["build_vocab"](merges)
    samples = [corpus, "", "aaaaaa", "O'Brien", "漢字🍌café", "banana"]
    for sample in samples:
        encoded = t["encode"](sample, merges)
        decoded = t["decode"](encoded, vocab)
        require(decoded == sample, f"roundtrip changed {sample!r} into {decoded!r}")
    corpus_tokens = t["encode"](corpus, merges)
    require(
        len(corpus_tokens) < len(corpus.encode("utf-8")),
        "learned merges do not compress the training corpus",
    )
    require(
        any(token >= 256 for token in corpus_tokens),
        "encoding the training corpus applies no learned merge",
    )
    return (
        f"{len(merges)} learned merges, {len(samples)} roundtrips, "
        f"{len(corpus.encode('utf-8'))} bytes -> {len(corpus_tokens)} tokens"
    )


def check_kv_cache() -> str:
    v = load_source("03-systems/microkv.py", "generate_with_cache")
    vocab_size = 6
    d = v["N_EMBD"]
    n_layer = v["N_LAYER"]
    weights: dict[str, list[list[float]]] = {
        "wte": v["extract"](v["make_matrix"](vocab_size, d)),
        "wpe": v["extract"](v["make_matrix"](v["BLOCK_SIZE"], d)),
        "lm_head": v["extract"](v["make_matrix"](vocab_size, d)),
    }
    for layer in range(n_layer):
        for name, rows, cols in [
            ("wq", d, d),
            ("wk", d, d),
            ("wv", d, d),
            ("wo", d, d),
            ("fc1", 4 * d, d),
            ("fc2", d, 4 * d),
        ]:
            weights[f"l{layer}.{name}"] = v["extract"](v["make_matrix"](rows, cols))

    # Observe the logits each generator actually computes by wrapping the
    # module's own linear_f; the lm_head projection output is the logit vector.
    original_linear = v["linear_f"]
    captured: list[list[float]] = []

    def observing_linear(
        x: list[float], w: list[list[float]], counter: list[int]
    ) -> list[float]:
        out: list[float] = original_linear(x, w, counter)
        if w is weights["lm_head"]:
            captured.append(list(out))
        return out

    v["linear_f"] = observing_linear
    gen_len = 5
    tokens_plain, _ = v["generate_no_cache"](2, weights, vocab_size, gen_len)
    logits_plain = list(captured)
    captured.clear()
    tokens_cached, _, cache_sizes = v["generate_with_cache"](
        2, weights, vocab_size, gen_len
    )
    logits_cached = list(captured)

    require(tokens_plain == tokens_cached, "cached and uncached tokens differ")
    require(
        len(logits_plain) == len(logits_cached) == gen_len,
        f"expected {gen_len} logit vectors from each generator, got "
        f"{len(logits_plain)} and {len(logits_cached)}",
    )
    worst = 0.0
    for step, (plain, cached) in enumerate(zip(logits_plain, logits_cached)):
        require(len(plain) == len(cached) == vocab_size, "logit width mismatch")
        for a, b in zip(plain, cached):
            # Both generators perform the same float64 operations on the same
            # keys and values; allow a few ulps of the operand magnitude.
            tolerance = 64 * EPS * max(1.0, abs(a), abs(b))
            require(
                abs(a - b) <= tolerance,
                f"step {step}: cached logit {b!r} differs from uncached {a!r}",
            )
            worst = max(worst, abs(a - b))
    require(
        max(logits_plain[0]) - min(logits_plain[0]) > 1e-6,
        "fixture logits are constant across the vocabulary; the comparison is vacuous",
    )
    require(
        any(abs(a - b) > 1e-6 for a, b in zip(logits_plain[0], logits_plain[1])),
        "fixture logits do not change between positions; the comparison is vacuous",
    )
    expected_cache = [2 * n_layer * d * (i + 1) for i in range(gen_len)]
    require(
        cache_sizes == expected_cache,
        f"cache sizes {cache_sizes} do not grow as {expected_cache}",
    )
    return f"{gen_len} positions, max |cached - uncached| = {worst:.3g}, cache grows"


def check_rope() -> str:
    r = load_source("03-systems/microrope.py", "apply_rope")
    head_dim = r["HEAD_DIM"]
    freqs = r["rope_frequencies"](head_dim)
    rng = random.Random(7)
    q = [rng.uniform(-2.0, 2.0) for _ in range(head_dim)]
    k = [rng.uniform(-2.0, 2.0) for _ in range(head_dim)]

    def norm(vec: list[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    for pos in [0, 1, 7, 63, 500]:
        rotated = r["apply_rope"](q, pos, freqs)
        tolerance = 64 * EPS * norm(q)
        require(
            abs(norm(rotated) - norm(q)) <= tolerance,
            f"apply_rope changes the vector norm at position {pos}",
        )

    def score(pos_q: int, pos_k: int) -> float:
        value: float = r["rope_attention_score"](q, k, pos_q, pos_k, freqs)
        return value

    # Rotations compose, so <R(m)q, R(n)k> depends only on n - m. Each angle is
    # computed from a position up to ~600 rad, where float64 sin/cos carry an
    # absolute error of about pos * EPS; the bound scales with that.
    magnitude = norm(q) * norm(k)
    for offset in [0, 3, -5]:
        reference = score(2, 2 + offset)
        for shift in [10, 50, 100, 500]:
            shifted = score(2 + shift, 2 + offset + shift)
            tolerance = 64 * EPS * magnitude * (shift + 8) * head_dim
            require(
                abs(shifted - reference) <= tolerance,
                f"score at offset {offset} changes under shift {shift}: "
                f"{reference!r} vs {shifted!r}",
            )
    sensitivity = abs(score(2, 5) - score(2, 6))
    require(
        sensitivity > 1e-6 * magnitude,
        "rope_attention_score does not depend on the relative position",
    )
    return (
        f"norm preserved, relative identity over 4 shifts x 3 offsets, "
        f"|score(2,5) - score(2,6)| = {sensitivity:.3g}"
    )


def extract_lora_phase(source: Path) -> tuple[ast.Module, ast.Module]:
    """Return the actual LoRA setup statements and update loop from the main block."""
    module = ast.parse(source.read_text(encoding="utf-8"))
    main = [
        node
        for node in module.body
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
    ]
    require(len(main) == 1, "microlora.py must have exactly one main block")
    setup_names = {"lora_adapters", "lora_param_list", "m_lora", "v_lora"}
    setup = [
        node
        for node in main[0].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in setup_names
            for target in node.targets
        )
    ]
    assigned = {
        target.id
        for node in setup
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    require(
        assigned == setup_names,
        f"LoRA setup assigns {sorted(assigned)}, expected {sorted(setup_names)}",
    )
    loops = [
        node
        for node in main[0].body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and any(
            isinstance(n, ast.Name) and n.id == "LORA_STEPS"
            for n in ast.walk(node.iter)
        )
    ]
    require(len(loops) == 1, "microlora.py must have exactly one LORA_STEPS loop")
    return (
        ast.Module(body=list(setup), type_ignores=[]),
        ast.Module(body=[loops[0]], type_ignores=[]),
    )


def check_lora() -> str:
    relative = "02-alignment/microlora.py"
    lora = load_source(relative, "gpt_forward")
    setup, loop = extract_lora_phase(REPO_ROOT / relative)
    # Tiny deterministic fixture standing in for the names dataset the main
    # block would download: two characters plus BOS, three short documents.
    lora.update(
        {
            "params": lora["init_parameters"](3),
            "BOS": 2,
            "VOCAB_SIZE": 3,
            "unique_chars": ["n", "z"],
            "lora_docs": ["nz", "zn", "nn"],
            "LORA_STEPS": 3,
        }
    )
    lora["base_param_list"] = lora["flatten_params"](lora["params"])
    base_before = [p.data for p in lora["base_param_list"]]
    # exec runs statements parsed from the repository's own teaching source, so
    # the check exercises the real update loop rather than a copy of it.
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(setup, f"<{relative}: LoRA setup>", "exec"), lora)  # noqa: S102
        adapters_before = [p.data for p in lora["lora_param_list"]]
        exec(compile(loop, f"<{relative}: LoRA update loop>", "exec"), lora)  # noqa: S102
    base_after = [p.data for p in lora["base_param_list"]]
    adapters_after = [p.data for p in lora["lora_param_list"]]
    require(
        base_after == base_before,
        f"{sum(a != b for a, b in zip(base_before, base_after))} frozen base "
        f"weights changed during the LoRA update loop",
    )
    changed = sum(a != b for a, b in zip(adapters_before, adapters_after))
    require(changed > 0, "the LoRA update loop changed no adapter weight")
    return (
        f"{len(base_before)} base weights unchanged, "
        f"{changed}/{len(adapters_after)} adapter weights updated"
    )


def check_quantization() -> str:
    q = load_source("03-systems/microquant.py", "quantize_zeropoint_int8")

    def bound(weights: list[list[float]], step: float) -> float:
        # Round-to-nearest on a uniform grid errs by at most half a step. The
        # float64 divide/round/multiply adds a few ulps of the magnitudes
        # involved (|w| and the grid extent).
        extent = max(abs(w) for row in weights for w in row) + 256 * step
        return step / 2 + 16 * EPS * extent

    def max_error(a: list[list[float]], b: list[list[float]]) -> float:
        return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))

    zero = [[0.0, 0.0, 0.0]]
    positive_constant = [[0.73, 0.73]]
    negative_constant = [[-0.73, -0.73, -0.73]]
    mixed_sign = [[-1.73, -0.01, 0.0], [0.04, 0.91, 1.2]]
    shifted = [[2.0, 2.5, 3.1], [2.2, 2.9, 2.05]]
    fixtures = {
        "zero": zero,
        "positive constant": positive_constant,
        "negative constant": negative_constant,
        "mixed sign": mixed_sign,
        "shifted": shifted,
    }
    checked = 0

    for suffix, lowest, limit in [("int8", -127, 127), ("int4", -8, 7)]:
        for label, weights in fixtures.items():
            encoded, scale = q[f"quantize_absmax_{suffix}"](weights)
            restored = q["dequantize_absmax"](encoded, scale)
            max_abs = max(abs(w) for row in weights for w in row)
            expected_scale = max_abs / limit if max_abs > 0 else 1.0
            require(
                math.isclose(scale, expected_scale, rel_tol=4 * EPS),
                f"absmax {suffix} {label}: scale {scale!r} is not max|W|/{limit}",
            )
            require(
                all(lowest <= c <= limit for row in encoded for c in row),
                f"absmax {suffix} {label}: code outside [{lowest}, {limit}]",
            )
            error = max_error(weights, restored)
            require(
                error <= bound(weights, scale),
                f"absmax {suffix} {label}: error {error!r} exceeds half a step",
            )
            checked += 1

    for label, weights in fixtures.items():
        encoded, scale, zero_point = q["quantize_zeropoint_int8"](weights)
        restored = q["dequantize_zeropoint"](encoded, scale, zero_point)
        values = [w for row in weights for w in row]
        low, high = min(values), max(values)
        if low == high == 0.0:
            # The all-zero tensor keeps its existing exact representation.
            require(
                scale == 1.0 and zero_point == 0 and restored == weights,
                "zero-point int8 zero: all-zero tensor is not reconstructed exactly",
            )
            checked += 1
            continue
        if low == high:
            # Degenerate interval: the grid must span [min(c, 0), max(c, 0)].
            low, high = min(low, 0.0), max(high, 0.0)
        require(
            math.isclose(scale, (high - low) / 255, rel_tol=4 * EPS),
            f"zero-point int8 {label}: scale {scale!r} is not (max - min)/255 "
            f"over [{low}, {high}]",
        )
        require(
            all(0 <= c <= 255 for row in encoded for c in row),
            f"zero-point int8 {label}: code outside [0, 255]",
        )
        error = max_error(weights, restored)
        require(
            error <= bound(weights, scale),
            f"zero-point int8 {label}: error {error!r} exceeds half a step "
            f"(reconstructed {restored!r})",
        )
        checked += 1

    unequal_rows = [[100.0, 0.13, -10.0], [0.0, 0.0, 0.0], [0.002, -0.001, 0.0005]]
    for label, weights in {**fixtures, "unequal row ranges": unequal_rows}.items():
        encoded, scales = q["quantize_per_channel_int8"](weights)
        restored = q["dequantize_per_channel"](encoded, scales)
        require(len(scales) == len(weights), f"per-channel {label}: one scale per row")
        for row, row_codes, row_restored, scale in zip(
            weights, encoded, restored, scales
        ):
            max_abs = max(abs(w) for w in row)
            expected_scale = max_abs / 127 if max_abs > 0 else 1.0
            require(
                math.isclose(scale, expected_scale, rel_tol=4 * EPS),
                f"per-channel {label}: row scale {scale!r} is not max|row|/127",
            )
            require(
                all(-127 <= c <= 127 for c in row_codes),
                f"per-channel {label}: code outside [-127, 127]",
            )
            error = max_error([row], [row_restored])
            require(
                error <= bound([row], scale),
                f"per-channel {label}: row error {error!r} exceeds half its step",
            )
        checked += 1
    return f"{checked} scheme/fixture combinations within half a quantization step"


BEHAVIOR_CHECKS: dict[str, tuple[str, Callable[[], str]]] = {
    "tokenizer roundtrip and compression": (
        "01-foundations/microtokenizer.py",
        check_tokenizer,
    ),
    "kv cache logits and growth": ("03-systems/microkv.py", check_kv_cache),
    "rope norm, relative position and sensitivity": (
        "03-systems/microrope.py",
        check_rope,
    ),
    "lora frozen base during update loop": ("02-alignment/microlora.py", check_lora),
    "quantization error bounds": ("03-systems/microquant.py", check_quantization),
}


def behavior_child(name: str, results: Queue[tuple[str, str]]) -> None:
    """Entry point of the spawned interpreter that runs one behavior check.

    A violated property is reported through the queue. Any other exception is
    left to propagate: the child prints its traceback to stderr and exits
    nonzero, which the parent reports as an error.
    """
    with tempfile.TemporaryDirectory(prefix="no-magic-behavior-") as workdir:
        os.chdir(workdir)
        random.seed(42)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                detail = BEHAVIOR_CHECKS[name][1]()
        except BehaviorFailure as failure:
            results.put(("fail", str(failure)))
            return
    results.put(("pass", detail))


def await_result(
    child: multiprocessing.process.BaseProcess, results: Queue[tuple[str, str]]
) -> tuple[str, str]:
    """Wait for the child's result, its exit, or the behavior deadline."""
    deadline = time.monotonic() + BEHAVIOR_TIMEOUT_SECONDS
    while True:
        try:
            return results.get(timeout=0.5)
        except queue.Empty:
            if not child.is_alive():
                break
            if time.monotonic() > deadline:
                child.kill()
                child.join()
                return "timeout", f"no result within {BEHAVIOR_TIMEOUT_SECONDS} seconds"
    try:
        return results.get(timeout=1)
    except queue.Empty:
        child.join()
        return "error", f"child exited with code {child.exitcode}; traceback above"


def run_behavior() -> bool:
    """Run every behavior check in its own spawned interpreter."""
    context = multiprocessing.get_context("spawn")
    any_failed = False
    for name, (relative, _) in BEHAVIOR_CHECKS.items():
        results: Queue[tuple[str, str]] = context.Queue()
        child = context.Process(target=behavior_child, args=(name, results))
        child.start()
        status, detail = await_result(child, results)
        child.join()
        if status == "pass" and child.exitcode != 0:
            status, detail = "error", f"child exited with code {child.exitcode}"
        if status == "pass":
            print(f"  OK    {name} [{relative}]: {detail}")
        else:
            any_failed = True
            print(f"  FAIL  {name} [{relative}] ({status})")
            for line in detail.splitlines():
                print(f"        {line}")
    print()
    print(
        f"Behavior check: {len(BEHAVIOR_CHECKS)} properties — "
        f"{'FAIL' if any_failed else 'PASS'}"
    )
    return not any_failed


def terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill whatever remains of the script's process group, then reap the leader."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        # Every process in the group has already exited.
        pass
    proc.wait()


def run_script(script_path: Path) -> tuple[str, float]:
    """Run script, return (status, elapsed_seconds). Status: 'pass', 'fail', 'timeout'.

    The script runs in its own session/process group with the verifier's
    stdout/stderr inherited, so its native output is retained unchanged. On
    timeout or completion the whole process group is killed, so no grandchild
    outlives the deadline.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=TIMEOUT_SECONDS)
        status = "pass" if returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        terminate_process_group(proc)
    return status, round(time.monotonic() - start, 1)


def format_duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s"


def print_summary(results: list[tuple[str, str, float]]) -> None:
    """Print summary table. results: list of (label, status, elapsed)."""
    print()
    print("Verify results")
    print("─" * 55)
    passed = failed = timed_out = 0
    for label, status, elapsed in results:
        if status == "pass":
            marker = "Pass"
            passed += 1
        elif status == "timeout":
            marker = "TIMEOUT"
            timed_out += 1
        else:
            marker = "FAIL"
            failed += 1
        dots = "." * max(1, 40 - len(label))
        print(f"  {label} {dots} {marker}  {format_duration(elapsed)}")

    total = len(results)
    print("─" * 55)
    parts = [f"{passed}/{total} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if timed_out:
        parts.append(f"{timed_out} timed out")
    print("  " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-magic scripts locally and report pass/fail/timeout."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Fast gate: syntax, seed, and import checks only (~2 seconds).",
    )
    mode.add_argument(
        "--behavior",
        action="store_true",
        help="Numerical property checks against the actual teaching source.",
    )
    mode.add_argument(
        "--list-json",
        action="store_true",
        help="Print the selected script paths as a JSON array; run nothing.",
    )
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        help="Run only scripts in the given tier directory.",
    )
    parser.add_argument(
        "--changed-from",
        metavar="COMMIT",
        help=(
            "Select runnable tier scripts added, modified or renamed between "
            "merge-base(COMMIT, HEAD) and HEAD."
        ),
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Specific script filenames to verify (e.g. microgpt.py).",
    )
    args = parser.parse_args()

    if args.section and args.scripts:
        parser.error("--section and positional script names are mutually exclusive.")
    if args.changed_from is not None and (args.section or args.scripts):
        parser.error("--changed-from is mutually exclusive with --section and names.")
    if args.behavior and (args.section or args.scripts or args.changed_from):
        parser.error("--behavior runs its fixed property set and takes no selector.")

    if args.behavior:
        sys.exit(0 if run_behavior() else 1)

    all_scripts = discover_scripts()
    if not all_scripts:
        print("Error: no algorithm scripts found.", file=sys.stderr)
        sys.exit(1)

    if args.section:
        targets = filter_by_section(all_scripts, args.section)
    elif args.scripts:
        targets = filter_by_names(all_scripts, args.scripts)
    elif args.changed_from is not None:
        try:
            targets = filter_by_changed(all_scripts, args.changed_from)
        except SelectionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        targets = all_scripts

    selected = ordered_targets(targets)
    if args.changed_from is not None and not selected:
        print(
            f"No runnable tier script changed since merge-base with "
            f"{args.changed_from}; nothing was executed (this is not a runtime pass).",
            file=sys.stderr,
        )

    if args.list_json:
        print(json.dumps([p.relative_to(REPO_ROOT).as_posix() for p in selected]))
        sys.exit(0)

    if args.quick:
        ok = run_quick(targets)
        sys.exit(0 if ok else 1)

    if not selected:
        sys.exit(0)

    results: list[tuple[str, str, float]] = []
    any_failed = False

    for script_path in selected:
        label = str(script_path.relative_to(REPO_ROOT))
        print(f"==> Running {label}", flush=True)
        status, elapsed = run_script(script_path)
        print(f"<== {label}: {status} after {format_duration(elapsed)}", flush=True)
        if status != "pass":
            any_failed = True
        results.append((label, status, elapsed))

    print_summary(results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()

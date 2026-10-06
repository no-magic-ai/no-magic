"""
Generate catalog.json from all algorithm scripts in the repository.

Extracts tier, filename, display name, thesis docstring, paper slug, line
count, teaching kind, data source and adaptation note for each .py file in the
tier directories. Output is written to docs/catalog.json and consumed by the
no-magic-ai.github.io website and the no-magic-papers invariant validator.

This file is the single authority for per-script catalog metadata. Script
slugs (file basenames) and paper slugs (no-magic-papers card names) are
separate namespaces: every script needs an explicit SCRIPT_TO_PAPER entry and
an explicit SCRIPT_CONTRACTS entry. Neither is inferred from the other or from
the filename. The build fails on a discovered script without both entries and
on entries that name no discovered script — this enforces SOP §7.3 invariant 3
at catalog-generation time.

Usage:
    python scripts/generate_catalog.py          # write docs/catalog.json
    python scripts/generate_catalog.py --check  # exit 1 if it is stale; never writes
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, get_args

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER_DIRS = ["01-foundations", "02-alignment", "03-systems", "04-agents"]
OUTPUT = REPO_ROOT / "docs" / "catalog.json"

DISPLAY_OVERRIDES: dict[str, str] = {
    "microgpt": "Autoregressive GPT",
    "micrornn": "RNN vs GRU",
    "microlstm": "LSTM",
    "microtokenizer": "BPE Tokenizer",
    "microembedding": "Word Embeddings",
    "microrag": "RAG Pipeline",
    "microbert": "BERT",
    "microconv": "CNN",
    "microdiffusion": "Denoising Diffusion",
    "microgan": "GAN",
    "microoptimizer": "Optimizer Comparison",
    "microvae": "VAE",
    "microvit": "Vision Transformer",
    "microresnet": "ResNet",
    "microlora": "LoRA",
    "microdpo": "DPO",
    "microppo": "PPO (RLHF)",
    "micromoe": "Mixture of Experts",
    "microbatchnorm": "Batch Normalization",
    "microdropout": "Dropout",
    "microgrpo": "GRPO",
    "microqlora": "QLoRA",
    "microreinforce": "REINFORCE",
    "microattention": "Attention Variants",
    "microbeam": "Beam Search",
    "microflash": "Flash Attention",
    "microkv": "KV-Cache",
    "microquant": "Quantization",
    "microrope": "RoPE",
    "microssm": "State Space Models",
    "microcheckpoint": "Activation Checkpointing",
    "micropaged": "PagedAttention",
    "microparallel": "Model Parallelism",
    "microbm25": "BM25",
    "microcomplexssm": "Complex SSM",
    "microdiscretize": "Discretization",
    "microroofline": "Roofline Model",
    "microspeculative": "Speculative Decoding",
    "microvectorsearch": "Vector Search",
    "microbandit": "Multi-Armed Bandit",
    "micromcts": "Monte Carlo Tree Search",
    "micromemory": "Memory-Augmented Network",
    "microminimax": "Minimax + Alpha-Beta",
    "microreact": "ReAct",
    "attention_vs_none": "Attention vs None",
    "adam_vs_sgd": "Adam vs SGD",
    "rnn_vs_gru_vs_lstm": "RNN vs GRU vs LSTM",
}

# Maps each script in this repo to its paper card slug in no-magic-papers.
# Required from no-magic v3.0 per SOP §7.3 invariant 3. Missing entries fail
# the build. Comparison-style scripts (e.g. attention_vs_none, microattention)
# share a paper card per the multi-implementations[] precedent set by mamba-2.
SCRIPT_TO_PAPER: dict[str, str] = {
    "adam_vs_sgd": "adam",
    "attention_vs_none": "transformer",
    "microattention": "transformer",
    "microbandit": "ucb1",
    "microbatchnorm": "batchnorm",
    "microbeam": "nucleus-sampling",
    "microbert": "bert",
    "microbm25": "bm25",
    "microcheckpoint": "gradient-checkpointing",
    "microcomplexssm": "mamba-2",
    "microconv": "lenet-5",
    "microdiffusion": "ddpm",
    "microdiscretize": "mamba-2",
    "microdpo": "dpo",
    "microdropout": "dropout",
    "microembedding": "word2vec",
    "microflash": "flash-attention",
    "microgan": "gan",
    "microgpt": "gpt-1",
    "microgrpo": "grpo",
    "microkv": "kv-cache",
    "microlora": "lora",
    "microlstm": "lstm",
    "micromcts": "uct",
    "micromemory": "ntm",
    "microminimax": "alpha-beta",
    "micromoe": "moe-shazeer",
    "microoptimizer": "adam",
    "micropaged": "pagedattention",
    "microparallel": "megatron-lm",
    "microppo": "ppo",
    "microqlora": "qlora",
    "microquant": "llm-int8",
    "microrag": "rag",
    "microreact": "react",
    "microreinforce": "reinforce",
    "microresnet": "resnet",
    "micrornn": "rnn-elman",
    "microroofline": "roofline",
    "microrope": "rope",
    "microspeculative": "speculative-decoding",
    "microssm": "mamba-2",
    "microtokenizer": "bpe",
    "microturboquant": "turboquant",
    "microvae": "vae",
    "microvectorsearch": "hnsw",
    "microvit": "vit",
    "rnn_vs_gru_vs_lstm": "gru",
}

# What running a script demonstrates. Assigned per script below from its source;
# never inferred from the filename or the thesis.
#   comparison     — trains two or more alternative arms for a common teaching task
#                    and reports arm-versus-arm quality or training-cost outcomes.
#                    Checked first: it applies even when the arms also run inference.
#                    Arms are the trained alternatives compared against each other;
#                    models trained only as auxiliaries of a decoding or serving
#                    method (draft, reference) and sequential stages of one pipeline
#                    are not arms
#   train_infer    — learns parameters or estimates from data, then uses the learned
#                    state for inference, generation, decisions or evaluation,
#                    including one trained model run through alternative decoding or
#                    serving methods
#   forward_pass   — runs untrained forward computations of a model component to
#                    show a mechanism; no learning phase
#   algorithm_demo — runs a non-learning algorithm (scoring, hashing, search,
#                    quantization); no model is trained
TeachingKind = Literal["train_infer", "comparison", "forward_pass", "algorithm_demo"]

# Where the script's data comes from on a fresh checkout.
#   names_download — downloads makemore names.txt via urllib on first run and caches
#                    it as names.txt in the current working directory; needs network
#                    unless that file already exists there
#   in_script      — generated or written inline by the script; no network
DataSource = Literal["names_download", "in_script"]

TEACHING_KINDS: tuple[str, ...] = get_args(TeachingKind)
DATA_SOURCES: tuple[str, ...] = get_args(DataSource)


@dataclass(frozen=True)
class TeachingContract:
    """Source-observed teaching contract for one script.

    `adaptation` discloses, in plain source-state terms, where the script differs
    from the linked paper card (different cited sources, extra or missing
    components, toy-scale simplifications). It is not a paper-replication claim,
    and a missing note does not mean the script replicates its card exactly.
    """

    kind: TeachingKind
    data: DataSource
    adaptation: str | None = None


SCRIPT_CONTRACTS: dict[str, TeachingContract] = {
    "adam_vs_sgd": TeachingContract(
        "comparison",
        "names_download",
        "Trains the same character bigram model with Adam and with SGD on identical data. "
        "The source also cites Robbins & Monro (1951) for SGD; the linked card covers Adam.",
    ),
    "attention_vs_none": TeachingContract(
        "comparison",
        "names_download",
        "Trains two GRU character models, one with additive attention over hidden states "
        "and one without; the source cites Bahdanau et al. (2015). It does not implement "
        "the Transformer self-attention architecture of the linked card.",
    ),
    "microattention": TeachingContract(
        "forward_pass",
        "in_script",
        "Untrained side-by-side forward computations of multi-head, grouped-query, "
        "multi-query and sliding-window attention. Beyond the linked Transformer card, "
        "the source cites Shazeer (2019), Ainslie et al. (2023) and Beltagy et al. (2020).",
    ),
    "microbandit": TeachingContract("comparison", "in_script"),
    "microbatchnorm": TeachingContract("comparison", "in_script"),
    "microbeam": TeachingContract(
        "train_infer",
        "names_download",
        "Trains a small model, then compares six decoding strategies: greedy, temperature, "
        "top-k, top-p (nucleus), beam search and speculative decoding. The linked card "
        "covers nucleus sampling only; the source also cites Leviathan et al. (2023).",
    ),
    "microbert": TeachingContract("train_infer", "names_download"),
    "microbm25": TeachingContract("algorithm_demo", "in_script"),
    "microcheckpoint": TeachingContract(
        "comparison",
        "in_script",
        "Trains two identically initialized MLPs for the same number of SGD steps, one "
        "with standard backpropagation and one with checkpointed recomputation, and "
        "compares loss, time, stored activations and gradient equality. It has no "
        "separate inference phase.",
    ),
    "microcomplexssm": TeachingContract(
        "comparison",
        "in_script",
        "Teaches the complex-to-real (data-dependent RoPE) SSM equivalence. The source "
        "cites Mamba-3 (arXiv:2603.15569) Proposition 3 and RoPE (Su et al., 2021); the "
        "linked Mamba-2 card's structured state space duality is not implemented.",
    ),
    "microconv": TeachingContract("train_infer", "in_script"),
    "microdiffusion": TeachingContract("train_infer", "in_script"),
    "microdiscretize": TeachingContract(
        "comparison",
        "in_script",
        "Compares Euler, zero-order-hold and trapezoidal discretization. The source cites "
        "Mamba-3 (arXiv:2603.15569) Section 3 and S4 (Gu et al., 2022); the linked "
        "Mamba-2 card's structured state space duality is not implemented.",
    ),
    "microdpo": TeachingContract("train_infer", "names_download"),
    "microdropout": TeachingContract("comparison", "names_download"),
    "microembedding": TeachingContract("train_infer", "names_download"),
    "microflash": TeachingContract("forward_pass", "in_script"),
    "microgan": TeachingContract("train_infer", "in_script"),
    "microgpt": TeachingContract(
        "train_infer",
        "names_download",
        "Linked to the GPT-1 card (Radford et al., 2018), but the source says it follows "
        "the GPT-2 architecture (Radford et al., 2019) with RMSNorm instead of "
        "LayerNorm, ReLU instead of GELU and no bias terms, and that its algorithmic flow "
        "was inspired by Karpathy's microgpt.py. It trains a character-level model on "
        "names and samples from it; there is no supervised task fine-tuning stage.",
    ),
    "microgrpo": TeachingContract("train_infer", "names_download"),
    "microkv": TeachingContract(
        "train_infer",
        "names_download",
        "Trains a small model, then compares generation with and without a KV cache "
        "and adds a paged-allocation simulation. Beyond the linked Pope et al. (2022) "
        "card, the source cites Kwon et al. (2023) for the paged allocation.",
    ),
    "microlora": TeachingContract("train_infer", "names_download"),
    "microlstm": TeachingContract("train_infer", "names_download"),
    "micromcts": TeachingContract(
        "algorithm_demo",
        "in_script",
        "Plays tic-tac-toe with UCB1 child selection and uniformly random rollouts. The "
        "source cites Coulom (2006) and Silver et al. (2016) rather than the linked UCT "
        "card (Kocsis & Szepesvari, 2006). Unlike AlphaGo, it trains no policy or value "
        "network.",
    ),
    "micromemory": TeachingContract("train_infer", "in_script"),
    "microminimax": TeachingContract(
        "train_infer",
        "in_script",
        "Trains a small MLP to score Connect Four positions from random-play game "
        "outcomes, then uses it as the leaf evaluator of depth-limited minimax with "
        "alpha-beta pruning and iterative deepening. The source cites Knuth & Moore "
        "(1975) and Shannon (1950); their analysis concerns search and pruning given "
        "position values, not this learned evaluator.",
    ),
    "micromoe": TeachingContract(
        "train_infer",
        "names_download",
        "Per the source, expert MLPs use plain floats with analytically computed "
        "gradients while the router uses scalar autograd, to keep pure-Python runtime "
        "tractable.",
    ),
    "microoptimizer": TeachingContract(
        "comparison",
        "names_download",
        "Trains character bigram models with SGD, Momentum, RMSProp and Adam, plus a "
        "warmup and cosine-decay extension citing Loshchilov & Hutter (2016). The linked "
        "card covers Adam only.",
    ),
    "micropaged": TeachingContract("forward_pass", "in_script"),
    "microparallel": TeachingContract("comparison", "in_script"),
    "microppo": TeachingContract(
        "train_infer",
        "names_download",
        "Per the source, the reward model is a small MLP trained separately with a "
        "pairwise ranking loss and manual SGD, and the reward model and value function "
        "use plain floats without autograd.",
    ),
    "microqlora": TeachingContract(
        "train_infer",
        "names_download",
        "Toy-scale simulation: a one-layer, 16-dimensional base model trained on names, "
        "NF4 levels from a normal-quantile approximation and quantization blocks of 8 "
        "weights (the source notes production QLoRA uses 64). The source says the "
        "architecture follows the microgpt pattern with pedagogical simplifications.",
    ),
    "microquant": TeachingContract("train_infer", "names_download"),
    "microrag": TeachingContract("train_infer", "in_script"),
    "microreact": TeachingContract("train_infer", "in_script"),
    "microreinforce": TeachingContract("comparison", "in_script"),
    "microresnet": TeachingContract("comparison", "in_script"),
    "micrornn": TeachingContract(
        "comparison",
        "names_download",
        "Trains a vanilla RNN and a GRU side by side. The source cites Rumelhart et al. "
        "for the vanilla RNN and Cho et al. (2014) for the GRU; the linked card is "
        "Elman (1990).",
    ),
    "microroofline": TeachingContract("comparison", "in_script"),
    "microrope": TeachingContract("forward_pass", "in_script"),
    "microspeculative": TeachingContract("train_infer", "names_download"),
    "microssm": TeachingContract(
        "train_infer",
        "names_download",
        "The source cites Mamba (Gu & Dao, 2023) and builds a simplified Mamba-style "
        "selective SSM with Euler discretization. The linked Mamba-2 card's structured "
        "state space duality is not implemented.",
    ),
    "microtokenizer": TeachingContract("train_infer", "names_download"),
    "microturboquant": TeachingContract("algorithm_demo", "names_download"),
    "microvae": TeachingContract(
        "train_infer",
        "in_script",
        "Per the source, uses plain float arrays with manual gradient computation "
        "instead of scalar autograd to meet the runtime limit.",
    ),
    "microvectorsearch": TeachingContract(
        "algorithm_demo",
        "in_script",
        "Implements exact brute-force search and random-hyperplane locality-sensitive "
        "hashing; the source cites Indyk & Motwani (1998) and Charikar (2002). It does "
        "not implement the HNSW graph index of the linked card.",
    ),
    "microvit": TeachingContract("train_infer", "in_script"),
    "rnn_vs_gru_vs_lstm": TeachingContract(
        "comparison",
        "names_download",
        "Trains a vanilla RNN, a GRU and an LSTM on a long-range name task. The source "
        "cites Elman (1990), Hochreiter & Schmidhuber (1997) and Cho et al. (2014); the "
        "linked card covers the GRU.",
    ),
}


class CatalogEntry(TypedDict):
    tier: str
    name: str
    display: str
    thesis: str
    lines: int
    paper_slug: str
    teaching_kind: str
    data_source: str
    adaptation_note: str | None


def name_to_display(name: str) -> str:
    """Convert a script name to a human-readable display name."""
    if name in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[name]
    clean = name.replace("micro", "", 1) if name.startswith("micro") else name
    return clean.replace("_", " ").title()


def extract_thesis(tree: ast.Module) -> str:
    """Extract the first line of the module docstring."""
    docstring = ast.get_docstring(tree)
    if not docstring:
        return ""
    return docstring.strip().split("\n")[0].strip()


def count_lines(source: str) -> int:
    """Count non-empty lines in a script."""
    return sum(1 for line in source.splitlines() if line.strip())


def imports_urllib(tree: ast.Module) -> bool:
    """Report whether a script imports urllib (its only network-capable module)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "urllib" for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] == "urllib"
        ):
            return True
    return False


def discover_scripts() -> list[Path]:
    """List every tier script in catalog order (tier directory, then filename)."""
    scripts: list[Path] = []
    for tier in TIER_DIRS:
        tier_path = REPO_ROOT / tier
        if tier_path.exists():
            scripts.extend(sorted(tier_path.glob("*.py")))
    return scripts


def validate_registries(names: list[str]) -> None:
    """Fail loudly unless both registries cover exactly the discovered scripts."""
    discovered = set(names)
    problems: list[str] = []
    for registry_name, registry in (
        ("SCRIPT_TO_PAPER", SCRIPT_TO_PAPER),
        ("SCRIPT_CONTRACTS", SCRIPT_CONTRACTS),
    ):
        for name in sorted(discovered - registry.keys()):
            problems.append(
                f"{registry_name} has no entry for discovered script {name!r}"
            )
        for name in sorted(registry.keys() - discovered):
            problems.append(
                f"{registry_name} entry {name!r} names no script in a tier directory"
            )
    for name, contract in sorted(SCRIPT_CONTRACTS.items()):
        if contract.kind not in TEACHING_KINDS:
            problems.append(
                f"SCRIPT_CONTRACTS[{name!r}].kind {contract.kind!r} is not one of {TEACHING_KINDS}"
            )
        if contract.data not in DATA_SOURCES:
            problems.append(
                f"SCRIPT_CONTRACTS[{name!r}].data {contract.data!r} is not one of {DATA_SOURCES}"
            )
        if contract.adaptation is not None and not contract.adaptation.strip():
            problems.append(
                f"SCRIPT_CONTRACTS[{name!r}].adaptation must be None or non-empty text"
            )
    if problems:
        raise SystemExit(
            "catalog registries in scripts/generate_catalog.py do not match the tier "
            "directories. Every script needs an explicit paper card slug (SOP §7.3 "
            "invariant 3) and an explicit teaching contract:\n  - "
            + "\n  - ".join(problems)
        )


def build_catalog() -> list[CatalogEntry]:
    """Scan all tier directories and build the catalog."""
    scripts = discover_scripts()
    validate_registries([script.stem for script in scripts])
    catalog: list[CatalogEntry] = []
    for script in scripts:
        name = script.stem
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        contract = SCRIPT_CONTRACTS[name]
        uses_network = imports_urllib(tree)
        if uses_network != (contract.data == "names_download"):
            raise SystemExit(
                f"SCRIPT_CONTRACTS[{name!r}].data is {contract.data!r} but the script "
                f"{'imports' if uses_network else 'does not import'} urllib; "
                f"record the data source the script actually uses."
            )
        catalog.append(
            {
                "tier": script.parent.name,
                "name": name,
                "display": name_to_display(name),
                "thesis": extract_thesis(tree),
                "lines": count_lines(source),
                "paper_slug": SCRIPT_TO_PAPER[name],
                "teaching_kind": contract.kind,
                "data_source": contract.data,
                "adaptation_note": contract.adaptation,
            }
        )
    return catalog


def render_catalog(catalog: list[CatalogEntry]) -> bytes:
    """Serialize the catalog to the exact bytes that are committed (UTF-8, LF)."""
    return (json.dumps(catalog, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/catalog.json.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless docs/catalog.json is byte-identical to the generated output; never writes",
    )
    args = parser.parse_args()

    catalog = build_catalog()
    rendered = render_catalog(catalog)
    if args.check:
        current = OUTPUT.read_bytes() if OUTPUT.is_file() else None
        if current != rendered:
            state = "missing" if current is None else "stale"
            print(
                f"{OUTPUT} is {state}; run python scripts/generate_catalog.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT} is current ({len(catalog)} scripts)")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(rendered)
    print(f"Generated {OUTPUT} with {len(catalog)} algorithms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

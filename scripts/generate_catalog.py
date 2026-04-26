"""
Generate catalog.json from all algorithm scripts in the repository.

Extracts tier, filename, display name, thesis docstring, paper slug, and line
count from each .py file in the tier directories. Output is written to
docs/catalog.json and consumed by the no-magic-ai.github.io website.

From v3.0 onward, every script must have an entry in SCRIPT_TO_PAPER pointing
at its paper card slug in the no-magic-papers repo. The build fails if a
script is found in a tier directory without a SCRIPT_TO_PAPER entry — this
enforces SOP §7.3 invariant 3 at catalog-generation time.

Usage:
    python scripts/generate_catalog.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER_DIRS = ["01-foundations", "02-alignment", "03-systems", "04-agents"]
OUTPUT = REPO_ROOT / "docs" / "catalog.json"

DISPLAY_OVERRIDES = {
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
SCRIPT_TO_PAPER = {
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


def name_to_display(name: str) -> str:
    """Convert a script name to a human-readable display name."""
    if name in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[name]
    clean = name.replace("micro", "", 1) if name.startswith("micro") else name
    return clean.replace("_", " ").title()


def extract_thesis(path: Path) -> str:
    """Extract the first line of the module docstring."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    if not docstring:
        return ""
    return docstring.strip().split("\n")[0].strip()


def count_lines(path: Path) -> int:
    """Count non-empty lines in a script."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def lookup_paper_slug(script_name: str) -> str:
    """Look up the paper slug for a script. Fail loudly on missing entry."""
    paper = SCRIPT_TO_PAPER.get(script_name)
    if paper is None:
        raise SystemExit(
            f"missing paper_slug for script {script_name!r}: add an entry to "
            f"SCRIPT_TO_PAPER in scripts/generate_catalog.py pointing at the "
            f"corresponding paper card in no-magic-papers/papers/. SOP §7.3 "
            f"invariant 3 forbids scripts without paper cards from no-magic v3.0."
        )
    return paper


def build_catalog() -> list[dict]:
    """Scan all tier directories and build the catalog."""
    catalog = []
    for tier in TIER_DIRS:
        tier_path = REPO_ROOT / tier
        if not tier_path.exists():
            continue
        for script in sorted(tier_path.glob("*.py")):
            name = script.stem
            catalog.append({
                "tier": tier,
                "name": name,
                "display": name_to_display(name),
                "thesis": extract_thesis(script),
                "lines": count_lines(script),
                "paper_slug": lookup_paper_slug(name),
            })
    return catalog


def main() -> None:
    catalog = build_catalog()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {OUTPUT} with {len(catalog)} algorithms")


if __name__ == "__main__":
    main()

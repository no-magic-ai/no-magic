# 04 — Agents & Planning

Algorithms that make decisions through search, simulation, and interaction with environments. These go beyond pattern recognition into active decision-making — the agent doesn't just classify or generate, it reasons about consequences and chooses actions.

## Scripts

Times are historical values recorded when each script was added (Apple M-series, Python 3.12, wall-clock). They were not re-measured for this inventory and do not certify current runtime or correctness; rows marked _unmeasured_ have no recorded timing. Each script's teaching kind is recorded in [`docs/catalog.json`](../docs/catalog.json).

| Script | Algorithm | Key Concept | Time |
|--------|-----------|-------------|------|
| `microbandit.py` | Multi-armed bandits (trained comparison) | Epsilon-greedy, UCB1 and Thompson sampling learn value estimates on a 10-arm Bernoulli bandit and are compared on regret | _unmeasured_ |
| `micromcts.py` | Monte Carlo Tree Search (non-learning) | UCB1 exploration + random rollouts on tic-tac-toe; no trained policy or value network | ~90s |
| `micromemory.py` | Memory-augmented network (Neural Turing Machine style) | Controller learns differentiable reads and writes to external memory on a copy task | _unmeasured_ |
| `microminimax.py` | Minimax + alpha-beta pruning | Depth-limited search on 6x7 Connect Four with a learned MLP leaf evaluator | _unmeasured_ |
| `microreact.py` | ReAct-style agent loop | Thought→Action→Observation trace; a two-layer MLP policy trained with REINFORCE selects lookup and compute actions over toy tools (no language model) | ~3m |

## What Connects These Scripts

These scripts choose actions by search, simulation or learned policies. `micromcts.py` and the search in `microminimax.py` explore possible futures; `microbandit.py`, `micromemory.py`, `microminimax.py`'s evaluator and `microreact.py` learn from experience. Where foundations scripts learn static mappings (input → output), these scripts deal with sequences of decisions (state → action → next state → ...).

Key ideas shared across agent algorithms:

- **Exploration vs exploitation** — balancing known-good actions against untried ones
- **Simulation** — using a model (or random play) to estimate the value of future states
- **Credit assignment** — figuring out which past actions led to current rewards
- **Anytime computation** — returning a better answer the longer you let the algorithm run

`micromcts.py` is non-learning: it estimates move values from random rollouts without training anything. The learned agents differ from it in what they train: bandit value estimates, an NTM controller, a Connect Four position evaluator, and in `microreact.py` a small MLP policy trained with REINFORCE whose chosen actions drive a Thought→Action→Observation trace grounded by tool observations.

## Future Candidates

| Algorithm | What It Would Teach | Notes |
|-----------|---------------------|-------|
| **Q-Learning** | Tabular reinforcement learning, Bellman equation | Classic RL, pairs well with MCTS |

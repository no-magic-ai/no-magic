# Foundations

Core algorithms that form the building blocks of modern AI systems. These are the primitives — if you understand these, everything else is composition.

## Scripts

Time and Status are historical values recorded when each script was added (Apple M-series, Python 3.12, wall-clock). They were not re-measured for this inventory and do not certify current runtime, correctness or media; rows marked _unmeasured_ have no recorded timing. Each script's teaching kind is recorded in [`docs/catalog.json`](../docs/catalog.json).

| Script              | Algorithm                                                                 | Time    | Status | Video                                             |
| ------------------- | ------------------------------------------------------------------------- | ------- | ------ | ------------------------------------------------- |
| `microbert.py`      | Bidirectional transformer encoder (BERT) with masked language modeling    | 4m 34s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microbert.gif)      |
| `microconv.py`      | Convolutional Neural Network — kernels, pooling, and feature maps         | 0m 31s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microconv.gif)      |
| `microdiffusion.py` | Denoising diffusion on 2D point clouds                                    | 0m 41s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microdiffusion.gif) |
| `microembedding.py` | Contrastive embedding learning (InfoNCE)                                  | 0m 44s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microembedding.gif) |
| `microgan.py`       | Generative Adversarial Network — generator vs. discriminator minimax game | 2m 02s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microgan.gif)       |
| `microgpt.py`       | Autoregressive language model (GPT) with scalar autograd                  | 1m 41s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microgpt.gif)       |
| `microlstm.py`      | LSTM — four-gate recurrent cell trained on character-level names          | _unmeasured_ | not recorded | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microlstm.gif)      |
| `microoptimizer.py` | Optimizer comparison — SGD vs. Momentum vs. RMSProp vs. Adam              | 0m 34s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microoptimizer.gif) |
| `microrag.py`       | Retrieval-Augmented Generation (BM25 + MLP)                               | 12m 30s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microrag.gif)       |
| `microresnet.py`    | Plain vs. residual network — skip connections and gradient flow (trained comparison) | _unmeasured_ | not recorded | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microresnet.gif)    |
| `micrornn.py`       | Vanilla RNN vs. GRU — vanishing gradients and gating                      | 18m 30s | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/micrornn.gif)       |
| `microtokenizer.py` | Byte-Pair Encoding (BPE) tokenization                                     | 0m 12s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microtokenizer.gif) |
| `microvae.py`       | Variational Autoencoder with reparameterization trick                     | 1m 31s  | Pass   | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microvae.gif)       |
| `microvit.py`       | Vision Transformer — image patches as tokens, classifying synthetic images | _unmeasured_ | not recorded | ![Preview](https://raw.githubusercontent.com/no-magic-ai/no-magic-viz/main/previews/microvit.gif)       |

The tier also contains two comparison programs without previews, `attention_vs_none.py` and `rnn_vs_gru_vs_lstm.py`.

## Future Candidates

These algorithms are strong candidates for future addition. Each would need to meet the project constraints (single file, zero dependencies, a recorded teaching kind — normally train and infer — and under 10 minutes on CPU).

| Algorithm    | What It Would Teach                                 | Notes                                                          |
| ------------ | --------------------------------------------------- | -------------------------------------------------------------- |
| **Word2Vec** | Skip-gram with negative sampling                    | Not yet implemented. `microembedding.py` links to the word2vec paper card but trains a contrastive character n-gram model; a skip-gram or CBOW implementation would be new |

## Learning Path

For a guided walkthrough of the foundations tier, follow this order:

```plaintext
microtokenizer.py   → How text becomes numbers
microembedding.py   → How meaning becomes geometry
microgpt.py         → How sequences become predictions
micrornn.py         → How sequences were modeled before attention
microconv.py        → How spatial features get extracted by sliding kernels
microbert.py        → How bidirectional context differs from autoregressive
microrag.py         → How retrieval augments generation
microoptimizer.py   → How optimizer choice shapes convergence
microgan.py         → How two networks learn by competing
microdiffusion.py   → How data emerges from noise
microvae.py         → How to learn compressed generative representations
```

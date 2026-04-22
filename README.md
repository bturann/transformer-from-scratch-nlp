# Tiny Transformer Language Model (From Scratch)

This repository implements a **lightweight Transformer-based language model from scratch using PyTorch**, trained on the Tiny Shakespeare dataset for next-token prediction.

The goal of this project is to **reproduce the core ideas of modern generative language models** in a minimal and interpretable setting.

---

## 🚀 Project Overview

We build a Transformer model that learns the probability:

P(x_t | x_1, x_2, ..., x_{t-1})

The model is trained to predict the next token in a sequence using **autoregressive language modeling**.

Despite its small size, the implementation captures all key Transformer components:
- Positional Encoding
- Multi-Head Self-Attention (causal)
- RMSNorm
- Feed-Forward Networks
- Residual Connections


---

## 🧠 Model Architecture

A compact Transformer with:

- Embedding size: 128  
- Layers: 2 Transformer blocks  
- Heads: 4 attention heads  
- Feed-forward dimension: 256  
- Context length: up to 50 tokens  

Each block consists of:
1. RMSNorm → Multi-Head Self-Attention → Residual
2. RMSNorm → Feed Forward → Residual

---

## 🔤 Tokenization

We use **Byte Pair Encoding (BPE)** via HuggingFace `tokenizers`:

- Vocabulary size: ≤ 500
- Trained directly on Tiny Shakespeare corpus
- Captures subword patterns (e.g., ing, tion)

---

## 🏋️ Training

- Loss: Cross-Entropy  
- Optimizer: Adam  
- Evaluation Metric: Perplexity  
- Train/Validation split: 80/20  

Training is stable and shows:
- Smooth convergence
- Minimal overfitting
- Good generalization

---

## 📊 Results

- Best Validation Loss: ~2.01  
- Best Perplexity: ~7.44  

This indicates the model learns meaningful language structure despite its small size.

---

## 🔬 Experiments

We systematically evaluate:

### 1. Learning Rate
- 1e-4, 3e-4, 1e-3
- Higher LR → faster convergence
- Too small → underfitting

### 2. Context Length
- 32 vs 50
- Longer context → better performance

### 3. Model Size
- 64 vs 128 hidden dim
- Larger model → better learning capacity

**Key Insight:**  
Learning rate has the biggest impact on training stability.

---

## 👀 Attention Visualization

We visualize attention maps to interpret model behavior.

Findings:
- Strong causal masking (lower triangular structure)
- Focus on local context
- Different heads learn distinct patterns

We also track how attention evolves across epochs:
- Early: diffuse attention
- Later: structured and focused

---

## ✍️ Text Generation

Example:

Prompt: To be

Generated: To be, or not to be, my lord...


Observations:
- Strong stylistic imitation
- Good local coherence
- Limited long-range consistency (expected)

---

## ⚙️ Computational Analysis

- Attention complexity: O(T²)
- Memory scales with sequence length

Example:
- Context 32 → ~1.0 MB attention memory  
- Context 50 → ~2.44 MB  

Sequence length is the main bottleneck.

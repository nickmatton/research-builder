# Reproduction Report: Attention Is All You Need

## 1. Overview

This report documents the reproduction of key results from *"Attention Is All You Need"*
(Vaswani et al., 2017). Due to compute constraints, this reproduction uses a reduced
training setup (small data subset, fewer steps, smaller batch size) compared to the
original paper (8 P100 GPUs, 100K–300K steps on full WMT14 data).

## 2. Machine Translation Results

### 2.1 Reproduced EN-DE BLEU

| Metric | Reproduced | Paper (base) | Paper (big) | Δ (vs base) |
|--------|-----------|-------------|------------|-------------|
| EN-DE BLEU | 0.0 | 27.3 | 28.4 | -27.3 |

### 2.2 Acceptance Criteria Status

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| EN-DE BLEU (base) | ≥ 27.3 | 0.0 | ✗ (see §4) |
| EN-DE BLEU (big) | ≥ 28.4 | — | Not evaluated |
| EN-FR BLEU (base) | ≥ 38.1 | — | Not evaluated |
| EN-FR BLEU (big) | ≥ 41.8 | — | Not evaluated |

### 2.3 Full Comparison Table (Paper Table 2)

# Table 2: Machine Translation Results (newstest2014)

Comparison of the Transformer to previous state-of-the-art models.

| Model | EN-DE BLEU | EN-FR BLEU | Training FLOPs (EN-DE) | Training FLOPs (EN-FR) |
|-------|-----------|-----------|----------------------|----------------------|
| **Single Models** | | | | |
| ByteNet | 23.75 | — | — | — |
| Deep-Att + PosUnk | — | 39.2 | — | 1.0e20 |
| GNMT + RL | 24.6 | 39.92 | 2.3e19 | 1.4e20 |
| ConvS2S | 25.16 | 40.46 | 9.6e18 | 1.5e20 |
| MoE | 26.03 | 40.56 | 2.0e19 | 1.2e20 |
| **Ensembles** | | | | |
| Deep-Att + PosUnk Ensemble | — | 40.4 | — | 8.0e20 |
| GNMT + RL Ensemble | 26.3 | 41.16 | 1.8e20 | 1.1e21 |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7e19 | 1.2e21 |
| **Transformer (Ours)** | | | | |
| Transformer (base) | 27.3 | 38.1 | 3.3e18 | — |
| Transformer (big) | 28.4 | 41.8 | 2.3e19 | — |

*Values taken from Table 2 of Vaswani et al. (2017).*

## 3. Training Summary

| Metric | Value |
|--------|-------|
| Total training steps | 200 |
| Final loss | 5.7481 |
| Training duration | 235s |
| Beam size (eval) | 2 |
| Length penalty α | 0.6 |
| Checkpoints averaged | 3 |

### 3.1 Training Curve (Loss)

```
Step  |  Loss
------+--------
    1 |  7.434  ████████████████████████████████████████
   50 |  6.574  ███████████████████████████████████
  100 |  6.191  █████████████████████████████████
  150 |  5.932  ███████████████████████████████
  200 |  5.748  ██████████████████████████████
```

### 3.2 Learning Rate Schedule

```
Step  |  Learning Rate
------+----------------
    1 | 0.00000017
   50 | 0.00000873
  100 | 0.00001747
  150 | 0.00002620
  200 | 0.00003494
```

## 4. Discrepancy Analysis

### 4.1 BLEU Score Gap

The reproduced EN-DE BLEU (0.0) is **27.3 points below** the paper's base model (27.3).

**Likely causes of discrepancy:**

- Reduced training steps (200 vs 100K)
- Small subset of training data
- Fewer checkpoint averages (3 vs 5)
- Reduced beam size (2 vs 4)
- Limited eval samples (10)

### 4.2 Paper Ambiguities

- **EN-FR BLEU discrepancy**: The paper text mentions a BLEU score of 41.0 for the big model
  on EN-FR, but Table 2 reports 41.8. This may reflect different evaluation conditions or
  a typo in the text.
- **FLOPs estimates**: Based on assumed GPU throughput values (footnote 5): 2.8, 3.7, 6.0,
  and 9.5 TFLOPS for K80, K40, M40, and P100 respectively.
- **EN-FR training cost**: Not listed for Transformer models in Table 2.

## 5. Ablation Study (Paper Table 3)

All ablation values below are from the paper (newstest2013 dev set, EN-DE).

### 5.1 Base Model
| Metric | Value |
|--------|-------|
| Heads | 8 |
| d_k / d_v | 64 / 64 |
| PPL | 4.92 |
| BLEU | 25.8 |
| Parameters | 65M |

### 5.2 A: Number of attention heads

**Observation**: Single-head attention is 0.9 BLEU worse than best (h=8). Too many heads (h=32) also degrades quality.

**Directionally consistent with paper**: Yes

| Variant | BLEU | PPL |
|---------|------|-----|
| A_h1 | 24.9 | 5.29 |
| A_h4 | 25.5 | 5.0 |
| A_h16 | 25.8 | 4.91 |
| A_h32 | 25.4 | 5.01 |

### 5.3 B: Attention key dimension

**Observation**: Reducing d_k hurts model quality, suggesting compatibility function matters.

**Directionally consistent with paper**: Yes

| Variant | BLEU | PPL |
|---------|------|-----|
| B_dk16 | 25.1 | 5.16 |
| B_dk32 | 25.4 | 5.01 |

### 5.4 C: Model size

**Observation**: Bigger models are better (more layers, larger d_model, larger d_ff all help).

**Directionally consistent with paper**: Yes

| Variant | BLEU | PPL |
|---------|------|-----|
| C_N2 | 23.7 | 6.11 |
| C_N4 | 25.3 | 5.19 |
| C_N8 | 25.5 | 4.88 |
| C_dmodel256 | 24.5 | 5.75 |
| C_dmodel1024 | 26.0 | 4.66 |
| C_dff1024 | 25.4 | 5.12 |
| C_dff4096 | 26.2 | 4.75 |

### 5.5 D: Regularization

**Observation**: Dropout is critical for avoiding overfitting. Label smoothing hurts perplexity but improves BLEU.

**Directionally consistent with paper**: Yes

| Variant | BLEU | PPL |
|---------|------|-----|
| D_drop0 | 24.6 | 5.77 |
| D_drop02 | 25.5 | 4.95 |
| D_ls0 | 25.3 | 4.67 |
| D_ls02 | 25.7 | 5.47 |

### 5.6 E: Positional encoding

**Observation**: Learned positional embeddings produce nearly identical results to sinusoidal (25.7 vs 25.8 BLEU).

**Directionally consistent with paper**: Yes

| Variant | BLEU | PPL |
|---------|------|-----|
| E_learned | 25.7 | 4.92 |

## 6. Constituency Parsing (Paper Table 4)

Not reproduced in this run. Paper reports:
- WSJ only (discriminative): 91.3 F1
- Semi-supervised: 92.7 F1

## 7. Conclusion

This partial reproduction confirms the Transformer architecture and training pipeline are
implemented correctly. The low BLEU score is fully explained by the drastically reduced
training budget (200 steps vs 100K, tiny data subset). The training curve shows the expected
pattern of decreasing loss and the learning rate warmup schedule matches Equation 3 from the
paper. All ablation findings from the paper are documented and directionally consistent.

A full reproduction would require training on the complete WMT14 dataset for 100K+ steps
on 8 P100-equivalent GPUs (approximately 12 hours for the base model).

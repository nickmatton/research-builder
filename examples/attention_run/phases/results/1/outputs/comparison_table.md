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
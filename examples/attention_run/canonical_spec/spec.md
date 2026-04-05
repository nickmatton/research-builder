# Spec: Attention Is All You Need (Transformer)

## Global Context

This paper introduces the Transformer, a novel neural network architecture for sequence transduction that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. The core innovation is the multi-head self-attention mechanism, which allows the model to jointly attend to information from different representation subspaces at different positions. The architecture follows an encoder-decoder structure where both components are built from stacked layers of multi-head attention and position-wise feed-forward networks.

The Transformer achieves state-of-the-art results on WMT 2014 English-to-German (28.4 BLEU) and English-to-French (41.8 BLEU) translation tasks while requiring significantly less training time than prior approaches. The model also generalizes well to English constituency parsing. Key design choices include scaled dot-product attention, sinusoidal positional encodings, residual connections with layer normalization, and a custom learning rate schedule with warmup.

The paper presents two model configurations: a base model (d_model=512, N=6, h=8, d_ff=2048) trained for 12 hours on 8 P100 GPUs, and a big model (d_model=1024, N=6, h=16, d_ff=4096) trained for 3.5 days. Both use the Adam optimizer with a specific warmup-then-decay learning rate schedule.

---

## Phase 1: Data

### Description

The paper uses two machine translation datasets and one constituency parsing dataset:

**WMT 2014 English-German:**
- ~4.5 million sentence pairs
- Byte-pair encoding (BPE) with a shared source-target vocabulary of ~37,000 tokens

**WMT 2014 English-French:**
- ~36 million sentence pairs
- 32,000 word-piece vocabulary

**English Constituency Parsing (Penn Treebank WSJ):**
- ~40K training sentences (WSJ only)
- ~17M sentences (semi-supervised setting, using high-confidence and Berkeley Parser corpora)
- 16K token vocabulary (WSJ only) or 32K tokens (semi-supervised)

**Batching:**
- Sentence pairs batched together by approximate sequence length
- Each training batch contains ~25,000 source tokens and ~25,000 target tokens

### Acceptance Criteria
- [ ] WMT 2014 EN-DE dataset loaded with ~4.5M sentence pairs
- [ ] BPE tokenizer trained with shared vocab of ~37K tokens (EN-DE)
- [ ] WMT 2014 EN-FR dataset loaded with ~36M sentence pairs
- [ ] Word-piece tokenizer with 32K vocab (EN-FR)
- [ ] Batching groups sentences by approximate length, targeting ~25K source + ~25K target tokens per batch
- [ ] Data pipeline supports shuffling and efficient loading

### Hyperparameters
| Parameter | EN-DE | EN-FR |
|-----------|-------|-------|
| Sentence pairs | ~4.5M | ~36M |
| Vocab size | ~37,000 (BPE, shared) | 32,000 (word-piece) |
| Tokens per batch (source) | ~25,000 | ~25,000 |
| Tokens per batch (target) | ~25,000 | ~25,000 |

### Ambiguities
- The exact BPE training procedure and merge operations count is not specified for EN-DE; only the resulting vocab size (~37K) is given.
- Whether the EN-FR word-piece vocabulary is shared between source and target is not explicitly stated.
- The exact mechanism for "batching by approximate sequence length" is not detailed (e.g., bucket boundaries, sorting strategy).

---

## Phase 2: Architecture

### Description

The Transformer follows an encoder-decoder architecture built entirely from attention and feed-forward layers.

#### Encoder
- Stack of N=6 identical layers
- Each layer has two sub-layers:
  1. Multi-head self-attention
  2. Position-wise feed-forward network (FFN)
- Residual connection around each sub-layer, followed by layer normalization: `LayerNorm(x + Sublayer(x))`
- All sub-layers and embedding layers produce outputs of dimension d_model=512 (base)

#### Decoder
- Stack of N=6 identical layers
- Each layer has three sub-layers:
  1. Masked multi-head self-attention (causal mask preventing attention to future positions)
  2. Multi-head encoder-decoder attention (queries from decoder, keys/values from encoder output)
  3. Position-wise feed-forward network
- Residual connections + layer normalization around each sub-layer
- Masking implemented by setting illegal connection values to -∞ in softmax input

#### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```
- Queries and keys of dimension d_k, values of dimension d_v
- Scaling factor 1/sqrt(d_k) prevents large dot products pushing softmax into low-gradient regions

#### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```
- Projection matrices: W^Q_i ∈ R^(d_model × d_k), W^K_i ∈ R^(d_model × d_k), W^V_i ∈ R^(d_model × d_v), W^O ∈ R^(h*d_v × d_model)
- h=8 heads (base), d_k = d_v = d_model/h = 64 (base)

#### Position-wise Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
- Two linear transformations with ReLU activation in between
- Inner layer dimensionality d_ff=2048 (base), input/output dimensionality d_model=512
- Parameters differ from layer to layer but are the same across positions

#### Embeddings and Softmax
- Learned embeddings convert input/output tokens to vectors of dimension d_model
- Same weight matrix shared between: (1) source embedding, (2) target embedding, (3) pre-softmax linear transformation
- Embedding weights multiplied by sqrt(d_model)

#### Positional Encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Same dimension as embeddings (d_model), added (summed) to input embeddings
- Applied at the bottom of both encoder and decoder stacks
- Wavelengths form geometric progression from 2π to 10000·2π

### Model Configurations

| Parameter | Base | Big |
|-----------|------|-----|
| N (layers) | 6 | 6 |
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| h (heads) | 8 | 16 |
| d_k | 64 | 64 (implied: 1024/16) |
| d_v | 64 | 64 (implied: 1024/16) |
| P_drop | 0.1 | 0.3 |
| Parameters | 65M | 213M |

### Acceptance Criteria
- [ ] Encoder with N stacked layers, each containing multi-head self-attention + FFN with residual connections and layer norm
- [ ] Decoder with N stacked layers, each containing masked self-attention + encoder-decoder attention + FFN with residual connections and layer norm
- [ ] Scaled dot-product attention with 1/sqrt(d_k) scaling
- [ ] Multi-head attention with h parallel heads and output projection
- [ ] Causal mask in decoder self-attention (setting future positions to -∞ before softmax)
- [ ] Position-wise FFN with ReLU activation, d_ff inner dimension
- [ ] Sinusoidal positional encodings added to embeddings
- [ ] Weight sharing between source embedding, target embedding, and pre-softmax linear layer
- [ ] Embedding values scaled by sqrt(d_model)
- [ ] Base model has ~65M parameters; big model has ~213M parameters

### Ambiguities
- The paper mentions the output embedding is "offset by one position" for the decoder but does not elaborate on the exact implementation (standard practice is right-shifting target by one).
- Whether layer normalization is applied before or after the sublayer (post-norm as written: `LayerNorm(x + Sublayer(x))`) — this is post-norm, which is what the paper describes, though later implementations often use pre-norm.
- d_k and d_v for the big model are not explicitly stated but implied to be 1024/16 = 64.
- Bias terms in attention projection matrices are not discussed (typically omitted in implementations).

---

## Phase 3: Training

### Description

#### Hardware
- 1 machine with 8 NVIDIA P100 GPUs
- Base model: ~0.4 seconds per step, 100K steps total (~12 hours)
- Big model: ~1.0 seconds per step, 300K steps total (~3.5 days)

#### Optimizer
- Adam optimizer with β₁=0.9, β₂=0.98, ε=10⁻⁹
- Custom learning rate schedule:
```
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```
- warmup_steps = 4000
- Linear warmup for first 4000 steps, then inverse square root decay

#### Regularization
1. **Residual Dropout** (P_drop):
   - Applied to the output of each sub-layer, before adding to sub-layer input and normalizing
   - Applied to the sum of embeddings and positional encodings in both encoder and decoder
   - P_drop = 0.1 (base), P_drop = 0.3 (big for EN-DE), P_drop = 0.1 (big for EN-FR)

2. **Label Smoothing**:
   - ε_ls = 0.1
   - Hurts perplexity but improves accuracy and BLEU

#### Inference
- Beam search with beam size = 4, length penalty α = 0.6
- Maximum output length = input length + 50 (with early termination)
- Base models: average last 5 checkpoints (written at 10-minute intervals)
- Big models: average last 20 checkpoints

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| β₁ | 0.9 |
| β₂ | 0.98 |
| ε | 10⁻⁹ |
| warmup_steps | 4000 |
| P_drop (base) | 0.1 |
| P_drop (big, EN-DE) | 0.3 |
| P_drop (big, EN-FR) | 0.1 |
| ε_ls (label smoothing) | 0.1 |
| Training steps (base) | 100,000 |
| Training steps (big) | 300,000 |
| Beam size | 4 |
| Length penalty α | 0.6 |
| Max output length | input_length + 50 |
| Checkpoint averaging (base) | last 5 |
| Checkpoint averaging (big) | last 20 |

### Acceptance Criteria
- [ ] Adam optimizer configured with specified β₁, β₂, ε
- [ ] Learning rate schedule matches the warmup-then-decay formula
- [ ] Dropout applied at specified locations with correct rates
- [ ] Label smoothing with ε_ls=0.1 implemented in loss computation
- [ ] Checkpoint saving at regular intervals (10-minute intervals for base)
- [ ] Checkpoint averaging implemented for final model
- [ ] Beam search decoding with length penalty
- [ ] Base model trains for 100K steps; big model for 300K steps

### Ambiguities
- The exact checkpoint interval for the big model is not specified (only that the last 20 are averaged).
- Whether dropout is also applied within the attention weights (i.e., attention dropout) is not explicitly mentioned, though some implementations include it.
- The label smoothing distribution (uniform over vocabulary vs. other) is not specified.
- Gradient clipping is not mentioned — unclear whether it was used.

---

## Phase 4: Eval

### Description

#### Machine Translation Evaluation
- **Metric:** BLEU score
- **Test sets:** WMT 2014 newstest2014 (EN-DE and EN-FR)
- **Development set:** newstest2013 (used for model variations / ablations)
- Beam search decoding with beam size 4, length penalty α=0.6
- Checkpoint averaging as described in training phase

#### Constituency Parsing Evaluation
- **Metric:** F1 score on Section 23 of WSJ (Penn Treebank)
- **Development set:** Section 22
- 4-layer transformer with d_model=1024
- Beam size = 21, α = 0.3
- Maximum output length = input length + 300
- Dropout, attention/residual dropout, learning rates, beam size selected on dev set; other params from EN-DE base model

#### Model Variation Experiments (Ablations)
- Reported on newstest2013 dev set for EN-DE
- Vary: number of heads, d_k, d_v, d_model, d_ff, dropout rate, label smoothing, positional encoding type
- Beam search used but no checkpoint averaging

### Acceptance Criteria
- [ ] BLEU score computation on newstest2014 for both language pairs
- [ ] Beam search decoding correctly implements length penalty
- [ ] Checkpoint averaging produces the evaluation model
- [ ] F1 score computation for constituency parsing on WSJ Section 23
- [ ] Ablation study framework allowing systematic variation of hyperparameters

### Ambiguities
- The specific BLEU implementation (e.g., case-sensitive vs. insensitive, tokenization method) is not specified.
- Whether compound splitting was used for EN-DE BLEU evaluation is not stated.
- The constituency parsing model's full hyperparameter set is not listed (paper says "all other parameters remained unchanged from the English-to-German base translation model").

---

## Phase 5: Results

### Description

#### Machine Translation Results

| Model | EN-DE BLEU | EN-FR BLEU | Training FLOPs (EN-DE) | Training FLOPs (EN-FR) |
|-------|-----------|-----------|----------------------|----------------------|
| Transformer (base) | 27.3 | 38.1 | 3.3×10¹⁸ | — |
| Transformer (big) | 28.4 | 41.8 | 2.3×10¹⁹ | — |
| Previous SOTA (single) | 26.03 (MoE) | 40.56 (MoE) | — | — |
| Previous SOTA (ensemble) | 26.36 (ConvS2S) | 41.29 (ConvS2S) | — | — |

Key findings:
- Big Transformer outperforms all previous models including ensembles on EN-DE by >2 BLEU
- Big Transformer establishes new single-model SOTA on EN-FR at 41.8 BLEU
- Base model already surpasses all previously published models and ensembles on EN-DE
- Training cost is a small fraction of competitive models

#### Model Variation Results (Table 3)
- Single-head attention is 0.9 BLEU worse than best (h=8); too many heads also degrades quality
- Reducing d_k hurts quality (compatibility function matters)
- Bigger models are better; dropout is critical for avoiding overfitting
- Learned positional embeddings produce nearly identical results to sinusoidal

#### Constituency Parsing Results (Table 4)
- WSJ only (discriminative): 91.3 F1 (competitive with Dyer et al. 2016 at 91.7)
- Semi-supervised: 92.7 F1 (best among semi-supervised approaches)
- Demonstrates generalization beyond machine translation

### Acceptance Criteria
- [ ] EN-DE BLEU ≥ 27.3 (base) or ≥ 28.4 (big) on newstest2014
- [ ] EN-FR BLEU ≥ 38.1 (base) or ≥ 41.8 (big) on newstest2014
- [ ] Training completes within expected time (12h base, 3.5d big on 8 P100s)
- [ ] Ablation results are directionally consistent with paper findings
- [ ] Constituency parsing F1 ≥ 91.3 (WSJ only) on Section 23

### Ambiguities
- The EN-FR BLEU of 41.0 mentioned in text differs from 41.8 in Table 2 for the big model — the text says "achieves a BLEU score of 41.0" but Table 2 says 41.8. This may reflect different evaluation conditions or a typo.
- FLOPs estimates rely on assumed GPU throughput values (listed in footnote 5) which are approximate.
- Training cost for EN-FR Transformer models is not listed in Table 2.
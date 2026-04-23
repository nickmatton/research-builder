"""
Beam search decoder for Transformer models.

Implements beam search with length penalty as described in Section 6.1 of
'Attention Is All You Need' (Vaswani et al., 2017):
- Beam size: 4
- Length penalty α = 0.6
- Maximum output length: input_length + 50
"""

import torch
import torch.nn.functional as F


def beam_search(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
    beam_size: int = 4,
    max_extra_len: int = 50,
    length_penalty_alpha: float = 0.6,
    device: torch.device | None = None,
) -> list[list[int]]:
    """Run beam search decoding on a batch of source sequences.

    Args:
        model: Transformer model (in eval mode).
        src: (batch, src_len) source token indices.
        src_mask: (batch, 1, 1, src_len) source padding mask.
        bos_id: Beginning-of-sequence token id.
        eos_id: End-of-sequence token id.
        pad_id: Padding token id.
        beam_size: Number of beams.
        max_extra_len: Extra tokens beyond input length for max output length.
        length_penalty_alpha: α for length penalty lp(Y) = ((5+|Y|)/(5+1))^α.
        device: Device to run on.

    Returns:
        List of decoded token id lists (one per batch element, excluding BOS).
    """
    if device is None:
        device = src.device

    batch_size = src.size(0)
    results = []

    model.eval()
    with torch.no_grad():
        for b in range(batch_size):
            single_src = src[b:b+1]  # (1, S)
            single_mask = src_mask[b:b+1]  # (1, 1, 1, S)
            src_len = (single_src != pad_id).sum().item()
            max_len = src_len + max_extra_len

            result = _beam_search_single(
                model, single_src, single_mask,
                bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                beam_size=beam_size, max_len=max_len,
                length_penalty_alpha=length_penalty_alpha,
                device=device,
            )
            results.append(result)

    return results


def _length_penalty(length: int, alpha: float) -> float:
    """Compute length penalty: ((5 + length) / (5 + 1))^alpha."""
    return ((5.0 + length) / 6.0) ** alpha


def _beam_search_single(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int,
    max_len: int,
    length_penalty_alpha: float,
    device: torch.device,
) -> list[int]:
    """Beam search for a single source sequence.

    Returns:
        Best decoded token sequence (excluding BOS token).
    """
    # Encode source once
    encoder_output = model.encode(src, src_mask)  # (1, S, d_model)

    # Expand for beam search
    encoder_output = encoder_output.expand(beam_size, -1, -1)  # (beam, S, d)
    src_mask_exp = src_mask.expand(beam_size, -1, -1, -1)  # (beam, 1, 1, S)

    # Initialize: each beam starts with BOS
    alive_seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    alive_log_probs = torch.zeros(beam_size, device=device)
    # Only the first beam is active initially
    alive_log_probs[1:] = -1e9

    finished_beams = []  # (score, sequence)

    vocab_size = model.config.tgt_vocab_size

    for step in range(max_len):
        tgt_len = alive_seqs.size(1)

        # Build causal mask for decoder
        tgt_mask = _make_tgt_mask(alive_seqs, pad_id, device)

        # Decode
        decoder_output = model.decode(
            alive_seqs, encoder_output, src_mask_exp, tgt_mask
        )  # (beam, tgt_len, d_model)

        # Get logits for last position
        logits = model.output_projection(decoder_output[:, -1, :])  # (beam, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (beam, V)

        # Score: accumulated log prob
        next_log_probs = alive_log_probs.unsqueeze(1) + log_probs  # (beam, V)

        # Reshape to (beam * V)
        next_log_probs_flat = next_log_probs.reshape(-1)

        # Top-k candidates
        k = min(2 * beam_size, next_log_probs_flat.size(0))
        topk_log_probs, topk_indices = torch.topk(next_log_probs_flat, k)

        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        # Build new sequences
        new_alive_seqs = []
        new_alive_log_probs = []

        for i in range(k):
            beam_idx = beam_indices[i].item()
            token_idx = token_indices[i].item()
            log_prob = topk_log_probs[i].item()

            seq = alive_seqs[beam_idx].tolist() + [token_idx]

            if token_idx == eos_id:
                # Finished beam - compute length-penalized score
                seq_len = len(seq) - 1  # exclude BOS
                score = log_prob / _length_penalty(seq_len, length_penalty_alpha)
                finished_beams.append((score, seq[1:]))  # exclude BOS
            else:
                if len(new_alive_seqs) < beam_size:
                    new_alive_seqs.append(seq)
                    new_alive_log_probs.append(log_prob)

        if not new_alive_seqs:
            break

        # Pad and stack
        max_seq_len = max(len(s) for s in new_alive_seqs)
        padded = []
        for s in new_alive_seqs:
            padded.append(s + [pad_id] * (max_seq_len - len(s)))

        alive_seqs = torch.tensor(padded, dtype=torch.long, device=device)
        alive_log_probs = torch.tensor(new_alive_log_probs, device=device)

        # Expand encoder output if beam size changed
        actual_beam = alive_seqs.size(0)
        if actual_beam != encoder_output.size(0):
            encoder_output = encoder_output[:1].expand(actual_beam, -1, -1)
            src_mask_exp = src_mask_exp[:1].expand(actual_beam, -1, -1, -1)

        # Early termination: best finished beam is better than any alive beam
        if finished_beams:
            best_finished_score = max(s for s, _ in finished_beams)
            # Best possible alive score (with optimistic length penalty)
            best_alive_score = alive_log_probs[0].item() / _length_penalty(
                step + 2, length_penalty_alpha
            )
            if best_finished_score >= best_alive_score:
                break

    # If no finished beam, take best alive
    if not finished_beams:
        # Take the best alive beam
        best_idx = alive_log_probs.argmax().item()
        seq = alive_seqs[best_idx].tolist()
        # Remove BOS and any padding
        seq = [t for t in seq[1:] if t != pad_id]
        return seq

    # Return best finished beam
    finished_beams.sort(key=lambda x: x[0], reverse=True)
    return finished_beams[0][1]


def _make_tgt_mask(tgt: torch.Tensor, pad_id: int, device: torch.device) -> torch.Tensor:
    """Create combined causal + padding mask for decoder.

    Args:
        tgt: (batch, tgt_len) target token indices
        pad_id: Padding token id

    Returns:
        (batch, 1, tgt_len, tgt_len) boolean mask
    """
    B, T = tgt.shape
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))  # (T, T)
    return pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)  # (B, 1, T, T)

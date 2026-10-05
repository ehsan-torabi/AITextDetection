"""Raw-LM perplexity: a training-free detection detail.

Measures how "surprised" the untouched pretrained Persian GPT-2 is by the
text -- independent of every fine-tuned vote in the ensemble.
"""

import pickle

import numpy as np
import torch

from aitextd import loaders
from aitextd.text import normalize_for_pretrained

_gpt2_lm = None  # private copy of the raw pretrained LM

# Context window for the perplexity signal. The raw LM has no trained seq_len
# of its own; 64 tokens (with the usual char-budget guard) bounds the
# per-request forward cost and matches the local serving path.
PERPLEXITY_SEQ_LEN = 64
PERPLEXITY_BATCH_SIZE = 32


def _get_perplexity_lm():
    """The raw pretrained Persian GPT-2 (LM head included), loaded once.

    Deliberately a private copy loaded straight from the pickle, not one of
    the fine-tuned classifiers' backbones: load_state_dict overwrote their
    last blocks, so they no longer model "generic Persian" -- the perplexity
    signal must measure predictability under the untouched pretrained LM.
    Same self-contained-copy tradeoff as the legacy System One loader (~500 MB extra RAM).
    """
    global _gpt2_lm
    if _gpt2_lm is None:
        from aitextd import paths

        with open(paths.GPT2_LM_MDL, "rb") as f:
            _gpt2_lm = pickle.load(f)
        _gpt2_lm.eval()
    return _gpt2_lm


def warm_up() -> None:
    """Load the LM eagerly (call once at startup so the first request
    doesn't pay the pickle load)."""
    _get_perplexity_lm()


def gpt2_perplexity_batch(texts: list[str]) -> np.ndarray:
    """Perplexity of each text under the raw pretrained GPT-2-fa: exp of the
    mean next-token negative log-likelihood over real (non-pad) tokens.
    Same serving normalization and char-budget guard as the GPT-2 classifier,
    so the signal sees exactly what the deployed GPT-2 runtime sees."""
    lm = _get_perplexity_lm()
    tokenizer = loaders.get_gpt2_tokenizer()
    char_budget = PERPLEXITY_SEQ_LEN * 8
    results: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), PERPLEXITY_BATCH_SIZE):
            enc = tokenizer(
                [
                    normalize_for_pretrained(str(t))[:char_budget]
                    for t in texts[start : start + PERPLEXITY_BATCH_SIZE]
                ],
                return_tensors="pt",
                truncation=True,
                max_length=PERPLEXITY_SEQ_LEN,
                padding="max_length",
            )
            logits = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            # Logits at position j predict token j+1; score only real targets.
            target_ids = enc["input_ids"][:, 1:]
            target_mask = enc["attention_mask"][:, 1:].bool()
            token_nll = -log_probs[:, :-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            for i in range(token_nll.size(0)):
                nll = token_nll[i][target_mask[i]]
                # Texts too short for even one prediction stay NaN.
                results.append(float(nll.mean().exp()) if nll.numel() else float("nan"))
    return np.array(results)


def gpt2_perplexity(text: str) -> float:
    """Single-text convenience wrapper around gpt2_perplexity_batch."""
    return float(gpt2_perplexity_batch([text])[0])

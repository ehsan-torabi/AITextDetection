"""Train the JEV-style "System One" decision model on the Persian dataset.

Architecture: the fine-tuned ParsBERT encoder (parsbertClassifierModel.pt --
the deployed bidirectional classifier trained by training/train_parsbert.py)
is reused as the JEV backbone, its [CLS] representation feeding three small
*decision heads*, one per typed question, all scored in a single forward pass
instead of autoregressive answer generation:

- choice head -> probability distribution over ["human_written", "ai_generated"]
- score head  -> distribution over 5 ordinal machine-likeness levels; the
                 answer is the probability-weighted value (can fall between levels)
- noul head   -> probability for the yes/no "needs human review" proposition

Training data comes from training/jev_data/jev_{train,test}.jsonl (built by
training/prepare_jev_data.py) with soft targets distilled from the deployed
fine-tuned GPT-2 classifier (gpt2ClassifierModel.pt) -- the teacher the
ensemble already weights highest. Loss = weighted sum of soft cross-entropy
on the two distributions + binary cross-entropy on the noul head.

Same CPU budget as training/train_parsbert.py: only the last 4 of 12 encoder
layers are unfrozen (the pooler stays frozen -- [CLS] is pooled directly, so
the pooler is never called), SEQ_LEN=64, 3 epochs over the same 6000/1500
stratified subsample. Texts get the same light normalization the serving path
applies (utils._normalize_for_gpt2), once at load time.

The saved checkpoint is self-contained: it stores the backbone config, so the
server rebuilds the encoder offline and no pretrained files are needed at
serving time (the BERT branch of utils._load_jev_model).

Run from the project root: python training/train_jev.py
"""

import json
import os
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModel, AutoTokenizer, BertConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402  (serving pipeline: shared to guarantee train/serve parity)
from split import RANDOM_SEED  # noqa: E402

MODELS_JSON_PATH = "models.json"
MODEL_PATH = "jevModel.pt"
BASE_CHECKPOINT_PATH = "parsbertClassifierModel.pt"
BASE_BACKBONE_NAME = "HooshvareLab/bert-fa-base-uncased"  # tokenizer fallback
DATA_DIR = "training/jev_data"

SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR_HEAD = 1e-3
LR_BACKBONE = 2e-5
WEIGHT_DECAY = 0.01
UNFROZEN_LAYERS = 4  # fine-tune only the last N of 12 encoder layers
DROPOUT = 0.1

CHOICE_WEIGHT = 1.0
SCORE_WEIGHT = 0.5
NOUL_WEIGHT = 0.5

# Keep in sync with training/prepare_jev_data.py (which writes these into the
# JSONL schema) and with the Persian labels in utils.py.
CHOICE_CANDIDATES = ["human_written", "ai_generated"]
SCORE_LEVELS = [
    "definitely_human",
    "probably_human",
    "mixed_signals",
    "probably_ai",
    "definitely_ai",
]

# Decision policy thresholds served with the checkpoint (three tiers, as in
# the JEV guide: proceed / gather more info / reject).
PROCEED_THRESHOLD = 0.8
NEEDS_REVIEW_THRESHOLD = 0.6

torch.manual_seed(RANDOM_SEED)


class JevDecisionModel(nn.Module):
    """ParsBERT backbone (bidirectional, so pooled on [CLS]) + typed decision
    heads. pool_on_cls is a flag rather than an assumption so the serving copy
    in utils.py can also load the legacy GPT-2-backed checkpoint (pooled on the
    last real token); this trainer always sets it True. Must match the copy in
    utils.py exactly so the saved state_dict loads at serving time."""

    def __init__(self, backbone, hidden_size, n_choice, n_score_levels, pool_on_cls):
        super().__init__()
        self.backbone = backbone
        self.pool_on_cls = pool_on_cls
        self.dropout = nn.Dropout(DROPOUT)
        self.choice_head = nn.Linear(hidden_size, n_choice)
        self.score_head = nn.Linear(hidden_size, n_score_levels)
        self.noul_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool_on_cls:
            # [CLS] is always token position 0 for a BERT-style encoder: every
            # position already attends to the whole sequence, so unlike the
            # causal GPT-2 there is no attention-mask index arithmetic needed
            # to find "the token that saw everything".
            pooled = hidden[:, 0]
        else:
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), last_idx]
        pooled = self.dropout(pooled)
        return (
            self.choice_head(pooled),
            self.score_head(pooled),
            self.noul_head(pooled).squeeze(-1),
        )


def load_tokenizer():
    """Same offline-first fallback order as utils._load_parsbert_tokenizer."""
    if os.path.isdir("parsbert_tokenizer"):
        return AutoTokenizer.from_pretrained("parsbert_tokenizer")
    return AutoTokenizer.from_pretrained(BASE_BACKBONE_NAME)


def load_parsbert_backbone():
    """Rebuild the fine-tuned ParsBERT encoder exactly the way the server does
    (utils._load_parsbert_model): architecture from the checkpoint's stored
    backbone config, weights from the state_dict -- then hand the backbone to
    JEV with its fine-tuned classification knowledge intact."""
    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    base_config = checkpoint["config"]
    backbone = AutoModel.from_config(BertConfig.from_dict(base_config["backbone_config"]))
    classifier = utils.ParsBERTClassifier(backbone, base_config["hidden_size"])
    classifier.load_state_dict(checkpoint["state_dict"])
    return classifier.backbone


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def target_arrays(records):
    choice = np.array(
        [[r["targets"]["source"][c] for c in CHOICE_CANDIDATES] for r in records],
        dtype=np.float32,
    )
    score = np.array([r["targets"]["ai_level"] for r in records], dtype=np.float32)
    noul = np.array([r["targets"]["needs_review"] for r in records], dtype=np.float32)
    labels = np.array([r["label"] for r in records])
    return choice, score, noul, labels


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def soft_cross_entropy(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def expected_calibration_error(probs, labels, n_bins=15):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(bins):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def assert_serving_contract(model, backbone_config, hidden_size):
    """Rebuild the checkpoint the way the BERT branch of utils._load_jev_model
    does (BertConfig.from_dict -> AutoModel.from_config, i.e. with no
    pretrained weights) and check the state_dict keys line up. A silent
    key/shape mismatch here would only surface as a load error on the first
    request against the live server."""
    rebuilt_backbone = AutoModel.from_config(BertConfig.from_dict(backbone_config))
    rebuilt = JevDecisionModel(
        rebuilt_backbone,
        hidden_size,
        n_choice=len(CHOICE_CANDIDATES),
        n_score_levels=len(SCORE_LEVELS),
        pool_on_cls=True,
    )
    drift = set(model.state_dict()) ^ set(rebuilt.state_dict())
    assert not drift, f"state_dict does not match the serving class: {drift}"


def main():
    print("Loading JEV training data...")
    train_records = load_jsonl(f"{DATA_DIR}/jev_train.jsonl")
    test_records = load_jsonl(f"{DATA_DIR}/jev_test.jsonl")
    if not train_records or not test_records:
        raise SystemExit("No JEV records found -- run training/prepare_jev_data.py first.")
    print(f"{len(train_records)} train / {len(test_records)} test records")

    choice_t, score_t, noul_t, _ = target_arrays(train_records)
    _, score_t_test, noul_t_test, y_test = target_arrays(test_records)
    # Same light normalization the serving path applies for this runtime
    # (utils._predict_jev), applied once instead of per epoch.
    normalize_start = time.time()
    texts = [utils._normalize_for_gpt2(r["state"]["text"]) for r in train_records]
    texts_test = [utils._normalize_for_gpt2(r["state"]["text"]) for r in test_records]
    print(
        f"Normalized {len(texts) + len(texts_test)} texts in {time.time() - normalize_start:.0f}s"
    )

    print("Loading the fine-tuned ParsBERT backbone (parsbertClassifierModel.pt)...")
    backbone = load_parsbert_backbone()
    tokenizer = load_tokenizer()
    hidden_size = backbone.config.hidden_size
    n_layers = backbone.config.num_hidden_layers

    for p in backbone.parameters():
        p.requires_grad = False
    for layer in backbone.encoder.layer[n_layers - UNFROZEN_LAYERS :]:
        for p in layer.parameters():
            p.requires_grad = True
    # The pooler stays frozen: JEV pools the raw [CLS] hidden state, so the
    # pooler is never called (same as the ParsBERT classifier's own forward).

    device = torch.device("cpu")
    torch.set_num_threads(12)
    model = JevDecisionModel(
        backbone,
        hidden_size,
        n_choice=len(CHOICE_CANDIDATES),
        n_score_levels=len(SCORE_LEVELS),
        pool_on_cls=True,
    ).to(device)

    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": LR_BACKBONE},
            {"params": model.choice_head.parameters(), "lr": LR_HEAD},
            {"params": model.score_head.parameters(), "lr": LR_HEAD},
            {"params": model.noul_head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    def encode_batch(batch_texts):
        # The WordPiece tokenizer is fast (Rust-backed) and truncates before
        # its full encode, so no character-level pre-truncation is needed --
        # same encode path as training/train_parsbert.py.
        return tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    def loss_from_batch(enc, idx_batch):
        choice_logits, score_logits, noul_logit = model(enc["input_ids"], enc["attention_mask"])
        choice_loss = soft_cross_entropy(choice_logits, torch.from_numpy(choice_t[idx_batch]))
        score_loss = soft_cross_entropy(score_logits, torch.from_numpy(score_t[idx_batch]))
        noul_loss = nn.functional.binary_cross_entropy_with_logits(
            noul_logit, torch.from_numpy(noul_t[idx_batch])
        )
        return CHOICE_WEIGHT * choice_loss + SCORE_WEIGHT * score_loss + NOUL_WEIGHT * noul_loss

    n_batches = (len(train_records) + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(RANDOM_SEED + epoch)
        order = np.arange(len(train_records))
        rng.shuffle(order)
        epoch_start = time.time()
        total_loss = 0.0
        for step, idx_batch in enumerate(batches(order, BATCH_SIZE), start=1):
            enc = encode_batch([texts[i] for i in idx_batch])
            optimizer.zero_grad()
            loss = loss_from_batch(enc, idx_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx_batch)
            if step % 20 == 0 or step == n_batches:
                elapsed = time.time() - epoch_start
                rate = elapsed / step
                remaining = (n_batches - step) * rate
                print(
                    f"epoch {epoch} batch {step}/{n_batches} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.0f}s "
                    f"eta={remaining:.0f}s",
                    flush=True,
                )
        print(
            f"Epoch {epoch} done. avg_loss={total_loss / len(train_records):.4f} "
            f"time={time.time() - epoch_start:.0f}s"
        )

    print("Evaluating typed answers on the held-out subsample...")
    model.eval()
    choice_probs_all, score_probs_all, noul_probs_all = [], [], []
    eval_start = time.time()
    with torch.no_grad():
        for idx_batch in batches(np.arange(len(test_records)), 32):
            enc = encode_batch([texts_test[i] for i in idx_batch])
            choice_logits, score_logits, noul_logit = model(enc["input_ids"], enc["attention_mask"])
            choice_probs_all.append(torch.softmax(choice_logits, dim=-1).numpy())
            score_probs_all.append(torch.softmax(score_logits, dim=-1).numpy())
            noul_probs_all.append(torch.sigmoid(noul_logit).numpy())
    print(f"Eval forward passes done in {time.time() - eval_start:.0f}s")

    choice_probs = np.concatenate(choice_probs_all)
    score_probs = np.concatenate(score_probs_all)
    noul_probs = np.concatenate(noul_probs_all)

    preds = choice_probs.argmax(axis=1)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        # JEV per-type metrics: score answer error (probability-weighted value
        # vs target value on the 0..4 scale), noul ranking quality, and
        # calibration of the choice distribution.
        "scoreMae": round(
            float(
                mean_absolute_error(
                    score_t_test @ np.arange(len(SCORE_LEVELS)),
                    score_probs @ np.arange(len(SCORE_LEVELS)),
                )
            ),
            4,
        ),
        "noulAuc": None,
        "choiceEce": round(expected_calibration_error(choice_probs, y_test), 4),
    }
    # Both noul classes must be present for AUC; on a near-separable dataset
    # the uncertain band can be nearly empty.
    if len(np.unique(noul_t_test)) > 1:
        metrics["noulAuc"] = round(float(roc_auc_score(noul_t_test, noul_probs)), 4)
        metrics["noulAveragePrecision"] = round(
            float(average_precision_score(noul_t_test, noul_probs)), 4
        )
    print("Final metrics:", metrics)

    backbone_config = backbone.config.to_dict()
    assert_serving_contract(model, backbone_config, hidden_size)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": SEQ_LEN,
                "hidden_size": hidden_size,
                "n_choice": len(CHOICE_CANDIDATES),
                "n_score_levels": len(SCORE_LEVELS),
                "choice_candidates": CHOICE_CANDIDATES,
                "score_levels": SCORE_LEVELS,
                "thresholds": {
                    "proceed": PROCEED_THRESHOLD,
                    "needsReview": NEEDS_REVIEW_THRESHOLD,
                },
                # The server rebuilds the BERT encoder from this alone (no
                # pretrained download needed), then overwrites it with
                # state_dict. Its presence also tells utils._load_jev_model
                # this is a [CLS]-pooled BERT checkpoint, not the legacy
                # GPT-2-backed format.
                "backbone_config": backbone_config,
            },
            "metrics": {k: v for k, v in metrics.items() if v is not None},
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    metrics_for_json = {k: v for k, v in metrics.items() if v is not None}
    with open(MODELS_JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    # Carry over ensembleWeight from the current entry so a re-run can't
    # silently change every ensemble prediction (same convention as the Colab
    # notebook, which documents why: _ensemble_predict skips weights <= 0 and
    # rescales the rest).
    previous = next((m for m in existing if m["id"] == "jev_system_one"), {})
    entry = {
        "id": "jev_system_one",
        "name": "JEV System One (Decision Head)",
        "nameFa": "مدل تصمیم سیستم-یک (JEV)",
        "description": (
            "JEV-style 'System One' decision model: the fine-tuned ParsBERT "
            "encoder (the deployed bidirectional classifier, with its last 4 "
            "layers adapted further here) pooled on [CLS] into three typed "
            "decision heads answered in a single forward pass -- a choice "
            "question (human vs AI, as a probability distribution), a score "
            "question (machine-likeness on a 5-level scale, answered as a "
            "probability-weighted value between levels) and a noul question "
            "(probability that human review is needed). Trained on a "
            "stratified subsample with soft targets distilled from the "
            "deployed fine-tuned GPT-2 classifier."
        ),
        "descriptionFa": (
            "مدل تصمیم «سیستم-یک» به سبک JEV: رمزگذار دوسویه ParsBERT "
            "فاین‌تیون‌شده (همان دسته‌بند دوسویهٔ مستقر که ۴ لایه پایانی آن "
            "اینجا نیز تطبیق داده شده است) که بازنمایی توکن [CLS] آن به سه سر "
            "تصمیم نوع‌بندی‌شده می‌رود و همه در یک گذر رو به جلو پاسخ داده "
            "می‌شوند — سؤال انتخاب (انسان در برابر هوش مصنوعی به شکل توزیع "
            "احتمال)، سؤال امتیاز (ماشین‌مانندبودن روی طیف ۵ سطحی با مقدار "
            "وزن‌دار احتمالاتی که می‌تواند بین سطوح بیفتد) و سؤانول (احتمال "
            "نیاز به بازبینی انسانی). آموزش روی نمونهٔ طبقه‌بندی‌شده‌ای از داده "
            "با هدف‌های نرم تقطیرشده از دسته‌بند GPT-2 فاین‌تیون‌شدهٔ مستقر."
        ),
        "algorithm": "ParsBERT (fine-tuned encoder) + typed decision heads (choice/score/noul)",
        "type": "dl",
        "version": "2.0",
        "filename": MODEL_PATH,
        "runtime": "pytorch-jev",
        "ensembleWeight": float(previous.get("ensembleWeight", 1.0)),
        "metrics": metrics_for_json,
        "confusionMatrix": {
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "truePositive": int(tp),
        },
        "training": {
            "totalSamples": len(train_records) + len(test_records),
            "trainSamples": len(train_records),
            "testSamples": len(test_records),
            "featureCount": hidden_size,
            "testSplitRatio": len(test_records) / (len(train_records) + len(test_records)),
            "randomSeed": RANDOM_SEED,
            "dataset": (
                "processedDataset.csv (JEV JSONL, soft targets distilled from "
                "the fine-tuned GPT-2 classifier)"
            ),
        },
    }

    existing = [m for m in existing if m["id"] != entry["id"]]
    existing.append(entry)
    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("models.json updated.")


if __name__ == "__main__":
    main()

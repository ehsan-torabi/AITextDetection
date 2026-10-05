"""The ModelStore: loading, caching, and orchestrating every detection model.

Replaces the module-level globals of the former utils.py: one object owns the
vectorizer, the model metadata from models.json, and every loaded model
bundle. The FastAPI layer holds exactly one store; tests can instantiate
their own.
"""

import json
import math

import joblib

from aitextd import loaders, paths, perplexity, predictors, system_one
from aitextd.constants import ENSEMBLE_ID
from aitextd.features import compute_style_features, shannon_entropy
from aitextd.schemas import DetectionDetail, DetectionResponse, EnsembleVote
from aitextd.text import preprocess_persian_text


def detail_rows(
    text: str, style_features: dict[str, float], entropy: float
) -> list[DetectionDetail]:
    """The shared analysis breakdown: style features plus raw-LM perplexity.

    Model-independent, so every serving path (local /detect, online route)
    shows the same rows for the same text.
    """
    details = [
        DetectionDetail(
            label="entropy",
            labelFa="آنتروپی",
            value=style_features["entropy"],
            raw=f"{entropy:.2f} bits",
            description=f"آنتروپی کاراکترها: {entropy:.2f} بیت (نرمال‌شده بر حسب تنوع کاراکتر)",
        ),
        DetectionDetail(
            label="sentence_variety",
            labelFa="تنوع جملات",
            value=style_features["sentence_variety"],
            raw=f"{style_features['sentence_variety']:.2f}",
            description=f"پراکندگی طول جمله‌ها (ضریب تغییرات): {style_features['sentence_variety']:.2f}",
        ),
        DetectionDetail(
            label="repetition",
            labelFa="تکرار",
            value=style_features["repetition"],
            raw=f"{style_features['repetition'] * 100:.1f}%",
            description=f"درصد تکرار سه‌گرام‌ها: {style_features['repetition'] * 100:.1f}%",
        ),
        DetectionDetail(
            label="coherence",
            labelFa="انسجام",
            value=style_features["coherence"],
            raw=f"{style_features['coherence']:.2f}",
            description=f"میانگین شباهت جمله‌های مجاور: {style_features['coherence']:.2f}",
        ),
        DetectionDetail(
            label="lexical_diversity",
            labelFa="تنوع واژگانی",
            value=style_features["lexical_diversity"],
            raw=f"{style_features['lexical_diversity']:.2f}",
            description=f"نسبت واژه‌های یکتای متن به کل واژه‌ها: {style_features['lexical_diversity']:.2f}",
        ),
        DetectionDetail(
            label="sentence_length",
            labelFa="میانگین طول جمله",
            value=style_features["sentence_length"],
            raw=f"{style_features['sentence_length']:.2f}",
            description=f"میانگین طول جمله‌ها (نرمال‌شده، ۵۰ واژه = ۱): {style_features['sentence_length']:.2f}",
        ),
        DetectionDetail(
            label="function_word_ratio",
            labelFa="واژه‌های کارکردی",
            value=style_features["function_word_ratio"],
            raw=f"{style_features['function_word_ratio'] * 100:.1f}%",
            description=f"سهم واژه‌های کارکردی از کل واژه‌ها: {style_features['function_word_ratio'] * 100:.1f}%",
        ),
        DetectionDetail(
            label="punctuation_rate",
            labelFa="علائم نگارشی",
            value=style_features["punctuation_rate"],
            raw=f"{style_features['punctuation_rate'] * 100:.1f}%",
            description=f"چگالی علائم نگارشی در متن: {style_features['punctuation_rate'] * 100:.1f}%",
        ),
        DetectionDetail(
            label="lexical_burstiness",
            labelFa="انفجار واژگانی",
            value=style_features["lexical_burstiness"],
            raw=f"{style_features['lexical_burstiness']:.2f}",
            description=(
                "ضریب تغییرات فاصلهٔ وقوع واژه‌های تکرارشونده (انفجار واژگانی): "
                f"{style_features['lexical_burstiness']:.2f}"
            ),
        ),
        DetectionDetail(
            label="clause_variety",
            labelFa="تنوع جمله‌واره‌ها",
            value=style_features["clause_variety"],
            raw=f"{style_features['clause_variety']:.2f}",
            description=f"پراکندگی طول جمله‌واره‌ها (ضریب تغییرات): {style_features['clause_variety']:.2f}",
        ),
    ]

    # Raw-LM perplexity: a training-free view of the text (how "surprised"
    # the untouched pretrained GPT-2 is), independent of every fine-tuned
    # vote.
    ppl = perplexity.gpt2_perplexity(text)
    if not math.isnan(ppl):
        details.append(
            DetectionDetail(
                label="perplexity",
                labelFa="سرگشتگی",
                value=round(ppl, 2),
                raw=f"{ppl:.2f}",
                description=(
                    "سرگشتگی متن تحت مدل زبانی خام GPT-2 فارسی (بدون آموزش): "
                    "متنِ قابل‌پیش‌بینی‌تر امتیاز کمتری می‌گیرد؛ سیگنالی مستقل از "
                    "مدل‌های آموزش‌دیده."
                ),
            )
        )
    return details


class ModelStore:
    def __init__(self):
        self.vectorizer = None
        self.models_cache: dict[str, object] = {}  # model_id -> loaded bundle
        self.models_dict: dict[str, dict] = {}  # model_id -> models.json metadata

    # -- loading ----------------------------------------------------------

    def init(self) -> None:
        """Load the vectorizer and all models once at startup."""
        self.vectorizer = joblib.load(paths.TFIDF_VECTORIZER_JOBLIB)
        with open(paths.MODELS_JSON, encoding="utf-8") as f:
            metadata_list = json.load(f)
        self.models_dict = {md["id"]: md for md in metadata_list}

        for model_id, md in self.models_dict.items():
            self.models_cache[model_id] = self._load_bundle(md)

        # The perplexity signal runs on every detection request; warm it up
        # here so the first request doesn't pay the pickle load.
        perplexity.warm_up()

    def _load_bundle(self, md: dict) -> object:
        runtime = md.get("runtime", "sklearn-tfidf")
        if runtime == "sklearn-tfidf":
            return loaders.load_sklearn_model(md, self.vectorizer)
        return loaders.LOADERS.get(runtime, loaders.load_sklearn_model)(md)

    # -- metadata -----------------------------------------------------------

    def get_model_metadata(self, model_id: str) -> dict:
        if model_id not in self.models_dict:
            raise ValueError(f"Model ID '{model_id}' not found.")
        return self.models_dict[model_id]

    def get_ensemble_metadata(self) -> dict:
        """Virtual "model" metadata for the weighted ensemble, exposed to
        clients alongside the real models so it can be listed/selected like
        any other."""
        return {
            "id": ENSEMBLE_ID,
            "name": "Ensemble (All Models)",
            "nameFa": "مدل ترکیبی (رأی‌گیری وزن‌دار)",
            "description": (
                "Weighted-vote ensemble that combines the prediction of every "
                "trained model, giving extra weight to the fine-tuned GPT-2 model."
            ),
            "descriptionFa": (
                "ترکیب وزن‌دار پیش‌بینی همه مدل‌های آموزش‌دیده، با وزن بیشتر برای "
                "مدل GPT-2 ریزتنظیم‌شده که معمولاً دقیق‌ترین مدل است."
            ),
            "algorithm": "WeightedEnsemble",
            "type": "ensemble",
            "version": "1.0",
        }

    # -- prediction ---------------------------------------------------------

    def _predict_single(self, model_id: str, text: str, cleaned_text: str, style_features: dict):
        """Run one model: returns its AI probability as a float, except the
        System One decision model which returns the full typed-answers dict."""
        bundle = self.models_cache[model_id]
        md = self.models_dict[model_id]
        runtime = md.get("runtime", "sklearn-tfidf")
        predictor = predictors.PREDICTORS.get(runtime, predictors.predict_sklearn)
        return predictor(bundle, text, cleaned_text, style_features)

    def _ensemble_predict(
        self, text: str, cleaned_text: str, style_features: dict
    ) -> tuple[bool, float, list[EnsembleVote]]:
        """Weighted-average vote across every loaded model. Each model's weight
        comes from its "ensembleWeight" metadata field (models.json) -- the
        fine-tuned GPT-2 model is weighted higher there since it consistently
        outperforms the classical ML models on held-out text."""
        votes: list[EnsembleVote] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for model_id, md in self.models_dict.items():
            weight = float(md.get("ensembleWeight", 1.0))
            if weight <= 0.0:
                # Models can opt out of the vote via "ensembleWeight": 0.
                continue
            result = self._predict_single(model_id, text, cleaned_text, style_features)
            if isinstance(result, dict):
                # pytorch-system-one returns the full typed-answers dict; vote with
                # its choice-question probability, same as detection() does.
                ai_prob = result["aiProbability"]
            else:
                ai_prob = result
            weighted_sum += ai_prob * weight
            weight_total += weight
            votes.append(
                EnsembleVote(
                    modelId=model_id,
                    label=md.get("name", model_id),
                    labelFa=md.get("nameFa", md.get("name", model_id)),
                    aiProbability=round(ai_prob, 4),
                    weight=weight,
                )
            )

        ai_prob_avg = weighted_sum / weight_total if weight_total else 0.0
        is_ai = ai_prob_avg >= 0.5
        confidence = round(ai_prob_avg if is_ai else 1 - ai_prob_avg, 4)
        return is_ai, confidence, votes

    # -- API entrypoint -------------------------------------------------------

    def detail_breakdown(self, text: str) -> list[DetectionDetail]:
        """Model-independent analysis rows for a raw text (online path, or any
        client that wants the breakdown without running a classifier)."""
        cleaned_text = preprocess_persian_text(text)
        if not cleaned_text.strip():
            return []
        style_features = compute_style_features(text, cleaned_text, self.vectorizer)
        entropy = shannon_entropy(text)
        return detail_rows(text, style_features, entropy)

    def detection(self, text: str, model_id: str) -> DetectionResponse:
        """Detect whether the given Persian text is AI-generated using the
        specified model, or every model at once via the weighted ensemble
        (model_id == "ensemble")."""
        is_ensemble = model_id == ENSEMBLE_ID
        if not is_ensemble and (
            model_id not in self.models_cache or model_id not in self.models_dict
        ):
            raise ValueError(f"Model ID '{model_id}' not found.")

        cleaned_text = preprocess_persian_text(text)

        if not cleaned_text.strip():
            return DetectionResponse(
                isAI=False,
                confidence=0.0,
                verdict="Unable to analyze empty text",
                verdictFa="متن خالی قابل تحلیل نیست",
                details=[],
                modelUsed=model_id,
                textLength=len(text),
                wordCount=len(text.split()),
            )

        # Computed once and reused both as extra model input (sklearn-tfidf
        # models) and for the human-readable detail breakdown below.
        style_features = compute_style_features(text, cleaned_text, self.vectorizer)
        entropy = shannon_entropy(text)  # raw bits, for the "raw" display field only

        ensemble_votes = None
        typed_answers = None
        if is_ensemble:
            is_ai, confidence, ensemble_votes = self._ensemble_predict(
                text, cleaned_text, style_features
            )
        else:
            result = self._predict_single(model_id, text, cleaned_text, style_features)
            if isinstance(result, dict):
                # pytorch-system-one: one forward pass, three typed answers
                typed_answers = system_one.typed_answers(self.models_cache[model_id], result)
                ai_prob = result["aiProbability"]
            else:
                ai_prob = result
            is_ai = ai_prob >= 0.5
            confidence = round(ai_prob if is_ai else 1 - ai_prob, 4)

        word_count = len(text.split())
        char_count = len(text)

        details = detail_rows(text, style_features, entropy)

        if is_ai:
            verdict = "This text appears to be AI-generated."
            verdictFa = "این متن به نظر تولید شده توسط هوش مصنوعی است."
        else:
            verdict = "This text appears to be human-written."
            verdictFa = "این متن به نظر توسط انسان نوشته شده است."

        return DetectionResponse(
            isAI=is_ai,
            confidence=confidence,
            verdict=verdict,
            verdictFa=verdictFa,
            details=details,
            modelUsed=model_id,
            textLength=char_count,
            wordCount=word_count,
            ensembleVotes=ensemble_votes,
            typedAnswers=typed_answers,
        )

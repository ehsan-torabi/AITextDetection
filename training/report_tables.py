"""Generate the LaTeX fragments embedded in ``report/report.tex`` from artifacts.

The v2 chapters of the report must not contain hand-typed numbers: every table
and every inline count is emitted by this script from the files the pipeline
already writes, so a re-run of the dataset / splits / evaluation steps updates
the report instead of leaving stale figures behind.

Inputs (all optional; a missing one produces a visible placeholder):

    data/dataset_v2.manifest.json    composition, dedup statistics, provenance
    data/human/provenance.json       per-domain collection record and licence
    data/splits_v2.json              split inventory and integrity assertions
    reports/eval_results.json        metrics, CIs, decision policy, attack
    reports/error_analysis/summary.json   reviewed FP/FN and taxonomy counts

Outputs (``report/generated/``):

    macros.tex            inline counts used in the v2 prose (\\dtRows, \\spTest, ...)
    tab_dataset.tex       label mix, length / formality / origin mix
    tab_domains.tex       human domains with source and licence
    tab_splits.tex        every evaluation condition and its sizes
    tab_results.tex       in-domain metric suite per model
    tab_conditions.tex    macro-F1 per condition and model
    tab_attack.tex        clean vs paraphrased recall, paired drop
    tab_three_class.tex   three-class view and its binary collapse
    tab_errors.tex        false positive / false negative taxonomy

Run from the project root (after the pipeline steps whose numbers are wanted):
    python training/report_tables.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"
REPORTS = PROJECT_ROOT / "reports"
OUT_DIR = PROJECT_ROOT / "report" / "generated"
REPORT_TEX = PROJECT_ROOT / "report" / "report.tex"

MANIFEST = DATA / "dataset_v2.manifest.json"
HUMAN_PROVENANCE = DATA / "human" / "provenance.json"
SPLITS_JSON = DATA / "splits_v2.json"
RESULTS_JSON = REPORTS / "eval_results.json"
ERROR_SUMMARY = REPORTS / "error_analysis" / "summary.json"

DEFAULT_SEEDS = (42, 43, 44)

# Persian labels for the canonical values in aitextd/dataset.py
DOMAIN_LABELS = {
    "blog": "وبلاگ",
    "news": "خبر",
    "encyclopedia": "دانشنامه",
    "social_media": "شبکهٔ اجتماعی",
    "legal_admin": "اداری و حقوقی",
    "translated_subtitles": "زیرنویس ترجمه‌ای",
    "ai_multi": "چندموضوعی (ماشین)",
}
LABEL_LABELS = {"human": "انسان", "ai": "ماشین (خام)", "ai_revised": "بازنویسی‌شده"}
BUCKET_LABELS = {
    "short": "کوتاه ($\\le 100$ واژه)",
    "medium": "متوسط",
    "long": "بلند ($\\ge 500$ واژه)",
    "formal": "رسمی",
    "neutral": "خنثی",
    "informal": "خودمانی",
    "persian": "فارسی",
    "translated": "ترجمه‌ای",
}
# Short corpus names for the table; the full Hugging Face ids are in
# data/human/provenance.json, which the caption points to.
SOURCE_LABELS = {
    "blog": "پیکرهٔ وبلاگ پروژه",
    "news": "isna-news",
    "encyclopedia": "wikipedia:20231101.fa",
    "social_media": "PersianTwitterDataset",
    "legal_admin": "iran-legal-persian-qa",
    "translated_subtitles": "persian-subtitles",
}
# Licences stay Latin (they are identifiers); the project corpus has no
# external licence file, and the report says so instead of inventing one.
UNLICENSED_LABEL = "پیکرهٔ پروژه (بدون مجوز بیرونی)"

# Model keys as they appear in reports/eval_results.json
MODEL_LABELS = {
    "tfidf_svm": "\\en{TF-IDF + SVM}",
    "tfidf_logreg": "\\en{TF-IDF + LR}",
    "tfidf_nb": "\\en{TF-IDF + NB}",
    "tfidf_style_logreg": "\\en{TF-IDF} + سبک + \\en{LR}",
    "style_rf": "سبک‌شناختی + \\en{RF}",
    "tfidf_style_rf": "\\en{TF-IDF} + سبک + \\en{RF}",
    "length_only": "تنها طول",
    "perplexity_only": "تنها سرگشتگی",
}
# Macro-suffix form of the same keys (\\resFoneSvm, ...)
MODEL_SUFFIXES = {
    "tfidf_svm": "Svm",
    "tfidf_logreg": "Logreg",
    "tfidf_nb": "Nb",
    "tfidf_style_logreg": "TfidfStyleLogreg",
    "style_rf": "StyleRf",
    "tfidf_style_rf": "TfidfStyleRf",
    "length_only": "Length",
    "perplexity_only": "Perp",
}
MODEL_ORDER = list(MODEL_LABELS)

CONDITION_LABELS = {
    "cross_generator": "برون‌تولیدکننده (تولیدکننده و موضوع نادیده)",
    "cross_domain__news": "برون‌دامنه: خبر",
    "cross_domain__encyclopedia": "برون‌دامنه: دانشنامه",
    "cross_domain__social_media": "برون‌دامنه: شبکهٔ اجتماعی",
    "cross_domain__legal_admin": "برون‌دامنه: اداری و حقوقی",
    "cross_domain__blog": "برون‌دامنه: وبلاگ",
    "cross_domain__translated_subtitles": "برون‌دامنه: زیرنویس ترجمه‌ای",
    "cross_style__conversational_blog": "برون‌سبک: وبلاگِ گفت‌وگویی",
    "cross_style__formal_essay": "برون‌سبک: مقالهٔ رسمی",
    "cross_style__opinion_critical": "برون‌سبک: یادداشت انتقادی",
    "cross_style__qa_explainer": "برون‌سبک: پرسش و توضیح",
    "cross_style__summary_overview": "برون‌سبک: خلاصهٔ مروری",
    "few_shot__8": "نمونهٔ کم ($8$)",
    "few_shot__32": "نمونهٔ کم ($32$)",
    "few_shot__128": "نمونهٔ کم ($128$)",
    "length__short": "آزمون روی دامنهٔ کوتاه",
    "length__long": "آزمون روی دامنهٔ بلند",
    "v1_test": "آزمون قابل‌قیاس با نسخهٔ نخست",
}
CROSS_DOMAIN_SUFFIX = {
    "cross_domain__news": "News",
    "cross_domain__encyclopedia": "Encyclopedia",
    "cross_domain__social_media": "Social",
    "cross_domain__legal_admin": "Legal",
    "cross_domain__blog": "Blog",
    "cross_domain__translated_subtitles": "Subtitles",
}
CROSS_STYLE_SUFFIX = {
    "cross_style__conversational_blog": "Blog",
    "cross_style__formal_essay": "Formal",
    "cross_style__opinion_critical": "Opinion",
    "cross_style__qa_explainer": "Qa",
    "cross_style__summary_overview": "Summary",
}
ATTACK_METHOD_LABELS = {
    "backtranslation": "ترجمهٔ رفت‌وبرگشتی",
    "lexical_substitution": "جانشینی واژگانی",
    "sentence_perturbation": "جابه‌جایی مرزهای جمله",
    "punct_spacing_noise": "آشفتگی نیم‌فاصله و نشانه‌گذاری",
    "llm_paraphrase": "بازنویسی با مدل زبانی",
}
TAXONOMY_LABELS = {
    "translated_text": "متن ترجمه‌ای",
    "short_document": "سند کوتاه",
    "social_media_register": "لحن شبکهٔ اجتماعی",
    "legal_admin_register": "لحن اداری و حقوقی",
    "informal_register": "لحن خودمانی",
    "encyclopedic_style": "سبک دانشنامه‌ای",
    "news_style": "سبک خبری",
    "long_formal_text": "متن بلند رسمی",
    "short_generation": "تولید کوتاه",
    "qa_style_generation": "تولید به سبک پرسش و توضیح",
    "summary_style_generation": "تولید به سبک خلاصهٔ مروری",
    "informal_generation": "تولید با لحن خودمانی",
    "long_generation": "تولید بلند",
    "formal_generation": "تولید رسمی",
    "unclassified": "طبقه‌بندی‌نشده",
}

SPLIT_ROWS = [
    ("in_domain_group", "درون‌دامنه (گروه‌آگاه)"),
    ("cross_generator", "برون‌تولیدکننده"),
    ("cross_domain__news", "برون‌دامنه: خبر"),
    ("cross_domain__encyclopedia", "برون‌دامنه: دانشنامه"),
    ("cross_domain__social_media", "برون‌دامنه: شبکهٔ اجتماعی"),
    ("cross_domain__legal_admin", "برون‌دامنه: اداری و حقوقی"),
    ("cross_domain__blog", "برون‌دامنه: وبلاگ"),
    ("cross_domain__translated_subtitles", "برون‌دامنه: زیرنویس ترجمه‌ای"),
    ("cross_style__conversational_blog", "برون‌سبک: وبلاگِ گفت‌وگویی"),
    ("cross_style__formal_essay", "برون‌سبک: مقالهٔ رسمی"),
    ("cross_style__opinion_critical", "برون‌سبک: یادداشت انتقادی"),
    ("cross_style__qa_explainer", "برون‌سبک: پرسش و توضیح"),
    ("cross_style__summary_overview", "برون‌سبک: خلاصهٔ مروری"),
    ("few_shot__8", "یادگیری با نمونهٔ کم ($8$)"),
    ("few_shot__32", "یادگیری با نمونهٔ کم ($32$)"),
    ("few_shot__128", "یادگیری با نمونهٔ کم ($128$)"),
    ("length__short", "طول: کوتاه (آزمون)"),
    ("length__long", "طول: بلند (آزمون)"),
    ("v1_test", "آزمون نسخهٔ نخست"),
    ("paraphrase_attention", "ذخیره‌شدهٔ حملهٔ بازنویسی"),
    ("human_vs_ai_revised", "انسان در برابر بازنویسی‌شده"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_json(path: Path):
    if not path.exists():
        print(f"missing artifact: {path.relative_to(PROJECT_ROOT)}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: int | float) -> str:
    """31498 -> 31{,}498 (math-mode friendly thousands separator)."""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}".replace(",", "{,}")
    return f"{int(value):,}".replace(",", "{,}")


def latin(text: str) -> str:
    """Wrap a value in \\en{} only when it is actually Latin text."""
    return text if re.search(r"[\u0600-\u06ff]", text) else f"\\en{{{text}}}"


def share(part: int, whole: int) -> str:
    return f"${100.0 * part / max(whole, 1):.1f}\\%$"


def write(name: str, body: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def table(
    caption: str,
    label: str,
    spec: str,
    header: list[str],
    rows: list[list[str]],
    font: str = "",
) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        *( [f"\\{font}"] if font else [] ),
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{cell}}}" for cell in header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def placeholder(label: str, needed: str) -> str:
    # artifact paths contain underscores, which are math shifts in text mode
    needed = needed.replace("_", "\\_")
    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{این جدول هنوز تولید نشده است ({needed})}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{l}\n\\toprule\n"
        f"در انتظار تولید: {needed} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


# ---------------------------------------------------------------------------
# reading reports/eval_results.json
# ---------------------------------------------------------------------------


def condition_record(results: dict | None, name: str) -> dict | None:
    for record in (results or {}).get("conditions", []):
        if record.get("condition") == name:
            return record
    return None


def model_records(results: dict | None, condition: str, model: str) -> list[dict]:
    record = condition_record(results, condition) or {}
    return list((record.get("models") or {}).get(model, []))


def primary_split(condition: str) -> str:
    return f"{condition}__test"


def finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_values(
    results: dict | None,
    condition: str,
    model: str,
    key: str,
    split: str | None = None,
    source: str = "metrics",
) -> list[float]:
    """One metric per seed, read from the stored per-split record."""
    name = split or primary_split(condition)
    values: list[float] = []
    for record in model_records(results, condition, model):
        block = (record.get("splits") or {}).get(name) or {}
        value = finite((block.get(source) or {}).get(key))
        if value is not None:
            values.append(value)
    return values


def ci_values(results: dict | None, condition: str, model: str, key: str, split: str | None = None) -> list[float]:
    name = split or primary_split(condition)
    values: list[float] = []
    for record in model_records(results, condition, model):
        block = (record.get("splits") or {}).get(name) or {}
        value = finite((block.get("ci95") or {}).get(key))
        if value is not None:
            values.append(value)
    return values


def policy_values(results: dict | None, condition: str, model: str, key: str) -> list[float]:
    values: list[float] = []
    for record in model_records(results, condition, model):
        block = (record.get("splits") or {}).get(primary_split(condition)) or {}
        value = finite((block.get("policy_test") or {}).get(key))
        if value is not None:
            values.append(value)
    return values


def attack_values(results: dict | None, model: str, key: str) -> list[float]:
    values: list[float] = []
    for record in model_records(results, "paraphrase_attack", model):
        value = finite(record.get(key))
        if value is not None:
            values.append(value)
    return values


def attack_paired(results: dict | None, model: str) -> list[dict]:
    records = []
    for record in model_records(results, "paraphrase_attack", model):
        paired = record.get("recall_drop_paired_ci95") or {}
        if finite(paired.get("mean_difference")) is not None:
            records.append(paired)
    return records


def three_class_records(results: dict | None, model: str) -> list[dict]:
    record = condition_record(results, "three_class") or {}
    return [r for r in record.get("results", []) if r.get("model") == model]


def three_class_metric(results: dict | None, model: str, key: str, block: str = "multiclass") -> list[float]:
    values = []
    for record in three_class_records(results, model):
        value = finite(((record.get(block) or {}).get("metrics") or {}).get(key))
        if value is not None:
            values.append(value)
    return values


# ---------------------------------------------------------------------------
# numeric formatting for the tables
# ---------------------------------------------------------------------------


def percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"${100.0 * value:.{digits}f}\\%$"


def percent_list(values: list[float], digits: int = 1) -> str:
    """mean +- sd, both in percent; the plain value when a single seed exists."""
    values = [v for v in (finite(v) for v in values) if v is not None]
    if not values:
        return "—"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return percent(mean, digits)
    sd = statistics.stdev(values)
    return f"${100.0 * mean:.{digits}f} \\pm {100.0 * sd:.{digits}f}\\%$"


def number_list(values: list[float], digits: int = 3) -> str:
    values = [v for v in (finite(v) for v in values) if v is not None]
    if not values:
        return "—"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"${mean:.{digits}f}$"
    return f"${mean:.{digits}f} \\pm {statistics.stdev(values):.{digits}f}$"


def mean_of(values: list[float]) -> float | None:
    values = [v for v in (finite(v) for v in values) if v is not None]
    return statistics.fmean(values) if values else None


def diff_percent(left: list[float], right: list[float], digits: int = 1) -> str:
    """Signed difference of two means in percentage points, as a bare number
    (the prose supplies the unit)."""
    a, b = mean_of(left), mean_of(right)
    if a is None or b is None:
        return "—"
    return f"${100.0 * (a - b):+.{digits}f}$"


# ---------------------------------------------------------------------------
# results tables
# ---------------------------------------------------------------------------


def build_tab_results(results: dict | None) -> str:
    """In-domain metric suite per model: the table the acceptance criteria are
    read from.  Every cell is a mean over seeds, with the seed spread shown for
    the headline metric."""
    if not results:
        return placeholder("tab:v2-results", "بدون reports/eval_results.json")
    rows = []
    for model in MODEL_ORDER:
        if not model_records(results, "in_domain_group", model):
            continue
        rows.append(
            [
                MODEL_LABELS[model],
                percent_list(metric_values(results, "in_domain_group", model, "macro_f1")),
                number_list(metric_values(results, "in_domain_group", model, "roc_auc"), 3),
                number_list(metric_values(results, "in_domain_group", model, "pr_auc"), 3),
                percent(mean_of(metric_values(results, "in_domain_group", model, "fpr_at_tpr95")), 2),
                percent(mean_of(metric_values(results, "in_domain_group", model, "ece_width")), 2),
                percent(mean_of(metric_values(results, "in_domain_group", model, "brier")), 2),
                percent(mean_of(policy_values(results, "in_domain_group", model, "deferred_share"))),
            ]
        )
    if not rows:
        return placeholder("tab:v2-results", "شرط درون‌دامنه در eval_results.json نیست")
    note = (
        "همهٔ سنجه‌ها روی مجموعهٔ آزمونِ گروه‌آگاه و با احتمال‌های کالیبره محاسبه شده‌اند؛ "
        "ستون نخست میانگین و انحراف معیار روی بذرها و ستون‌های دیگر میانگین روی بذرهاست."
    )
    return table(
        "کارنامهٔ درون‌دامنه به‌ازای هر مدل (آزمون گروه‌آگاه)",
        "tab:v2-results",
        "lccccccc",
        [
            "مدل",
            "$F_1$ ماکرو",
            "\\en{ROC-AUC}",
            "\\en{PR-AUC}",
            "\\en{FPR@TPR95}",
            "\\en{ECE}",
            "بریر",
            "سهم بازبینی",
        ],
        rows,
        font="small",
    ) + "\n\n" + note


def build_tab_conditions(results: dict | None, splits: dict | None) -> str:
    """Macro-F1 of every generalisation condition, one column per model."""
    if not results:
        return placeholder("tab:v2-conditions", "بدون reports/eval_results.json")
    names = [name for name in CONDITION_LABELS if condition_record(results, name)]
    if not names:
        return placeholder("tab:v2-conditions", "شرط‌های برون‌دامنه هنوز اجرا نشده‌اند")
    # three columns at most: the table has to survive the report's text width
    preferred = [
        "tfidf_svm",
        "tfidf_logreg",
        "style_rf",
        "perplexity_only",
        "length_only",
        "tfidf_nb",
        "tfidf_style_rf",
        "tfidf_style_logreg",
    ]
    models = [
        model
        for model in preferred
        if any(model_records(results, name, model) for name in names)
    ][:3]
    counts = (splits or {}).get("counts", {})
    rows = []
    for name in names:
        value = counts.get(primary_split(name), counts.get(name))
        cells = [CONDITION_LABELS[name], f"${num(value)}$" if value is not None else "—"]
        for model in models:
            cells.append(percent_list(metric_values(results, name, model, "macro_f1")))
        rows.append(cells)
    return table(
        "معیار $F_1$ ماکرو در شرط‌های تعمیم (میانگین و انحراف معیار روی بذرها)؛ "
        "ستون «آزمون» اندازهٔ مجموعهٔ آزمون هر شرط است",
        "tab:v2-conditions",
        "lr" + "c" * len(models),
        ["شرط", "آزمون", *(MODEL_LABELS[m] for m in models)],
        rows,
        font="small",
    )


def build_tab_attack(results: dict | None) -> str:
    if not results or not condition_record(results, "paraphrase_attack"):
        return placeholder("tab:v2-attack", "بدون شرط paraphrase_attack")
    attack = condition_record(results, "paraphrase_attack") or {}
    if attack.get("skipped"):
        return placeholder("tab:v2-attack", str(attack["skipped"]))
    rows = []
    for model in MODEL_ORDER:
        records = model_records(results, "paraphrase_attack", model)
        if not records:
            continue
        paired = attack_paired(results, model)
        drop = mean_of([p["mean_difference"] for p in paired])
        lo = mean_of([p["lo"] for p in paired])
        hi = mean_of([p["hi"] for p in paired])
        interval = (
            f"[${100.0 * lo:.1f}$, ${100.0 * hi:.1f}$]"
            if lo is not None and hi is not None
            else "—"
        )
        rows.append(
            [
                MODEL_LABELS[model],
                percent(mean_of(attack_values(results, model, "clean_recall_at_0.5"))),
                percent(mean_of(attack_values(results, model, "attack_recall_at_0.5"))),
                (percent(drop) if drop is not None else "—") + " " + interval,
            ]
        )
    if not rows:
        return placeholder("tab:v2-attack", "مدلی در شرط حمله ثبت نشده است")
    body = table(
        "حملهٔ بازنویسی: بازیابی مدل روی متن پاک و روی متن حمله‌شده "
        "(میانگین روی بذرها، یک ردیف به ازای هر سند)",
        "tab:v2-attack",
        "lccc",
        ["مدل", "بازیابی پیش از حمله", "بازیابی پس از حمله", "افت جفتی $95\\%$"],
        rows,
    )
    method_rows = []
    methods = sorted((attack.get("methods") or {}).keys())
    for method in methods:
        n_values, prob_values, recall_values, drop_values = [], [], [], []
        for model in MODEL_ORDER:
            for record in model_records(results, "paraphrase_attack", model):
                block = (record.get("by_method") or {}).get(method)
                if not block:
                    continue
                n_values.append(block.get("n"))
                prob_values.append(block.get("mean_ai_probability"))
                recall_values.append(block.get("recall_at_0.5"))
                drop_values.append(block.get("recall_drop_vs_clean"))
        method_rows.append(
            [
                ATTACK_METHOD_LABELS.get(method, latin(method)),
                f"${num(mean_of(n_values) or 0)}$",
                percent(mean_of(prob_values)),
                percent(mean_of(recall_values)),
                percent(mean_of(drop_values)),
            ]
        )
    if method_rows:
        body += "\n\n" + table(
            "حمله به تفکیک روش بازنویسی (میانگین روی مدل‌ها و بذرها)؛ «احتمال میانگین» "
            "میانگین احتمال ماشینی است که مدل به متن حمله‌شده می‌دهد",
            "tab:v2-attack-methods",
            "lrrrr",
            ["روش بازنویسی", "ردیف", "احتمال میانگین", "بازیابی", "افت"],
            method_rows,
        )
    return body


def build_tab_three_class(results: dict | None) -> str:
    if not results or not condition_record(results, "three_class"):
        return placeholder("tab:v2-three-class", "بدون شرط three_class")
    if (condition_record(results, "three_class") or {}).get("skipped"):
        return placeholder("tab:v2-three-class", "ردیف بازنویسی‌شده موجود نیست")
    rows = []
    for model, label in (
        ("tfidf_logreg_3class", "\\en{TF-IDF} تنها"),
        ("tfidf_style_logreg_3class", "\\en{TF-IDF} + سبک"),
    ):
        if not three_class_records(results, model):
            continue
        rows.append(
            [
                label,
                percent_list(three_class_metric(results, model, "macro_f1")),
                percent(mean_of(three_class_metric(results, model, "f1_human"))),
                percent(mean_of(three_class_metric(results, model, "f1_ai"))),
                percent(mean_of(three_class_metric(results, model, "f1_ai_revised"))),
                percent(mean_of(three_class_metric(results, model, "recall_ai_revised"))),
                number_list(three_class_metric(results, model, "macro_roc_auc_ovr"), 3),
                percent(mean_of(three_class_metric(results, model, "macro_f1", block="binary_collapse"))),
            ]
        )
    if not rows:
        return placeholder("tab:v2-three-class", "مدلی در شرط three_class ثبت نشده است")
    first = three_class_records(results, "tfidf_logreg_3class") or three_class_records(
        results, "tfidf_style_logreg_3class"
    )
    counts = first[0].get("test_class_counts", {}) if first else {}
    mix = "، ".join(
        f"{LABEL_LABELS.get(k, k)}: ${num(v)}$" for k, v in sorted(counts.items())
    )
    note = f"ترکیب کلاس‌ها در آزمون: {mix}." if mix else ""
    return table(
        "نمای سه‌کلاسه (انسان / ماشین خام / بازنویسی‌شده) و فروکاست دودویی آن",
        "tab:v2-three-class",
        "lccccccc",
        [
            "مدل",
            "$F_1$ ماکرو",
            "$F_1$ انسان",
            "$F_1$ ماشین",
            "$F_1$ بازنویسی",
            "بازیابی بازنویسی",
            "\\en{ROC-AUC} ماکرو",
            "$F_1$ دودویی",
        ],
        rows,
    ) + ("\n\n" + note if note else "")


def build_tab_errors(errors: dict | None) -> str:
    if not errors:
        return placeholder("tab:v2-errors", "بدون reports/error_analysis/summary.json")
    rows = []
    for direction, label in (("fp", "مثبت کاذب"), ("fn", "منفی کاذب")):
        block = errors.get(direction) or {}
        reviewed = int(block.get("reviewed_rows", 0))
        for bucket, count in sorted(
            (block.get("taxonomy_mix") or {}).items(), key=lambda item: -item[1]
        ):
            rows.append(
                [
                    label,
                    TAXONOMY_LABELS.get(bucket, latin(bucket)),
                    f"${num(count)}$",
                    share(count, reviewed),
                ]
            )
    if not rows:
        return placeholder("tab:v2-errors", "طبقه‌بندی خطاها خالی است")
    return table(
        "طبقه‌بندی بازبینی خطاها روی مدل \\texttt{\\lr{" + str(errors.get("model", "?")) + "}} "
        "(سهم‌ها نسبت به ردیف‌های بازبینی‌شدهٔ همان جهت است)",
        "tab:v2-errors",
        "llrr",
        ["جهت", "طبقه", "ردیف", "سهم"],
        rows,
        font="small",
    )


# ---------------------------------------------------------------------------
# macros
# ---------------------------------------------------------------------------


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build_macros(
    manifest: dict | None,
    splits: dict | None,
    results: dict | None,
    errors: dict | None = None,
) -> str:
    if not manifest:
        raise SystemExit("dataset manifest is required (run training/build_dataset_v2.py)")
    rows = manifest["rows"]
    label_counts = manifest["label_counts"]
    dedup = manifest["dedup"]
    topics = manifest["topic_group_counts"]
    lines = [
        "% گردآوری خودکار از data/dataset_v2.manifest.json و data/splits_v2.json",
        "% با training/report_tables.py -- این فایل را دستی ویرایش نکنید.",
        f"\\newcommand{{\\dtRows}}{{{num(rows)}}}",
        f"\\newcommand{{\\dtHuman}}{{{num(label_counts.get('human', 0))}}}",
        f"\\newcommand{{\\dtAI}}{{{num(label_counts.get('ai', 0))}}}",
        f"\\newcommand{{\\dtRevised}}{{{num(label_counts.get('ai_revised', 0))}}}",
        f"\\newcommand{{\\dtGenerators}}{{{num(manifest.get('n_generators', 0))}}}",
        f"\\newcommand{{\\dtTopics}}{{{num(topics.get('ai_unique_topics', 0))}}}",
        f"\\newcommand{{\\dtHumanClusters}}{{{num(topics.get('human_clusters', 0))}}}",
        f"\\newcommand{{\\dtStyles}}{{{num(len(manifest.get('prompt_style_counts', {})))}}}",
        f"\\newcommand{{\\dtDomains}}{{{num(sum(1 for d in manifest['label_domain_counts'] if d != 'ai_multi'))}}}",
        f"\\newcommand{{\\dtSubtitleRows}}{{{num(manifest['label_domain_counts'].get('translated_subtitles', {}).get('human', 0))}}}",
        f"\\newcommand{{\\dupInput}}{{{num(dedup.get('input_rows', 0))}}}",
        f"\\newcommand{{\\dupExact}}{{{num(dedup.get('exact_duplicates_removed', 0))}}}",
        f"\\newcommand{{\\dupNear}}{{{num(dedup.get('near_duplicates_removed', 0))}}}",
        f"\\newcommand{{\\dupCross}}{{{num(dedup.get('cross_label_near_duplicates_removed', 0))}}}",
    ]
    counts = (splits or {}).get("counts", {})
    for macro, key in (
        ("spTrain", "in_domain_group__train"),
        ("spVal", "in_domain_group__val"),
        ("spTest", "in_domain_group__test"),
        ("spVoneTest", "v1_test"),
        ("spCrossGenTest", "cross_generator__test"),
        ("spAttackReserve", "paraphrase_attention__test"),
        ("spCrossGenDropped", "rows_dropped"),
    ):
        if macro == "spCrossGenDropped":
            value = (
                (splits or {}).get("details", {}).get("cross_generator", {}).get("rows_dropped_unseen_generator_on_train_topic")
            )
        else:
            value = counts.get(key)
        lines.append(f"\\newcommand{{\\{macro}}}{{{num(value) if value is not None else '?'}}}")
    unseen = (splits or {}).get("details", {}).get("cross_generator", {}).get("n_generators")
    lines.append(f"\\newcommand{{\\spCrossGenUnseenGen}}{{{num(unseen) if unseen else '?'}}}")
    seeds = results.get("seeds") if results else None
    lines.append(f"\\newcommand{{\\spSeeds}}{{{num(len(seeds)) if seeds else num(len(DEFAULT_SEEDS))}}}")
    lines.extend(build_result_macros(results, errors))
    return "\n".join(lines)


def build_result_macros(results: dict | None, errors: dict | None) -> list[str]:
    """Inline numeric macros for the results prose.

    Every macro is always defined -- with an em dash when the underlying run has
    not happened yet -- so the report keeps compiling while the pipeline is
    still being filled in, and turns into real numbers on the next regeneration.
    """
    res = results or {}
    ident = "in_domain_group"
    lines: list[str] = ["% --- ارقام نتایج (reports/eval_results.json) ---"]
    for model, suffix in MODEL_SUFFIXES.items():
        lines.append(macro(f"resFone{suffix}", percent_list(metric_values(res, ident, model, "macro_f1"))))
    lines.append(macro("resRocSvm", number_list(metric_values(res, ident, "tfidf_svm", "roc_auc"))))
    lines.append(macro("resPrSvm", number_list(metric_values(res, ident, "tfidf_svm", "pr_auc"))))
    lines.append(
        macro("resFprSvm", percent(mean_of(metric_values(res, ident, "tfidf_svm", "fpr_at_tpr95")), 2))
    )
    lines.append(
        macro("resEceSvm", percent(mean_of(metric_values(res, ident, "tfidf_svm", "ece_width")), 2))
    )
    lines.append(
        macro(
            "resEceRawSvm",
            percent(
                mean_of(
                    metric_values(
                        res, ident, "tfidf_svm", "ece_width", source="raw_uncalibrated"
                    )
                ),
                2,
            ),
        )
    )
    lines.append(
        macro(
            "resVoneDiffSvm",
            diff_percent(
                metric_values(res, ident, "tfidf_svm", "macro_f1", split="v1_test"),
                metric_values(res, ident, "tfidf_svm", "macro_f1"),
            ),
        )
    )
    lines.append(
        macro(
            "resEceBest",
            percent(
                min(
                    (
                        mean_of(metric_values(res, ident, model, "ece_width"))
                        for model in MODEL_ORDER
                        if metric_values(res, ident, model, "ece_width")
                    ),
                    default=None,
                ),
                2,
            ),
        )
    )
    for key, name in (
        ("deferred_share", "resReviewSvm"),
        ("share_auto_human", "resAutoHumanSvm"),
        ("ai_called_ai_automatically", "resAutoAiSvm"),
        ("human_called_ai_automatically", "resHumanCalledAiSvm"),
        ("review_threshold", "resReviewThrSvm"),
        ("auto_threshold", "resAutoThrSvm"),
    ):
        lines.append(macro(name, percent(mean_of(policy_values(res, ident, "tfidf_svm", key)), 2)))
    for model, name in (("tfidf_svm", "resRevisedDetect"), ("tfidf_logreg", "resRevisedDetectLogreg")):
        lines.append(
            macro(
                name,
                percent(
                    mean_of(
                        metric_values(
                            res, ident, model, "recall_ai", split="revision_sensitivity__all"
                        )
                    )
                ),
            )
        )
    lines.append(
        macro(
            "resCrossGenSvm",
            percent_list(metric_values(res, "cross_generator", "tfidf_svm", "macro_f1")),
        )
    )
    for condition, suffix in CROSS_DOMAIN_SUFFIX.items():
        if condition_record(res, condition):
            lines.append(
                macro(
                    f"resDomain{suffix}",
                    percent_list(metric_values(res, condition, "tfidf_svm", "macro_f1")),
                )
            )
    worst = worst_condition(res, list(CROSS_DOMAIN_SUFFIX))
    lines.append(macro("resWorstDomainName", CONDITION_LABELS.get(worst[0], "—") if worst else "—"))
    lines.append(macro("resWorstDomainFone", percent_list(worst[1]) if worst else "—"))
    for condition, suffix in CROSS_STYLE_SUFFIX.items():
        if condition_record(res, condition):
            lines.append(
                macro(
                    f"resStyle{suffix}",
                    percent_list(metric_values(res, condition, "tfidf_svm", "macro_f1")),
                )
            )
    worst_style = worst_condition(res, list(CROSS_STYLE_SUFFIX))
    lines.append(macro("resWorstStyleName", CONDITION_LABELS.get(worst_style[0], "—") if worst_style else "—"))
    lines.append(macro("resWorstStyleFone", percent_list(worst_style[1]) if worst_style else "—"))
    for condition, name, split in (
        ("few_shot__8", "resFewSmallSvm", "few_shot__8__test"),
        ("few_shot__32", "resFewMediumSvm", "few_shot__32__test"),
        ("few_shot__128", "resFewLargeSvm", "few_shot__128__test"),
        ("length__short", "resLenShortSvm", "length__short__test"),
        ("length__long", "resLenLongSvm", "length__long__test"),
        ("v1_test", "resVoneSvm", "v1_test"),
    ):
        # the length buckets and the v1-comparable rows are extra score sets of
        # the in-domain condition; the few-shot rows are their own condition
        source = ident if split in ("v1_test", "length__short__test", "length__long__test") else condition
        lines.append(
            macro(
                name,
                percent_list(metric_values(res, source, "tfidf_svm", "macro_f1", split=split)),
            )
        )
    for model, suffix in (("tfidf_svm", "Svm"), ("tfidf_logreg", "Logreg")):
        paired = attack_paired(res, model)
        lines.append(macro(f"resAttackClean{suffix}", percent(mean_of(attack_values(res, model, "clean_recall_at_0.5")))))
        lines.append(macro(f"resAttackRec{suffix}", percent(mean_of(attack_values(res, model, "attack_recall_at_0.5")))))
        lines.append(macro(f"resAttackDrop{suffix}", percent(mean_of([p["mean_difference"] for p in paired]))))
        lo = mean_of([p["lo"] for p in paired])
        hi = mean_of([p["hi"] for p in paired])
        lines.append(
            macro(
                f"resAttackDropCi{suffix}",
                f"[${100.0 * lo:.1f}$, ${100.0 * hi:.1f}$]" if lo is not None and hi is not None else "—",
            )
        )
    method_drops: dict[str, list[float]] = {}
    for model in MODEL_ORDER:
        for record in model_records(res, "paraphrase_attack", model):
            for method, block in (record.get("by_method") or {}).items():
                value = finite(block.get("recall_drop_vs_clean"))
                if value is not None:
                    method_drops.setdefault(method, []).append(value)
    if method_drops:
        worst_method = max(method_drops.items(), key=lambda item: statistics.fmean(item[1]))
        lines.append(
            macro("resAttackWorstMethod", ATTACK_METHOD_LABELS.get(worst_method[0], latin(worst_method[0])))
        )
        lines.append(macro("resAttackWorstDrop", percent(statistics.fmean(worst_method[1]))))
    else:
        lines.append(macro("resAttackWorstMethod", "—"))
        lines.append(macro("resAttackWorstDrop", "—"))
    for model, name in (
        ("tfidf_logreg_3class", "resThreeFone"),
        ("tfidf_style_logreg_3class", "resThreeStyleFone"),
    ):
        lines.append(macro(name, percent_list(three_class_metric(res, model, "macro_f1"))))
    lines.append(macro("resThreeRevisedFone", percent(mean_of(three_class_metric(res, "tfidf_logreg_3class", "f1_ai_revised")))))
    lines.append(macro("resThreeRevisedRecall", percent(mean_of(three_class_metric(res, "tfidf_logreg_3class", "recall_ai_revised")))))
    lines.append(macro("resThreeHumanFone", percent(mean_of(three_class_metric(res, "tfidf_logreg_3class", "f1_human")))))
    lines.append(macro("resThreeAiFone", percent(mean_of(three_class_metric(res, "tfidf_logreg_3class", "f1_ai")))))
    lines.append(
        macro(
            "resThreeBinaryFone",
            percent(mean_of(three_class_metric(res, "tfidf_logreg_3class", "macro_f1", block="binary_collapse"))),
        )
    )
    lines.append("% --- ارقام بازبینی خطاها (reports/error_analysis/summary.json) ---")
    err = errors or {}
    attack_record = condition_record(res, "paraphrase_attack") or {}
    lines.append(
        macro(
            "resAttackRows",
            f"${num(attack_record['n_attacked'])}$" if attack_record.get("n_attacked") else "—",
        )
    )
    lines.append(
        macro(
            "resAttackSources",
            f"${num(attack_record['n_clean_sources'])}$"
            if attack_record.get("n_clean_sources")
            else "—",
        )
    )
    lines.append(macro("errModel", latin(str(err.get("model", "—")))))
    for direction, name in (("fp", "Fp"), ("fn", "Fn")):
        block = err.get(direction) or {}
        if block:
            lines.append(macro(f"err{name}Rows", f"${num(block.get('reviewed_rows', 0))}$"))
            lines.append(
                macro(f"err{name}AtHalf", f"${num(block.get('above_decision_point', 0))}$")
            )
        else:
            lines.append(macro(f"err{name}Rows", "—"))
            lines.append(macro(f"err{name}AtHalf", "—"))
        mix = block.get("taxonomy_mix") or {}
        if mix:
            top = max(mix.items(), key=lambda item: item[1])
            lines.append(macro(f"err{name}TopBucket", TAXONOMY_LABELS.get(top[0], latin(top[0]))))
            lines.append(macro(f"err{name}TopShare", share(top[1], int(block.get("reviewed_rows", 1)))))
        else:
            lines.append(macro(f"err{name}TopBucket", "—"))
            lines.append(macro(f"err{name}TopShare", "—"))
    return lines


def worst_condition(results: dict | None, names: list[str]) -> tuple[str, list[float]] | None:
    """Condition with the lowest mean macro-F1 for the baseline model."""
    scored = []
    for name in names:
        values = metric_values(results, name, "tfidf_svm", "macro_f1")
        if values:
            scored.append((name, values))
    if not scored:
        return None
    return min(scored, key=lambda item: statistics.fmean(item[1]))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def build_tab_dataset(manifest: dict | None) -> str:
    if not manifest:
        return placeholder("tab:v2-dataset", "بدون manifest داده")
    rows = manifest["rows"]
    counts = manifest["label_counts"]
    body_rows = []
    for key in ("human", "ai", "ai_revised"):
        body_rows.append([LABEL_LABELS[key], f"${num(counts.get(key, 0))}$", share(counts.get(key, 0), rows)])
    body_rows.append(["مجموع", f"${num(rows)}$", "$100.0\\%$"])
    labels = table(
        "ترکیب قاب دادهٔ نسخهٔ دوم (پس از حذف تکراری‌ها)",
        "tab:v2-dataset",
        "lrr",
        ["برچسب", "ردیف", "سهم"],
        body_rows,
    )
    mix_rows = []
    for key, group_label in (
        ("length_bucket_counts", "طول"),
        ("formality_counts", "فرمالیته"),
        ("origin_counts", "اصلِ زبان"),
    ):
        for value, count in manifest.get(key, {}).items():
            mix_rows.append(
                [group_label, BUCKET_LABELS.get(value, value), f"${num(count)}$", share(count, rows)]
            )
    mix = table(
        "ترکیب طول، فرمالیته و اصلِ زبان در قاب دادهٔ نسخهٔ دوم",
        "tab:v2-mix",
        "llrr",
        ["سنجه", "مقدار", "ردیف", "سهم"],
        mix_rows,
    )
    return labels + "\n\n" + mix


def build_tab_domains(manifest: dict | None, provenance: dict | None) -> str:
    if not manifest:
        return placeholder("tab:v2-provenance", "بدون manifest داده")
    provenance = provenance or {}
    rows = []
    for domain, label in DOMAIN_LABELS.items():
        if domain not in manifest["label_domain_counts"] or domain == "ai_multi":
            continue
        record = provenance.get(domain, {})
        n_rows = manifest["label_domain_counts"][domain].get("human", 0)
        license_id = record.get("license", "")
        license_label = (
            UNLICENSED_LABEL if license_id == "see project README" else latin(license_id or "?")
        )
        origin = "ترجمه‌ای" if record.get("origin") == "translated" else "فارسی"
        rows.append(
            [
                label,
                latin(SOURCE_LABELS.get(domain, record.get("source", "—"))),
                license_label,
                origin,
                f"${num(n_rows)}$",
            ]
        )
    return table(
        "دامنه‌های انسانی: منبع، مجوز و شمار ردیف‌های حاضر در قاب داده "
        "(شناسهٔ کامل منابع و درهم‌ساز فایل‌ها در \\texttt{\\lr{data/human/provenance.json}})",
        "tab:v2-provenance",
        "llllr",
        ["دامنه", "منبع", "مجوز", "اصل", "ردیف"],
        rows,
        font="small",
    )


def build_tab_splits(splits: dict | None) -> str:
    if not splits:
        return placeholder("tab:v2-splits", "بدون data/splits_v2.json")
    counts = splits.get("counts", {})
    rows = []
    for name, label in SPLIT_ROWS:
        cells = [label]
        for part in ("train", "val", "test"):
            # most conditions are stored as <name>__<part>; a few (v1_test,
            # paraphrase_attention) are a single, unsplit test set
            value = counts.get(f"{name}__{part}")
            if value is None and part == "test":
                value = counts.get(name)
            cells.append(f"${num(value)}$" if value is not None else "—")
        rows.append(cells)
    integrity = splits.get("integrity", {})
    ok = all(v for k, v in integrity.items() if isinstance(v, bool))
    note = (
        "همهٔ گزاره‌های سلامت در \\texttt{\\lr{splits\\_v2.json}} برقرارند."
        if ok
        else "\\textbf{هشدار: دست‌کم یک گزارهٔ سلامت برقرار نیست.}"
    )
    body = table(
        "شرط‌های ارزیابی و اندازهٔ بخش‌های هر شرط (ردیف)",
        "tab:v2-splits",
        "lrrr",
        ["شرط", "آموزش", "اعتبارسنجی", "آزمون"],
        rows,
    )
    return body + "\n\n" + note


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def check_macros_defined(definitions: str) -> None:
    """Every prefix macro used in report.tex must exist, or xelatex fails."""
    if not REPORT_TEX.exists():
        return
    pattern = r"\\((?:dt|dup|err)[A-Za-z]+|(?:sp|res)[A-Z][A-Za-z]*)"
    used = set(re.findall(pattern, REPORT_TEX.read_text(encoding="utf-8")))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", definitions))
    missing = sorted(used - defined)
    if missing:
        raise SystemExit(f"report.tex uses undefined macros: {missing}")


def main() -> None:
    manifest = load_json(MANIFEST)
    provenance = load_json(HUMAN_PROVENANCE)
    splits = load_json(SPLITS_JSON)
    results = load_json(RESULTS_JSON)
    errors = load_json(ERROR_SUMMARY)
    if errors is None:
        print("note: no error-analysis summary yet; the error table stays a placeholder")

    definitions = build_macros(manifest, splits, results, errors)
    check_macros_defined(definitions)
    write("macros.tex", definitions)
    write("tab_dataset.tex", build_tab_dataset(manifest))
    write("tab_domains.tex", build_tab_domains(manifest, provenance))
    write("tab_splits.tex", build_tab_splits(splits))
    write("tab_results.tex", build_tab_results(results))
    write("tab_conditions.tex", build_tab_conditions(results, splits))
    write("tab_attack.tex", build_tab_attack(results))
    write("tab_three_class.tex", build_tab_three_class(results))
    write("tab_errors.tex", build_tab_errors(errors))
    print("done")


if __name__ == "__main__":
    main()

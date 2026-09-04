"""
chatbot.py

The AI Compliance Assistant -- a Groq-powered chat interface that answers
questions about the model, its metrics, and (when a merchant is currently
loaded in the UI) that specific merchant's assessment.

This is the layer that turns a static dashboard into something a judge can
genuinely interrogate live during a panel demo: "why is precision only 26%?",
"what would happen if we reviewed 25% of accounts instead of 15%?", "explain
SHAP to me like I'm not a data scientist."

Design choice: the assistant is given the REAL model metrics and (if present)
the REAL current merchant context as system-prompt facts, and is instructed
to answer only from those facts -- not to invent numbers. This matters for a
compliance tool: an assistant that hallucinates risk figures would be worse
than no assistant at all.
"""

import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

CHAT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT_TEMPLATE = """You are Blade, the AI Compliance Assistant built into Freeze-Risk Guardian, a tool that \
predicts compliance-driven merchant account freezes for a payments platform (built for a hackathon, \
Track 02: AI Risk Manager). Your name nods to the razor-blade symbol in Razorpay's own logo -- sharp, \
precise, no wasted motion. Match that in your tone: direct, confident, no filler, no hedging when the \
facts support a clear answer.

Answer questions about the model, its methodology, and the current merchant (if given) using \
ONLY the facts below. Never invent numbers that aren't provided. If asked something the facts \
don't cover, say so plainly rather than guessing.

MODEL FACTS:
- Model type: Random Forest classifier, 300 trees, trained on 6,000 synthetic merchant snapshots
- Deployed decision policy: capacity-constrained -- flags only the riskiest 15% of accounts per \
review cycle (chosen over a naive cost-minimization threshold, which was tested and rejected \
because it flagged ~98% of accounts -- not operationally reviewable)
- Held-out test set performance: precision {precision:.1%}, recall {recall:.1%}, F1 {f1:.3f}, \
ROC-AUC {roc_auc}
- Confusion matrix on test set: {tp} true positives, {fp} false positives, {fn} false negatives, \
{tn} true negatives
- Cost assumptions: a false positive (wasted compliance review) costs an estimated ₹150; a false \
negative (missed freeze) costs an estimated ₹45,000 -- roughly a 300x asymmetry, which is why \
recall is prioritized over precision within the review-capacity budget
- Estimated savings on test set vs. flagging nobody: ₹{savings:,}
- Explainability: per-merchant contributions computed via a z-score-weighted feature deviation \
method (SHAP-style local attribution), not just global feature importance -- so two merchants \
can show different top reasons for their score
- Fairness check: flag-rate-to-actual-risk ratios across segments range 1.0-1.83 on the test set \
(Geography mismatch and Medium international-share segments run somewhat higher, ~1.7-1.8x, with \
modest sample sizes of ~100-280 -- worth monitoring, not evidence of severe bias). No segment is \
flagged at a wildly disproportionate rate.
- Labels are RULE-BASED SYNTHETIC ground truth (built from documented public freeze triggers: \
KYC gaps, volume spikes, chargeback ratios, geography mismatch), not real Razorpay data, since \
real freeze data is private

DATASET-LEVEL STATS (for general questions not tied to one merchant):
{dataset_stats}

THRESHOLD SWEEP TABLE (what precision/recall/savings would be at other review-capacity levels \
-- use this to answer "what if" questions with REAL numbers instead of guessing):
{threshold_table}

{batch_context}

{merchant_context}

{tab_context}

If someone asks about a specific merchant but no merchant context is given above, tell them to \
select that merchant in the "Assess a Merchant" tab and click "Assess Risk" first -- you cannot \
answer merchant-specific questions without that.

Keep answers concise (2-4 sentences unless more detail is explicitly asked for). You may be \
asked by a hackathon judge, so be precise and honest about limitations -- do not oversell the \
model's accuracy."""


def _format_threshold_table(sweep_data):
    """Condense the full 91-point sweep into a few representative rows so the
    assistant can answer capacity-tradeoff questions with real numbers."""
    if not sweep_data:
        return "(not available)"
    sweep = sweep_data.get("sweep", [])
    if not sweep:
        return "(not available)"
    # Pick rows near common capacity levels by flag_rate
    targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    lines = []
    for target in targets:
        closest = min(sweep, key=lambda p: abs(p["flag_rate"] - target))
        lines.append(
            f"- ~{closest['flag_rate']:.0%} of accounts flagged (threshold {closest['threshold']}): "
            f"precision {closest['precision']:.1%}, recall {closest['recall']:.1%}, "
            f"savings ₹{closest['savings_vs_no_detection_inr']:,}"
        )
    return "\n".join(lines)


def _format_dataset_stats(dataset_stats):
    if not dataset_stats:
        return "(not available)"
    return (
        f"- Training/test population: {dataset_stats.get('n', 0):,} synthetic merchants\n"
        f"- Overall freeze-risk positive rate: {dataset_stats.get('positive_rate', 0):.1%}\n"
        f"- Average KYC completeness score: {dataset_stats.get('avg_kyc', 0):.1f}/100\n"
        f"- Geo-mismatch present in: {dataset_stats.get('geo_mismatch_rate', 0):.1%} of merchants\n"
        f"- High-risk category merchants: {dataset_stats.get('high_risk_rate', 0):.1%}"
    )


def build_system_prompt(metrics, current_merchant=None, threshold_sweep=None, last_batch=None,
                         dataset_stats=None, current_tab=None):
    merchant_context = "No specific merchant is currently loaded."
    if current_merchant:
        merchant_context = (
            f"CURRENTLY LOADED MERCHANT: {current_merchant.get('merchant_id', 'unknown')}, "
            f"risk score {current_merchant.get('risk_score', 0):.1%}, "
            f"flagged={current_merchant.get('flagged')}. "
            f"Top contributing factors: {current_merchant.get('top_factors_summary', 'n/a')}."
        )
    batch_context = ""
    if last_batch:
        batch_context = (
            f"MOST RECENT BATCH UPLOAD RESULT: {last_batch.get('total_merchants', 0)} merchants scored, "
            f"{last_batch.get('flagged_count', 0)} flagged "
            f"({last_batch.get('flag_rate', 0):.1%}), average risk score "
            f"{last_batch.get('avg_risk_score', 0):.1%}."
        )
    tab_context = f"The user is currently looking at the \"{current_tab}\" tab." if current_tab else ""
    cm = metrics.get("confusion_matrix", {})
    return SYSTEM_PROMPT_TEMPLATE.format(
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0),
        roc_auc=metrics.get("roc_auc", 0),
        tp=cm.get("TP", 0), fp=cm.get("FP", 0), fn=cm.get("FN", 0), tn=cm.get("TN", 0),
        savings=metrics.get("estimated_cost_inr", {}).get("estimated_savings_inr", 0),
        threshold_table=_format_threshold_table(threshold_sweep),
        dataset_stats=_format_dataset_stats(dataset_stats),
        batch_context=batch_context,
        merchant_context=merchant_context,
        tab_context=tab_context,
    )


def chat(message, history, metrics, current_merchant=None, threshold_sweep=None, last_batch=None,
         dataset_stats=None, current_tab=None, api_key=None):
    """
    message: the new user message (str)
    history: list of {"role": "user"|"assistant", "content": str} prior turns
    metrics: the model's metrics.json dict
    current_merchant: optional dict with the currently-assessed merchant's result
    threshold_sweep: optional dict (threshold_sweep.json content) for "what if" questions
    last_batch: optional dict with the most recent batch-assess summary
    dataset_stats: optional dict with population-level dataset statistics
    current_tab: optional str, which tab the user is currently viewing
    Returns: (reply_text, source) where source is "groq_llm" or "unavailable"
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key or not GROQ_AVAILABLE:
        return (
            "Blade needs a Groq API key to answer live questions. "
            "Set GROQ_API_KEY and restart the app -- see the README for a free key.",
            "unavailable",
        )

    system_prompt = build_system_prompt(
        metrics, current_merchant, threshold_sweep, last_batch, dataset_stats, current_tab
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-8:])  # keep last few turns only -- bounded context
    messages.append({"role": "user", "content": message})

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=450,
        )
        return response.choices[0].message.content.strip(), "groq_llm"
    except Exception as e:
        return f"Blade hit an error reaching Groq: {e}", "error"

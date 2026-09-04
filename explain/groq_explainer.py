"""
groq_explainer.py

Turns a model's risk score + top contributing features into:
    1. A plain-language explanation of WHY an account was flagged
    2. A prioritized remediation checklist the merchant can act on

This is the layer that makes the project "GenAI-powered," not just a bare
classifier -- and it's the layer Razorpay's own dashboard doesn't have today
(it shows THAT a hold happened, not WHY, and not what to fix before it does).

Uses the Groq API (same pattern as the AI Interview Bot project) because it
has a generous free tier -- good for a 3-day build with no budget.

Set your key as an environment variable before running:
    export GROQ_API_KEY="your_key_here"
Get a free key at https://console.groq.com
"""

import os
import json

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

MODEL_NAME = "openai/gpt-oss-20b"  # fast + free-tier friendly; check console.groq.com/docs/models if this changes again

SYSTEM_PROMPT_PLAIN = """You are a compliance risk analyst assistant for a payments platform.
Given a merchant's risk score and the factors driving it, write:
1. A 2-3 sentence plain-language explanation of why this account is flagged,
   written for a non-technical small business owner (not a data scientist).
2. A prioritized checklist (3-5 items) of concrete actions the merchant can
   take to reduce their risk before it becomes an account freeze.

Rules:
- Never claim the account WILL be frozen -- only that it is at elevated risk.
- Be specific to the factors given, not generic advice.
- Keep total output under 150 words.
- Return ONLY valid JSON with keys: "explanation" and "checklist" (a list of strings).
No markdown, no preamble, no code fences."""

SYSTEM_PROMPT_TECHNICAL = """You are a risk-model analyst writing for a compliance/data team, not a merchant.
Given a merchant's risk score and its SHAP-style feature contributions, write:
1. A 2-3 sentence technical explanation referencing the actual contribution
   values and feature names, suitable for a model audit log.
2. A prioritized checklist (3-5 items) of concrete remediation actions.

Rules:
- Never claim the account WILL be frozen -- only that it is at elevated risk.
- Reference actual numeric contribution values where relevant.
- Keep total output under 150 words.
- Return ONLY valid JSON with keys: "explanation" and "checklist" (a list of strings).
No markdown, no preamble, no code fences."""


def build_user_prompt(risk_score, top_factors, merchant_snapshot):
    factor_lines = "\n".join(
        f"- {f['feature']}: {f['direction']} (value: {f['value']}, contribution weight: {f['contribution']:+.3f})"
        for f in top_factors
    )
    return f"""Merchant risk score: {risk_score:.1%} (flagged as elevated risk)

Top contributing factors for THIS specific merchant:
{factor_lines}

Generate the explanation and checklist as specified."""


def get_explanation(risk_score, top_factors, merchant_snapshot, api_key=None, style="plain"):
    """
    risk_score: float 0-1, model's predicted probability
    top_factors: list of (feature_name, importance_weight) tuples, highest first
    merchant_snapshot: dict of the merchant's raw feature values (for context)
    style: "plain" (merchant-facing) or "technical" (compliance-team-facing)
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key or not GROQ_AVAILABLE:
        return _fallback_explanation(risk_score, top_factors)

    system_prompt = SYSTEM_PROMPT_TECHNICAL if style == "technical" else SYSTEM_PROMPT_PLAIN

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(risk_score, top_factors, merchant_snapshot)},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        parsed["source"] = "groq_llm"
        return parsed
    except Exception as e:
        fallback = _fallback_explanation(risk_score, top_factors)
        fallback["error"] = f"Groq call failed, used rule-based fallback: {e}"
        return fallback


def _fallback_explanation(risk_score, top_factors):
    """Deterministic backup if the API key is missing or the call fails --
    an app that breaks with no key is a bad look in a live panel demo."""
    increasing = [f for f in top_factors if f["contribution"] > 0][:3]
    top_names = [f["feature"].replace("_", " ") for f in increasing] or \
                [f["feature"].replace("_", " ") for f in top_factors[:3]]
    return {
        "explanation": (
            f"This account scores {risk_score:.0%} on freeze risk, driven mainly by "
            f"{', '.join(top_names)}. These factors match patterns commonly associated "
            f"with compliance-driven account holds."
        ),
        "checklist": [
            f"Review and address: {top_names[0]}" if top_names else "Review KYC documentation completeness",
            "Ensure KYC documents are current and match declared business details",
            "Flag any recent unusual volume or geography changes to your account manager proactively",
        ],
        "source": "rule_based_fallback",
    }

# Freeze-Risk Guardian
**Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**

Predicting compliance-driven merchant account freezes and reserve holds *before* they happen — and explaining, in plain language, what a merchant needs to fix.

## The problem (and why it's real)

Razorpay's own published guidance and independent 2026 merchant reviews agree on the same thing: the biggest operational pain for merchants isn't fraud loss — it's unexpected account freezes and rolling reserves triggered by compliance reviews (incomplete KYC, sudden volume spikes, transaction patterns inconsistent with a merchant's declared profile). Merchant surveys report that unexpected holds have a severe cash-flow impact for a large share of businesses. Today, Razorpay's dashboard tells a merchant *that* a hold happened — nothing predicts it, or explains what to fix beforehand.

Razorpay's existing AI stack (Vulcan, chargeback-response agents, subscription-recovery agents) already covers classic fraud and churn. This does not. That's the gap this project targets — a class of "loss" (locked-up cash flow, compliance friction) distinct from the fraud/chargeback examples Track 02 lists, but squarely inside its "stop the merchant losing money" mandate.

## What it does

1. **Detector** — a Random Forest classifier scores a merchant snapshot's probability of triggering a compliance hold, trained on synthetic data whose labeling logic is built directly from documented, public freeze triggers (see `data/generate_synthetic_data.py` for the exact rules and sourcing rationale).
2. **Decision policy** — rather than an arbitrary 0.5 cutoff, the deployed threshold is chosen under a **review-capacity constraint** (flag only the riskiest ~15% of accounts a compliance team could plausibly review per cycle), not a naive cost-minimization formula. See "What broke" below — this was a real design pivot, not an assumption made upfront. A **live interactive slider** in the dashboard lets you drag the threshold and watch precision/recall/cost/confusion-matrix update instantly (precomputed as a threshold-sweep table so it's not calling the model on every drag).
3. **Local, per-merchant explainability** — every flagged merchant gets a **SHAP-based explanation specific to their own numbers**, not a static global "top factors" list. Two merchants can be flagged for entirely different reasons and the dashboard shows that. An LLM (Groq API) turns those SHAP contributions into a plain-language explanation and a prioritized remediation checklist, with a deterministic rule-based fallback if the API key is missing or the call fails.
4. **Model performance view** — precision/recall/F1/ROC-AUC plus a live ROC curve and confusion matrix rendered directly in the dashboard, not buried in a JSON file.
5. **Batch portfolio assessment** — upload a CSV of many merchants at once and get a portfolio-level risk distribution and a ranked list, closer to how a real compliance team would actually use this than one-at-a-time scoring. A sample CSV is one click away for demos.
6. **Audit trail** — every assessment is logged (timestamp, inputs, score, threshold, explanation source) to `logs/audit_log.jsonl`, viewable in the app.
7. **AI Compliance Assistant** — a Groq-powered chat widget (bottom-right, any tab) that answers questions about the model, its metrics, the currently-loaded merchant, the full threshold-sweep table (so "what if we reviewed 25% instead of 15%?" gets a real number, not a guess), and the most recent batch upload — all grounded in actual data, never invented.
8. **What-If Simulator** — drag sliders for any of the 12 features and watch the risk score update live, before even clicking "Assess."
9. **Compare Merchants** — score two merchants side by side to contrast a low-risk and high-risk profile directly.
10. **Downloadable report** — export any assessment as a plain-text compliance readiness report.
11. **Plain-English / Technical explanation toggle** — switch the LLM explanation between merchant-facing language and compliance-team technical language (references actual contribution values).
12. **Risk grade badge (A–F)** — quick-read letter grade alongside the precise percentage.
13. **Fairness check** — segment-level analysis on the held-out test set comparing flag rate to actual freeze-risk rate across merchant category, geo-mismatch, and international-transaction-share segments, with a flag-rate-to-actual-risk ratio per group so over- or under-targeting is visible, not assumed away.

## Architecture

```
data/generate_synthetic_data.py  -> merchant_freeze_risk_dataset.csv
model/train_model.py             -> freeze_risk_model.joblib, metrics.json, roc_curve.json, threshold_sweep.json
explain/local_explain.py         -> per-merchant SHAP explanation (with z-score fallback)
explain/groq_explainer.py        -> LLM plain-language explanation layer (with rule-based fallback)
app/app.py                       -> Flask API + dashboard (assess, batch, threshold sweep, ROC, audit)
app/templates/index.html         -> tabbed single-page demo UI
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python3 data/generate_synthetic_data.py

# 2. Train the model and produce metrics.json
python3 model/train_model.py

# 3. (Optional but recommended) Add a free Groq API key for live explanations
export GROQ_API_KEY="your_key_here"   # https://console.groq.com

# 4. Run the app
python3 app/app.py
# open http://localhost:5000
```

Works without a Groq key too — falls back to a deterministic rule-based explanation so the demo never breaks live.

## Model results (held-out test set, see `model/metrics.json` for the full run)

- Deployed policy: flag the riskiest ~15% of accounts per review cycle (capacity-constrained, not accuracy-optimized)
- Precision ~25%, Recall ~37% at that operating point
- Estimated savings vs. flagging nobody: **~₹25 lakh** on the test set alone, under the stated cost assumptions
- A naive "minimize raw cost" threshold was also computed and *rejected* — it flagged ~98% of all accounts, which is not operationally reviewable. This tradeoff is documented in `model/train_model.py` and is one of the strongest technical talking points for the panel.

**Be honest about this in the pitch**: precision and recall here are modest, and that's the correct, honest answer for a genuinely noisy risk signal on synthetic data — not a number to inflate. The panel explicitly said they want honest metrics, not a demo that "just looks good."

## Known limitations (say these out loud, don't wait to be asked)

- Labels are **rule-based synthetic ground truth**, not real Razorpay freeze data (which is private). The pipeline is built so real anonymized data could drop in directly if access were granted.
- Feature set (12 fields) is a reasonable proxy for documented freeze triggers, not an exhaustive one.
- Precision at ~25% means roughly 3 in 4 flagged accounts are false positives — acceptable given the ~300x cost asymmetry between a wasted review and a missed freeze, but worth stating plainly rather than hiding behind the recall number.

## What broke, and how it was fixed (for the "what went wrong" panel question)

1. **Naive cost-minimization threshold flagged 98% of accounts.** Early version chose the decision threshold by pure expected-cost minimization. Mathematically "optimal" given the assumed 300x cost asymmetry, but operationally absurd — no compliance team can review 98% of a merchant base. Fixed by switching to a review-capacity-constrained threshold (top-K% by risk score) — a more realistic model of how real risk teams actually operate. The dashboard's interactive slider now lets a panel see this tradeoff for themselves instead of taking the claim on faith.
2. **App crashed on missing `groq` package/API key.** First version hard-imported the Groq client and had no fallback — meaning a live demo with no internet or an unset key would crash mid-pitch. Fixed with a try/except import guard and a deterministic rule-based fallback explanation, so the app degrades gracefully instead of failing.
3. **Random sample merchants sometimes gave a boring demo** (no clearly-flagged case in the random sample shown). Fixed by explicitly surfacing the top-3 and bottom-3 scoring merchants for the walkthrough, rather than a random draw.
4. **Global feature importance looked the same for every merchant.** The first explainability pass showed identical "top factors" regardless of which account was being assessed — which isn't really an explanation, it's a static list. Replaced with per-merchant SHAP contributions (TreeExplainer), so two flagged merchants now visibly show different reasons, with a z-score-weighted fallback if SHAP isn't available in the runtime.

## Suggested 5-minute pitch structure

1. (30s) The real problem — cite the documented pain point, not a hypothetical
2. (30s) Why it's a gap, not already solved (contrast with Vulcan/chargeback agent/subscription recovery)
3. (90s) Live demo — flag a risky merchant, show the SHAP explanation + checklist, switch to a safe merchant to show the contrast, show the audit log
4. (60s) The threshold-selection tradeoff — drag the live slider from the deployed 15%-capacity point down toward the naive cost-optimal point and watch the confusion matrix explode. This is your strongest technical-depth moment
5. (45s) Batch upload — score a portfolio of 25 merchants at once, show the risk distribution
6. (45s) Limitations, honestly stated, and what real data access would unlock
7. (30s) What broke and how you fixed it

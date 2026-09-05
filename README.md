[README-2.md](https://github.com/user-attachments/files/31869283/README-2.md)
# Freeze-Risk Guardian

**Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**

Predicting compliance-driven merchant account freezes and reserve holds *before* they happen — and explaining, in plain language, what a merchant needs to fix.

🔗 **Live demo:** [Add your Render URL here]
🎥 **Video walkthrough:** [Add your video link here]

---

## The problem (and why it's real)

Razorpay's own published guidance and independent 2026 merchant reviews agree on the same thing: the biggest operational pain for merchants isn't fraud loss — it's unexpected account freezes and rolling reserves triggered by compliance reviews (incomplete KYC, sudden volume spikes, transaction patterns inconsistent with a merchant's declared profile). Merchant surveys report that unexpected holds have a severe cash-flow impact for a large share of businesses. Today, a dashboard tells a merchant *that* a hold happened — nothing predicts it, or explains what to fix beforehand.

Razorpay's existing AI stack (fraud detection, chargeback-response agents, subscription-recovery agents) already covers classic fraud and churn. It doesn't cover this. That's the gap this project targets — a class of loss (locked-up cash flow from compliance friction) distinct from fraud/chargeback examples, but squarely inside the "stop the merchant losing money" mandate of Track 02.

---

## What it does

1. **Detector** — a Random Forest classifier (300 trees, scikit-learn) scores a merchant snapshot's probability of triggering a compliance hold, trained on synthetic data whose labeling logic is built directly from documented, public freeze triggers (KYC completeness, geo mismatch, chargeback ratio, refund ratio, volume spikes, prior risk flags, etc.).
2. **Explainer** — for every flagged merchant, the system generates a signed, per-feature local explanation (which factors pushed risk up, which pulled it down, and by how much), plus a plain-language summary and recommended next actions. Explanations are generated via an LLM (Groq) with a deterministic rule-based fallback if the LLM call fails or times out.
3. **Simulator** — a What-If tool lets a user drag feature sliders and see the risk band update live, without needing a full re-assessment call.
4. **Comparator** — side-by-side comparison of two merchants across the same feature set.
5. **Batch scorer** — CSV upload to score an entire merchant portfolio in one pass, reusing the same scoring endpoint as the single-merchant assessment.
6. **Governance panels** — model performance (ROC-AUC, precision/recall, confusion matrix), a live threshold tradeoff simulator, a segment-level fairness audit, a business impact projector, and a full audit trail of every prediction made.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                    │
│   HTML / CSS / JS — sidebar navigation, live sliders,        │
│   dropdowns, charts (ROC curve, confusion matrix, etc.)      │
└───────────────────────────┬───────────────────────────────────┘
                             │  fetch() calls to relative API routes
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask Application (app/)                 │
│                                                               │
│  Routes:                                                      │
│   /                     → renders dashboard                  │
│   /api/assess           → single merchant risk scoring       │
│   /api/simulate         → live what-if scoring (no logging)  │
│   /api/compare          → two-merchant side-by-side           │
│   /api/batch-assess     → CSV upload → bulk scoring           │
│   /api/model-metrics    → performance stats for dashboard     │
│   /api/fairness         → segment-level fairness stats        │
│   /api/business-impact  → savings projection calculator       │
│   /api/audit-trail      → prediction log retrieval             │
│                                                               │
└──────┬───────────────────────────┬────────────────────────────┘
       │                           │
       ▼                           ▼
┌───────────────┐         ┌─────────────────────────┐
│  model/        │         │  explain/                │
│  risk_model    │         │  - Groq LLM call for     │
│  .pkl          │         │    natural-language      │
│  (Random       │         │    explanation           │
│  Forest,       │         │  - rule_based_fallback   │
│  300 trees)    │         │    if LLM call fails      │
└───────────────┘         └─────────────────────────┘
       │
       ▼
┌───────────────┐
│  data/         │  6,000 synthetic merchant snapshots
│                │  (training + held-out test set: 1,500 accounts)
└───────────────┘
       │
       ▼
┌───────────────┐
│  logs/         │  audit_trail.jsonl — every prediction logged with
│                │  input features, flag decision, explanation source,
│                │  timestamp
└───────────────┘
```

---

## Tech stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| Backend            | Flask (Python)                                |
| Model              | scikit-learn — Random Forest, 300 trees        |
| Explanation engine | Groq LLM API + rule-based fallback             |
| Deployment         | Render (Gunicorn WSGI server)                  |
| Frontend           | HTML / CSS / vanilla JS                        |
| Data               | 6,000 synthetic merchant snapshots (custom generator) |

---

## Model details

- **Algorithm:** Random Forest Classifier, 300 estimators
- **Training data:** 6,000 synthetic merchant snapshots. Real compliance-freeze data isn't publicly available, so feature correlations (KYC completeness ↔ risk, geo mismatch ↔ risk, chargeback ratio ↔ risk, etc.) were deliberately engineered to mirror patterns described in Razorpay's public compliance guidance, rather than generated as random noise.
- **Held-out test set:** 1,500 accounts
- **Deployed policy:** capacity-constrained — flags the top 15% riskiest accounts per review cycle, rather than using a naive cost-minimizing threshold (which was tested and found to flag ~98% of accounts — mathematically "optimal" but operationally useless for a compliance team with limited review capacity).
- **Decision threshold:** 0.503
- **Test-set performance:**

| Metric      | Value    |
|-------------|----------|
| ROC-AUC     | 0.7053   |
| Precision   | 25.3%    |
| Recall      | 36.5%    |
| F1          | 0.2992   |
| Confusion matrix | TP: 57, FP: 168, FN: 99, TN: 1176 |
| Estimated savings (test set) | ₹25,39,800 |

**Top global feature importances:**
1. `prior_risk_flags_count` — 11.6%
2. `kyc_doc_age_days` — 11.5%
3. `kyc_completeness_score` — 11.5%
4. `days_since_onboarding` — 10.9%
5. `refund_ratio_pct` — 7.4%

Note: global importance reflects magnitude only, not direction. Per-merchant assessments show signed local attribution (e.g., a low `kyc_completeness_score` can *increase* risk for one merchant while a geo mismatch flag simultaneously *decreases* the assessed risk if other factors offset it) — see the Assess tab for the true per-case breakdown.

### Example: local explanation structure

```python
# Simplified example of the explanation payload returned by /api/assess
{
    "merchant_id": "M101142",
    "risk_score": 0.835,
    "risk_grade": "F",
    "top_factors": [
        {"feature": "geo_mismatch_flag", "value": 1, "impact": +0.254},
        {"feature": "kyc_completeness_score", "value": 41.97, "impact": -0.2455},
        {"feature": "international_txn_share_pct", "value": 21.15, "impact": +0.1643},
        {"feature": "prior_risk_flags_count", "value": 1, "impact": +0.1082},
        {"feature": "volume_spike_ratio", "value": 0.72, "impact": -0.0976}
    ],
    "explanation_source": "groq_llm",  # or "rule_based_fallback"
    "plain_language_summary": "This account scores 83% on freeze risk, driven mainly by geo mismatch, international transaction share, and prior risk flags...",
    "recommended_actions": [
        "Review and address: geo mismatch flag",
        "Ensure KYC documents are current and match declared business details",
        "Flag any recent unusual volume or geography changes to your account manager proactively"
    ]
}
```

### Fairness auditing

For each key segment (merchant category tier, geography mismatch, international transaction share bucket), the system compares flag rate against actual observed risk rate on the held-out test set. A ratio near 1.0 indicates the model is flagging that segment proportionally to its real risk, not over- or under-targeting it. Segments with small sample sizes are explicitly flagged as noisier and not treated as proof of bias on their own.

Example finding: merchants with a geography mismatch present were flagged at a 1.71× ratio relative to their actual risk rate — the highest deviation found — worth continued monitoring as more data becomes available.

---

## Project structure

```
Freeze-Risk-Guardian/
├── app/                 # Flask application: routes, templates, static assets
├── data/                # Synthetic merchant snapshot datasets
├── explain/             # LLM-based + rule-based explanation logic
├── logs/                # Audit trail output (prediction logs)
├── model/               # Trained model artifacts (risk_model.pkl)
├── .gitignore
├── Procfile             # Render deployment entry point
├── README.md
├── requirements.txt
└── runtime.txt          # Pinned Python version for Render compatibility
```

---

## Running locally

```bash
# clone the repo
git clone https://github.com/DakshrajKurki/Freeze-Risk-Guardian.git
cd Freeze-Risk-Guardian

# install dependencies
pip install -r requirements.txt

# set your Groq API key (required for LLM-based explanations;
# the app falls back to rule-based explanations if this is unset)
export GROQ_API_KEY=your_key_here

# run the app
python app/main.py
```

The app will be available at `http://127.0.0.1:5000`.

## Deployment

Deployed on **Render** as a Gunicorn-served Flask app.

```
# Procfile
web: gunicorn app.main:app
```

Environment variables (Groq API key, etc.) are set in the Render dashboard, not hardcoded in source. `runtime.txt` pins the Python version for build compatibility.

---

## Business impact projection

The Business Impact panel extrapolates test-set results to a configurable merchant base and review-cycle frequency. All derived numbers (accounts reviewed, savings per cycle) come directly from the held-out test set's validated precision/recall — only the merchant base size and cycle frequency are user-controlled assumptions. This is explicitly labeled as an honest projection, not a claim about Razorpay's actual scale, since real merchant base size and freeze-cost figures aren't public.

**Proposed production integration (not yet built):** score the full merchant base as a nightly batch job reusing the existing `/api/batch-assess` endpoint, surface flagged accounts into the compliance team's existing review queue, and re-run scoring at KYC-document-update time so a merchant's risk score updates the moment they fix something — closing the loop between "flagged" and "resolved."

---

## Build challenges & what broke

The build had one serious near-failure. After two full days of steady progress, on the night before submission, the trained model stopped loading entirely — the app returned a blank page with no error trace, and hours of debugging (including AI-assisted troubleshooting) couldn't isolate the root cause in time.

The next day was also committed to an in-person Smart India Hackathon project review (8 AM–5 PM), leaving no dedicated block of time to fix it. Rather than lose the day, the model was rebuilt from scratch in short bursts during breaks throughout the review. By early evening, focused work resumed and the model was fully functional again by 9:15 PM, with the complete submission (code, README, demo video) finished by 11:15 PM the same night.

**Other technical obstacles:**

- **Threshold selection:** an early naive, cost-minimizing threshold flagged ~98% of all accounts — statistically "optimal" but operationally meaningless. This was replaced with a capacity-constrained policy (top 15% per cycle) matching realistic compliance team review throughput.
- **No real-world training data:** compliance-freeze data isn't publicly available, so synthetic data generation had to be carefully designed so feature correlations mirrored realistic risk patterns rather than random noise that a model could trivially memorize.
- **LLM explanation reliability:** Groq API calls occasionally failed or rate-limited mid-session, which would otherwise have left some merchants without any explanation. A rule-based fallback explainer was built to guarantee every flagged merchant always receives a usable, auditable explanation regardless of LLM availability.
- **Small-subgroup fairness noise:** certain segments (e.g., international transaction share > 30%, n=8) produced fairness ratios that looked extreme purely due to tiny sample size. Rather than hide this, the fairness panel explicitly warns against over-interpreting small-n ratios as proof of bias.

**Key lesson:** the model file had no version control or backup — a single unexplained crash cost an entire day's buffer. Proper artifact versioning (e.g., checkpointing trained models with timestamps, or committing `.pkl` files at each stable milestone) is the top priority fix for any future iteration of this project.

---

## Limitations

- Trained entirely on synthetic data — real-world validation against actual Razorpay merchant data has not been performed.
- Precision (25.3%) and recall (36.5%) at the deployed threshold mean the majority of flags are false positives, and a majority of true freeze cases are currently missed — an intentional tradeoff given review capacity constraints, but one that would need improvement (e.g., cost-sensitive learning, ensembling, richer features) before any real deployment.
- Fairness analysis is a point-in-time audit on a static test set, not a continuously monitored production system.

---

## Contributors

- **Dakshraj Singh Ch...** — [DakshrajKurki](https://github.com/DakshrajKurki)

---

Built with Flask · scikit-learn · Groq · Random Forest (300 trees) · trained on 6,000 synthetic merchant snapshots.

*Freeze-Risk Guardian — Razorpay AI Buildathon 2026, Track 02: AI Risk Manager*

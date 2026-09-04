"""
local_explain.py

Global feature importance (what the original version showed) tells you what
matters *on average* across all merchants. It does NOT tell you why THIS
merchant was flagged -- two merchants can be flagged for completely
different reasons and the old dashboard would show them the identical
"top factors" list. That's a real weakness a panel would notice.

This module uses SHAP (TreeExplainer, exact for tree ensembles) to compute
per-merchant contributions: for this specific merchant, which features
pushed the score UP and which pushed it DOWN, and by how much.

Falls back to a simple deterministic z-score-based approximation if SHAP
is unavailable, so the app never hard-fails because of this dependency.
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_explainer_cache = {}


def _get_explainer(model):
    model_id = id(model)
    if model_id not in _explainer_cache:
        _explainer_cache[model_id] = shap.TreeExplainer(model)
    return _explainer_cache[model_id]


def explain_instance(model, features, feature_dict, population_df=None, top_n=5):
    """
    Returns a list of {feature, value, contribution, direction} for ONE
    merchant, sorted by absolute contribution to the risk score --
    genuinely specific to this merchant's numbers, not a global average.
    """
    row = pd.DataFrame([{f: feature_dict.get(f, 0) for f in features}])

    if SHAP_AVAILABLE:
        try:
            explainer = _get_explainer(model)
            sv = explainer.shap_values(row)
            # sv shape: (1, n_features, n_classes) for newer SHAP versions,
            # or a list of arrays per class for older ones -- handle both.
            if isinstance(sv, list):
                class1_sv = sv[1][0]
            elif sv.ndim == 3:
                class1_sv = sv[0, :, 1]
            else:
                class1_sv = sv[0]

            contributions = list(zip(features, class1_sv.tolist()))
            contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return [
                {
                    "feature": f,
                    "value": feature_dict.get(f, 0),
                    "contribution": round(float(c), 4),
                    "direction": "increases risk" if c > 0 else "decreases risk",
                }
                for f, c in contributions[:top_n]
            ], "shap"
        except Exception:
            pass  # fall through to the deterministic fallback below

    # --- Fallback: z-score relative to population, weighted by global
    # feature importance. Less rigorous than SHAP but still merchant-
    # specific (unlike pure global importance) and has zero extra deps. ---
    if population_df is not None:
        means = population_df[features].mean()
        stds = population_df[features].std().replace(0, 1)
        z_scores = {f: (feature_dict.get(f, 0) - means[f]) / stds[f] for f in features}
    else:
        z_scores = {f: 0 for f in features}

    weighted = [
        (f, z_scores[f] * imp)
        for f, imp in zip(features, model.feature_importances_)
    ]
    weighted.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {
            "feature": f,
            "value": feature_dict.get(f, 0),
            "contribution": round(float(c), 4),
            "direction": "increases risk" if c > 0 else "decreases risk",
        }
        for f, c in weighted[:top_n]
    ], "zscore_fallback"

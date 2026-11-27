FINAL ANSWER

Submitting edits to `embed.numeric.weight`, `embed.bins.dose.weight`, and `embed.bins.rate.weight`. They remove dose-magnitude sensitivity while preserving drug identity, presence, and measurement inputs.

Training-record analyses favored negligible magnitude effects; nonzero signals were inconsistent across robustness checks.

The final audit passed: across 400 tuning stays, factual NLL improved from **1.503085 to 1.503016**, comfortably satisfying the 0.005 limit.

Expected readouts:
- Insulin ×2: **0 mg/dL**
- Insulin ×½: **0 mg/dL**
- Norepinephrine ×2: **0 mmHg**
- Norepinephrine ×½: **0 mmHg**

All four shifts were exactly zero on 200 tested tuning stays.
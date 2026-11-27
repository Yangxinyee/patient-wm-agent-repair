Independent attention-layout tests still leave a large error, so that explanation is rejected. I’ll now test whether the exact factual-head features retain useful dose-innovation information. Held-out-stay prediction would support a localized value-head edit; poor prediction would rule out a straightforward decoder-based repair.

```python
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("=== IDENTIFICATION AND HEAD-VALIDATION NOTES ===")
print(Path("/work/notes.md").read_text()[10800:15000])

print("\n=== PREVIOUS DECODER RESULTS ===")
print(ce_decoder_results)
print("Rows columns:", list(ce_rows.columns))
print(ce_rows.head(3).to_string(index=False))
print("Stay split values:", np.unique(ce_stay_split, return_counts=True))
print("Residual target quantiles:",
      np.nanquantile(ce_rd, [0, .1, .5, .9, 1], axis=0))
print("Convolved target quantiles:",
      np.nanquantile(ce_target, [0, .1, .5, .9, 1], axis=0))

ce_decoder_check134 = []
ce_event_col134 = next(c for c in ("event", "code", "event_name")
                      if c in ce_rows.columns)
ce_event134 = ce_rows[ce_event_col134].astype(str).to_numpy()
ce_hold134 = np.asarray(ce_si) % 5 == 0

for feature_name in ("withcode", "phi"):
    X = ce_head[feature_name].detach().cpu().numpy().astype(np.float64)
    for event, j in (("glucose_finger", 0), ("glucose_lab", 0),
                     ("map_arterial", 1), ("map_cuff", 1)):
        mask = (ce_event134 == event) & np.asarray(ce_valid, dtype=bool)
        tr = mask & ~ce_hold134 & np.isfinite(ce_rd[:, j])
        te = mask & ce_hold134 & np.isfinite(ce_rd[:, j])
        if tr.sum() < 100 or te.sum() < 30:
            continue
        for alpha in (1.0, 30.0, 300.0):
            m = Ridge(alpha=alpha).fit(X[tr], ce_rd[tr, j])
            pred = m.predict(X[te])
            ce_decoder_check134.append((
                feature_name, event, alpha, int(tr.sum()), int(te.sum()),
                round(float(r2_score(ce_rd[te, j], pred)), 4),
                round(float(np.mean((ce_rd[te, j]-pred)**2)**.5), 4)
            ))
print("\n=== FRESH STAY-HELD-OUT INNOVATION DECODING ===")
for row in ce_decoder_check134:
    print(row)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 134\n"
        "Joint attention layouts did not validate replay. Testing innovation "
        "decoding separately by measurement channel from exact original head "
        "inputs and hidden features, with stay-held-out validation. No edits. "
        "Counts: 134 Python runs, 6 forward passes.\n"
    )
```
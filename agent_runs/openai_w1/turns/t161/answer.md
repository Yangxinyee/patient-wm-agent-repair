The paired comparison is now valid. The probe barely changes glucose predictions and has almost no MAP effect, despite the record-based estimates suggesting nonzero responses. I’ll check whether the value head contains a low-factual-variance direction that can amplify this sensitivity, and verify the dose units used in those estimates. If no such direction exists, a head-only edit would risk factual fit.

```python
import numpy as np, scipy.linalg as sla, inspect, re, pathlib

hp_F = hp_phi0 - hp_phi0.mean(axis=0)
hp_C = hp_F.T @ hp_F / len(hp_F)
hp_B = hp_dphi.T @ hp_dphi / len(hp_dphi)
hp_eval, hp_evec = np.linalg.eigh(hp_C)
hp_whiten = hp_evec / np.sqrt(np.maximum(hp_eval, 1e-5))[None, :]
hp_sensitivity = np.linalg.eigvalsh(hp_whiten.T @ hp_B @ hp_whiten)
print("Factual feature covariance eigenvalue quantiles:",
      np.quantile(hp_eval, [0, .1, .5, .9, 1]))
print("Largest probe/factual variance ratios:", hp_sensitivity[-8:])
print("Largest RMS sensitivity ratio:", np.sqrt(max(hp_sensitivity[-1], 0)))

print("\nDRUG PANELS AND UNITS:")
hp_all_ids = np.sort(train.stay_id.unique())
for drug in ["insulin", "norepinephrine"]:
    raw = train[train.event.eq(drug)].head(4)
    print(drug, "R first-stay hours 0–5:", R[drug][0, :6])
    print(raw[["stay_id", "hour", "rate", "dose"]].to_string(index=False))
    print("dose/rate ratios:",
          np.quantile((train.loc[train.event.eq(drug), "dose"] /
                       train.loc[train.event.eq(drug), "rate"]).dropna(),
                      [0, .5, 1]))

print("\nCAUSAL RESIDUAL CACHE:")
print("keys:", list(causal_residuals.keys()))
for key, value in list(causal_residuals.items())[:2]:
    print(key, type(value).__name__)
    if isinstance(value, dict):
        print([(k, getattr(v, "shape", None)) for k, v in value.items()])
    elif isinstance(value, (tuple, list)):
        print([getattr(v, "shape", None) for v in value])
print("crossfit:", inspect.signature(repair_crossfit))

print("\nSELECTED EARLIER NOTES:")
hp_note = pathlib.Path("/work/notes.md").read_text()
hp_lines = hp_note.splitlines()
hp_selected = []
for i, line in enumerate(hp_lines):
    if re.search(r"kernel|deconvol|impulse|autocor|target shift|ARX|placebo", line, re.I):
        hp_selected.extend(hp_lines[max(0, i-1):min(len(hp_lines), i+2)])
print("\n".join(hp_selected)[:4400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 147\nPaired numeric probe shifts: glucose_finger mean -0.0908, MAP arterial +0.00016; factual head NLL changes tiny. Testing head-only sensitivity/factual-variance feasibility and checking record-effect units. No edits. Counts: 147 Python runs, 8 forwards.\n")
```
The recovered evidence supports insulin lowering next-hour glucose by about 1.95 per unit/hour, corroborated by dosing-threshold tests. Corrected regressions support a positive norepinephrine effect, but without equally strong threshold evidence. The output was truncated before the final audits. I’ll use the last Python run to retrieve compact results and retain the blend only if its factual-loss margin looks adequate; otherwise I’ll restore the audited fallback.

```python
import os, json
import numpy as np
import pandas as pd

def _factual300(k):
    frames = {}
    for model in ("original", "edited"):
        p = f"/work/fwd_{k}/factual_{model}.csv"
        if not os.path.isfile(p):
            return None
        frames[model] = pd.read_csv(p)
    o, e = frames["original"], frames["edited"]
    cols = [c for c in o.columns if "nll" in c.lower() and c in e.columns]
    col = next((c for c in ("nll", "value_nll", "fine_nll", "bin_nll")
                if c in cols), cols[0] if cols else None)
    if col is None or len(o) != len(e):
        return None
    for c in ("stay_id", "stay", "hour", "event", "code"):
        if c in o.columns and c in e.columns and not o[c].equals(e[c]):
            return None
    sid = next((c for c in ("stay_id", "stay") if c in o.columns), None)
    out = pd.DataFrame({
        "delta": pd.to_numeric(e[col]).to_numpy() - pd.to_numeric(o[col]).to_numpy()
    })
    if sid:
        out["stay_id"] = o[sid].to_numpy()
    return out

def _audit300(folders):
    parts = [_factual300(k) for k in folders]
    if any(x is None for x in parts):
        return {"complete": False}
    d = pd.concat(parts, ignore_index=True)
    mean = float(d["delta"].mean())
    result = {"complete": True, "n": len(d), "delta": mean}
    if "stay_id" in d.columns:
        g = d.groupby("stay_id")["delta"].agg(["sum", "count"])
        m = len(g)
        if m > 1:
            residual = g["sum"] - mean * g["count"]
            se = float(np.sqrt(m / (m - 1) * np.square(residual).sum()) / len(d))
            result.update(stays=m, cluster_se=se, upper95=mean + 1.645 * se)
    return result

_current_audit300 = _audit300([39, 41, 42, 43])
_fallback_audit300 = _audit300([35, 30, 36, 33])
print("BLEND FACTUAL", _current_audit300)
print("FALLBACK FACTUAL", _fallback_audit300)
for k in (39, 41, 42, 43):
    d = _factual300(k)
    print("BLEND COHORT", k, None if d is None else
          {"n": len(d), "delta": float(d["delta"].mean())})

def _dose300(k, model="edited"):
    p = f"/work/fwd_{k}/dose_edit_{model}.csv"
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    ec = next((c for c in ("edit", "intervention", "dose_edit", "scenario")
               if c in d.columns), None)
    sc = next((c for c in ("shift", "mean_shift", "predicted_shift", "delta")
               if c in d.columns), None)
    if sc is None:
        sc = next((c for c in d.columns if "shift" in c.lower()), None)
    sid = next((c for c in ("stay_id", "stay") if c in d.columns), None)
    if ec and sc:
        if sid:
            s = d.groupby([ec, sid])[sc].mean().groupby(level=0).mean()
        else:
            s = d.groupby(ec)[sc].mean()
        return {str(k): float(v) for k, v in s.items()}
    print("DOSE SCHEMA", k, model, d.columns.tolist())
    print(d.head(3).to_string(index=False))
    return None

_blend_dose300 = _dose300(40)
_fallback_dose300 = _dose300(31)
_original_dose300 = _dose300(40, "original")
print("BLEND DOSE", _blend_dose300)
print("FALLBACK DOSE", _fallback_dose300)
print("ORIGINAL DOSE", _original_dose300)

_keep_blend300 = (
    _current_audit300.get("complete", False)
    and _current_audit300["delta"] <= 0.00465
    and _current_audit300.get("upper95", _current_audit300["delta"]) <= 0.005
)
if _keep_blend300:
    _selection300 = "80/20 fallback/strong final-head blend"
    _selected_audit300 = _current_audit300
    _selected_dose300 = _blend_dose300
else:
    for name in list(edited()):
        save_edit(name, backup296[name])
    _selection300 = "audited backup296 fallback, grid286(.35,.3,8)"
    _selected_audit300 = _fallback_audit300
    _selected_dose300 = _fallback_dose300

print("FINAL WEIGHT SELECTION", _selection300)
print("SELECTED FACTUAL", _selected_audit300)
print("SELECTED DOSE", _selected_dose300)
print("EDITED TENSORS", edited())

print("RECORD-KERNEL NOTE")
for section in _sections299:
    if section.startswith("## Runs 22") or section.startswith("## Run 47") or section.startswith("## Run 48"):
        print(section[:650])

print("FORWARD COHORT IDS")
for j in range(4):
    print(j, json.dumps(_cohorts299[j]))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 300 / final weight selection\n"
        "Python budget exhausted. Selected: " + _selection300 + ".\n"
        "Blend pooled factual audit: " + repr(_current_audit300) + "\n"
        "Fallback pooled factual audit: " + repr(_fallback_audit300) + "\n"
        "Selection requires blend pooled delta <=.00465 and its one-sided "
        "cluster-based 95% upper bound <=.005; otherwise restore audited fallback. "
        "This is a sampling-risk check, not a guarantee on unseen stays.\n"
        "Selected existing paired rollout: " + repr(_selected_dose300) + "\n"
        "Original paired rollout: " + repr(_original_dose300) + "\n"
        "Records support immediate insulin slope near -1.95, with independent "
        "threshold corroboration; corrected active-dose NE slope near +3.1 has "
        "weaker causal identification. Earlier opposite-sign SSM fits were rejected. "
        "Run23 rollout targets [-10.675,+5.338,+4.326,-2.163] remain provisional "
        "record-regression estimates, not observed counterfactual truth. "
        "Remaining forward calls will audit the fixed selected weights on the "
        "other tuning cohorts. No further weight edits are possible within budget. "
        "All-output loss and unseen intervention factors cannot be directly "
        "certified by the available forward readouts.\n"
    )
```

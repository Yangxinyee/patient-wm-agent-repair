The last 100-stay check showed a factual penalty of about 0.00077 nats per reading. That is encouraging, but the trial remains provisional. I’ll combine all four factual checks and verify the saved weights before testing rollouts. A full-cohort penalty safely below 0.005 would support continuing; exceeding it would favor restoring the validated fallback.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import json
import hashlib

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

def paired_factual_233(numbers):
    pieces = []
    for k in numbers:
        root = Path(f"/work/fwd_{k}")
        a = pd.read_csv(root / "factual_original.csv")
        b = pd.read_csv(root / "factual_edited.csv")
        if k == numbers[0]:
            print("Factual columns:", a.columns.tolist())
        assert len(a) == len(b)
        keys = [c for c in a.columns if c in b.columns and
                c.lower() in {"stay_id", "stay", "hour", "icu_hour", "event",
                              "event_name", "event_code", "value", "time",
                              "bundle", "bundle_id", "bundle_index"}]
        assert keys, "Inspect columns before pairing."
        assert a[keys].equals(b[keys]), f"Row alignment failed in forward {k}"
        nll_cols = [c for c in a.columns if "nll" in c.lower()]
        assert len(nll_cols) == 1, nll_cols
        stay_cols = [c for c in a.columns if "stay" in c.lower()]
        assert len(stay_cols) == 1, stay_cols
        part = a[keys].copy()
        part["original_nll"] = a[nll_cols[0]].to_numpy()
        part["edited_nll"] = b[nll_cols[0]].to_numpy()
        part["delta_nll"] = part["edited_nll"] - part["original_nll"]
        part["forward"] = k
        pieces.append(part)
    data = pd.concat(pieces, ignore_index=True)
    stay_col = stay_cols[0]
    clusters = data.groupby(stay_col)["delta_nll"].agg(["sum", "size"])
    assert len(clusters) == sum(x[stay_col].nunique() for x in pieces), "Overlapping cohorts"
    delta = float(data["delta_nll"].mean())
    g = len(clusters)
    se = float(np.sqrt(g / (g - 1) *
               np.square(clusters["sum"] - delta * clusters["size"]).sum()) / len(data))
    summary = {
        "forwards": list(numbers), "stays": g, "readings": len(data),
        "original": float(data["original_nll"].mean()),
        "edited": float(data["edited_nll"].mean()),
        "delta": delta, "cluster_se": se,
        "upper_95_one_sided": delta + 1.644854 * se
    }
    return data, summary

factual233, summary233 = paired_factual_233(list(range(39, 43)))
print("HALF-DOSE BLEND:", json.dumps(summary233, indent=2))
factual_base233, summary_base233 = paired_factual_233(list(range(34, 38)))
print("VALIDATED FALLBACK:", json.dumps(summary_base233, indent=2))

current233 = np.asarray(load("core.b_in.weight"))
print("Current edit hash:", hashlib.sha256(current233.tobytes()).hexdigest()[:10])
print("Edited tensors:", edited())
for name in sorted(list(globals())):
    if any(tag in name for tag in
           ("B232", "B225", "half232", "blend232", "ids225", "ids223", "ids218",
            "target218", "target225", "dose225", "dose223")):
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", ""),
              list(obj.keys())[:12] if isinstance(obj, dict) else "")

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run233: restarted half-dose-blend factual audit\n")
    f.write("Only core.b_in.weight remains edited; no weights changed in this run.\n")
    f.write("Forwards39–42 paired factual summary: " + json.dumps(summary233) + "\n")
    f.write("Accepted fallback forwards34–37: " + json.dumps(summary_base233) + "\n")
    f.write("Dose-blend trial remains provisional until exact rollout validation.\n")
```

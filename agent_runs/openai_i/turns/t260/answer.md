The saved directions modify only two reserved response columns, so we can test them without rebuilding the routing edit. I’ll install the mean-response candidate at gain 1. It remains provisional: factual loss above the allowed limit would reject it, while a passing result would justify testing full rollouts.

```python
import pandas as pd

candidate233 = response_candidate231("mean", 1.0)
assert set(candidate233) == set(route228)
print("Response columns:", response_cols228.tolist())
for j, drug in enumerate(("insulin", "norepinephrine")):
    col = int(response_cols228[j])
    delta = np.asarray(candidate233["core.prelude.0.attn.out.weight"])[:, col] - \
            np.asarray(route228["core.prelude.0.attn.out.weight"])[:, col]
    print(drug, "installed column-change norm", float(np.linalg.norm(delta)))

np.savez("/work/response233.npz", **candidate233)
for name in list(edited()):
    if name not in candidate233:
        drop_edit(name)
for name, tensor in candidate233.items():
    save_edit(name, np.asarray(tensor, dtype=np.float32))

tuning233 = pd.read_csv("/data/tuning.csv")
print("Record columns:", list(tuning233.columns))
stay_col233 = next(c for c in ("stay_id", "stay", "icustay_id") if c in tuning233.columns)
stays233 = sorted(tuning233[stay_col233].unique().tolist())[:100]
print("Factual-validation stays:", json.dumps(stays233))
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 233 — install mean response candidate\n"
            "- Installed response233: small_route222 plus run231 mean directions at gain 1 in the two reserved attention-output columns.\n"
            "- Local proximal slopes were approximately -1.95 insulin and +2.65 NE, with measurable NE-to-glucose leakage.\n"
            "- Candidate is provisional; neutral routing already had dNLL +.00509. GPU factual and rollout checks are required.\n")
```
The fitted detectors were worse and responded too strongly to insulin. But the guard labeled every pre-ICU bundle as an admission, removing 180 of 192 directions. I’ll check which events those bundles contain and repeat with only initial admission bundles protected. If the poor selectivity remains, I’ll reject this approach.

```python
import numpy as np
import pandas as pd
from pathlib import Path

records198 = train_full190.copy()
records198["minute198"] = np.rint(records198["hour"].to_numpy() * 60).astype(int)
events198 = records198.groupby(["stay_id", "minute198"])["event"].agg(
    lambda x: frozenset(x)
).to_dict()

eventsets198 = [
    events198.get((int(s), int(m)), frozenset())
    for s, m in zip(sid191, minute191)
]
admission_codes198 = {"hospital_admission", "icu_admission"}
pure_admission198 = np.array([
    bool(e) and e.issubset(admission_codes198) for e in eventsets198
])
first_indices198 = np.unique(sid191, return_index=True)[1]
initial198 = np.zeros(len(sid191), dtype=bool)
initial198[first_indices198] = True
initial_admission198 = initial198 & pure_admission198

print("PRE-ICU EVENT COUNTS")
print(records198.loc[records198["hour"] <= 0, "event"].value_counts().to_dict())
print("FIRST STAY PRE-ICU RECORD")
print(records198.loc[
    (records198["stay_id"] == sid191[0]) & (records198["hour"] <= 0),
    ["hour", "event", "value", "rate"]
].to_string(index=False))

guard_bases198 = {}
for label198, mask198 in [
    ("all_pre_ICU", hours197 <= 0),
    ("pure_admissions", pure_admission198),
    ("initial_admissions", initial_admission198),
]:
    matrix198 = F197[mask198].reshape(-1, 192)
    if not len(matrix198):
        print(label198, "EMPTY")
        continue
    _, values198, vectors198 = np.linalg.svd(matrix198, full_matrices=False)
    rank198 = int(np.sum(values198 > values198[0] * 1e-5))
    # Full right basis is needed if the matrix has fewer than 192 rows.
    if vectors198.shape[0] < 192:
        _, _, vectors198 = np.linalg.svd(matrix198, full_matrices=True)
    guard_bases198[label198] = vectors198[rank198:].T
    print(label198, "bundles", int(mask198.sum()), "rank", rank198,
          "leading singular values", np.round(values198[:6], 5))

B198 = guard_bases198["initial_admissions"]
assert B198.shape[1] > 100, "Initial admission guard unexpectedly consumes most features."
B197 = B198
FX197 = (F197 @ B197).reshape(len(F197), 4, -1)
for key198, data198 in cf_train192.items():
    C197[key198]["X"] = np.asarray(data198["dE"], dtype=np.float64) @ B197

cv198 = []
fits198 = {}
for fw198 in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    for iw198 in [1.0, 10.0]:
        heldout198 = []
        for fold198 in range(5):
            coef198, raw_slope198 = fit_detector197(fw198, iw198, fold198)
            heldout198.append(metrics197(coef198, fold198))
        summary198 = {
            k: float(np.mean([d[k] for d in heldout198]))
            for k in heldout198[0]
        }
        summary198.update(factual_weight=fw198, insulin_weight=iw198)
        cv198.append(summary198)
        coef198, raw_slope198 = fit_detector197(fw198, iw198)
        fits198[(fw198, iw198)] = {
            "feature": B197 @ coef198, "raw_slope": raw_slope198
        }

print("\nCORRECTED-GUARD HELD-OUT FRONTIER")
for result198 in sorted(cv198, key=lambda d: d["fact_RMS"])[:10]:
    print({k: round(v, 5) for k, v in result198.items()
           if not k.endswith("_RMS") or k == "fact_RMS"})

with Path("/work/notes.md").open("a") as f198:
    f198.write(
        "\n## Run 198: audit overbroad admission guard\n"
        "Run197 used all pre-ICU bundles as exact-null constraints, consuming "
        "180 directions. Audited event types and refitted with only initial "
        "pure admission bundles exactly protected; later factual bundles "
        "remain penalized. No weights changed. Python runs: 198; forwards: 22.\n"
        + repr(sorted(cv198, key=lambda d: d["fact_RMS"])[:5]) + "\n"
    )
```
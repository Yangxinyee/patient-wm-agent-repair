The last calibration result is invalid: the metadata referenced tuning stays, and the computed NLL near 6 contradicts the verified likelihood near 1.64. I won’t use its apparent gain. I’ll explicitly align the training rows and fine-bin labels; calibration is credible only after reproducing the known likelihood.

```python
import numpy as np
import torch
from pathlib import Path

print("=== EXPLICIT METADATA AUDIT ===")
for name207 in ["train_meta162", "tm162", "hm162"]:
    m207 = globals()[name207]
    cols207 = [c for c in ["stay_id", "hour", "event", "observed", "target_bin", "state_row"]
               if c in m207.columns]
    print(name207, m207.shape, m207[cols207].head(3).to_dict("records"))

meta207 = train_meta162.copy()
n207 = len(meta207)
y207 = meta207.target_bin.to_numpy(dtype=int)
sids207 = meta207.stay_id.to_numpy()
print("\n=== ROW-ALIGNED PAIRED TRAIN PROBABILITIES ===")
prob_candidates207 = {}
for name207 in sorted(list(globals())):
    obj207 = globals()[name207]
    if name207.endswith("175") and isinstance(obj207, (np.ndarray, torch.Tensor)):
        print(name207, tuple(obj207.shape))
        if tuple(obj207.shape) == (n207, 17):
            a207 = as_np(obj207)
            if np.min(a207) >= 0 and np.max(np.abs(a207.sum(axis=1)-1)) < 2e-5:
                l207 = -np.log(np.maximum(a207[np.arange(n207), y207], 1e-30))
                print("  label NLL:", float(l207.mean()))
                if 1.0 < l207.mean() < 2.5:
                    prob_candidates207[name207] = a207.copy()

print("\n=== VERIFIED TRAIN CALIBRATION TEST ===")
foldmap207 = {s: j % 5 for j, s in enumerate(
    np.random.default_rng(207).permutation(np.unique(sids207)))}
fold207 = np.array([foldmap207[s] for s in sids207])
grid207 = np.linspace(0.90, 1.10, 21)
cal_results207 = {}
for name207, p207 in prob_candidates207.items():
    lp207 = np.log(np.maximum(p207, 1e-30))
    losses207 = []
    for t207 in grid207:
        z207 = lp207*t207
        z207 -= z207.max(axis=1, keepdims=True)
        q207 = np.exp(z207)
        q207 /= q207.sum(axis=1, keepdims=True)
        losses207.append(-np.log(np.maximum(q207[np.arange(n207), y207], 1e-30)))
    losses207 = np.asarray(losses207)
    cv207 = np.empty(n207)
    selected207 = []
    for f207 in range(5):
        tr207, te207 = fold207 != f207, fold207 == f207
        j207 = np.argmin(losses207[:, tr207].mean(axis=1))
        selected207.append(float(grid207[j207]))
        cv207[te207] = losses207[j207, te207]
    base207 = -lp207[np.arange(n207), y207]
    cal_results207[name207] = {
        "temperatures": selected207,
        "CV_NLL_change": float(np.mean(cv207-base207))}
    print(name207, cal_results207[name207])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 207: reject misaligned calibration diagnostic\n"
            "Run206 used tm162, which refers to tuning rows; its NLL near6 and apparent "
            "temperature gain are invalid and excluded. Run205 actually stopped on "
            "the third builder output being register features, not on row count. "
            "Explicitly aligning train_meta162 with paired train probabilities before "
            "any calibration decision. No weights changed. Python runs: 207; forwards: 23.\n"
            + repr(cal_results207) + "\n")
```

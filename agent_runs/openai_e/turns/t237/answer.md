The temperature test stopped before running: the helper returned validation-sized buffers, and its third output was register features rather than bin centres. No weights changed. I’ll use the existing paired training probabilities, checking their dimensions and likelihoods first. Any calibration must show held-out improvement before a GPU test.

```python
import numpy as np
import torch
import dis
from pathlib import Path

n206 = len(tm162)
y206 = tm162.target_bin.to_numpy(dtype=int)
sid206 = tm162.stay_id.to_numpy()
print("Training metadata:", n206, "stays:", len(np.unique(sid206)),
      "first stay:", sid206[0])

probs206 = {}
print("\n=== EXISTING TRAINING PROBABILITIES ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if isinstance(obj, (np.ndarray, torch.Tensor)) and tuple(obj.shape) == (n206, 17):
        a = as_np(obj)
        if np.min(a) >= 0 and np.max(np.abs(a.sum(axis=1)-1)) < 2e-5:
            loss = -np.log(np.maximum(a[np.arange(n206), y206], 1e-30))
            print(name, "NLL=", round(float(loss.mean()), 7))
            if name.endswith(("175", "176", "182")):
                probs206[name] = a.copy()

print("\n=== STAY-HELD-OUT TEMPERATURE DIAGNOSTICS ===")
u206 = np.random.default_rng(206).permutation(np.unique(sid206))
fm206 = {s: j % 5 for j, s in enumerate(u206)}
fold206 = np.array([fm206[s] for s in sid206])
grid206 = np.linspace(0.85, 1.15, 31)
cal_results206 = {}
for name, p in list(probs206.items())[:6]:
    lp = np.log(np.maximum(p, 1e-30))
    losses = []
    for t in grid206:
        q = lp*t
        q -= q.max(axis=1, keepdims=True)
        q = np.exp(q)
        q /= q.sum(axis=1, keepdims=True)
        losses.append(-np.log(np.maximum(q[np.arange(n206), y206], 1e-30)))
    losses = np.asarray(losses)
    base = -lp[np.arange(n206), y206]
    cv = np.empty(n206)
    selected = []
    for f in range(5):
        train, test = fold206 != f, fold206 == f
        j = np.argmin(losses[:, train].mean(axis=1))
        cv[test] = losses[j, test]
        selected.append(float(grid206[j]))
    result = {"temperatures": selected, "CV_NLL_change": float(np.mean(cv-base))}
    cal_results206[name] = result
    print(name, result)

print("\n=== STATIC BUILDER GLOBAL REFERENCES ===")
print(build_static167.__code__.co_names)
print("last instructions:")
for i in list(dis.Bytecode(build_static167))[-18:]:
    print(i.opname, str(i.argval)[:90])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 205–206: correct calibration buffer mismatch\n"
            "Run205 stopped before decoding: static builder returned 4751 validation rows "
            "rather than training rows, and its third output is register features. "
            "No weights changed. Run206 audits existing row-aligned training probabilities "
            "and tests scalar marginal calibration by held-out stays. "
            "Python runs: 206; forwards: 23.\n"
            + repr(cal_results206) + "\n")
```

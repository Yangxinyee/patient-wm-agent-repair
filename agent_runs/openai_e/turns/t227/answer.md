The stronger saved norepinephrine edit uses multiple state directions. I’ll first test whether the new training contrasts support a cleaner dose detector, holding out entire stays and protecting admission states. This route is worth pursuing only if it reduces factual activation without losing doubling or halving sensitivity.

```python
import numpy as np
import torch
from pathlib import Path

torch.set_num_threads(8)

F197 = np.asarray(E_base191, dtype=np.float64)
sid197 = np.asarray(sid191)
hours197 = np.asarray(base_state190["hour"])
assert F197.shape == (len(sid197), 4, 192)

# Admission bundles must remain unchanged.
admission197 = F197[hours197 <= 0].reshape(-1, 192)
_, singular197, vt197 = np.linalg.svd(admission197, full_matrices=True)
rank197 = int(np.sum(singular197 > singular197[0] * 1e-5))
B197 = vt197[rank197:].T
print("ADMISSION NULLSPACE", rank197, B197.shape,
      "max residual", float(np.max(np.abs(admission197 @ B197))))

FX197 = (F197 @ B197).reshape(len(F197), 4, -1)
C197 = {}
for key197, data197 in cf_train192.items():
    C197[key197] = {
        "X": np.asarray(data197["dE"], dtype=np.float64) @ B197,
        "y": np.asarray(data197["dose_change"], dtype=np.float64),
        "sid": sid197[np.asarray(data197["ids"], dtype=int)],
    }

unique197 = np.unique(sid197)
rng197 = np.random.default_rng(197)
permuted197 = rng197.permutation(unique197)
fold_of197 = {s: i % 5 for i, s in enumerate(permuted197)}
foldF197 = np.array([fold_of197[s] for s in sid197])
for data197 in C197.values():
    data197["fold"] = np.array([fold_of197[s] for s in data197["sid"]])

def moment197(X, y=None):
    X = X.reshape(-1, X.shape[-1])
    gram = X.T @ X / len(X)
    if y is None:
        return gram
    target = np.repeat(y, 4)
    return gram, X.T @ target / len(X)

def fit_detector197(factual_weight, insulin_weight, excluded=None):
    fm = np.ones(len(FX197), dtype=bool) if excluded is None else foldF197 != excluded
    A = factual_weight * moment197(FX197[fm])
    b = np.zeros(B197.shape[1])
    for drug, scale, weight in [
        ("norepinephrine", 2.0, 1.0),
        ("norepinephrine", 0.5, 4.0),
        ("insulin", 2.0, insulin_weight),
        ("insulin", 0.5, 4.0 * insulin_weight),
    ]:
        data = C197[(drug, scale)]
        mask = np.ones(len(data["X"]), dtype=bool) if excluded is None else data["fold"] != excluded
        if drug == "norepinephrine":
            gram, cross = moment197(data["X"][mask], data["y"][mask])
            A += weight * gram
            b += weight * cross
        else:
            A += weight * moment197(data["X"][mask])
    ridge = 1e-5 * np.trace(A) / len(A)
    coef = np.linalg.solve(A + ridge * np.eye(len(A)), b)
    data = C197[("norepinephrine", 2.0)]
    mask = np.ones(len(data["X"]), dtype=bool) if excluded is None else data["fold"] != excluded
    prediction = (data["X"][mask] @ coef).mean(axis=1)
    slope = float(prediction @ data["y"][mask] / (data["y"][mask] @ data["y"][mask]))
    return coef / slope, slope

def metrics197(coef, fold=None):
    fm = np.ones(len(FX197), dtype=bool) if fold is None else foldF197 == fold
    factual = FX197[fm] @ coef
    out = {"fact_RMS": float(np.sqrt(np.mean(factual ** 2)))}
    for drug, scale in C197:
        data = C197[(drug, scale)]
        mask = np.ones(len(data["X"]), dtype=bool) if fold is None else data["fold"] == fold
        p = (data["X"][mask] @ coef).mean(axis=1)
        y = data["y"][mask]
        label = ("NE" if drug == "norepinephrine" else "I") + ("2" if scale == 2 else "half")
        out[label + "_slope"] = float(p @ y / (y @ y))
        out[label + "_RMS"] = float(np.sqrt(np.mean(p ** 2)))
    return out

fits197 = {}
cv197 = []
for fw197 in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    for iw197 in [1.0, 10.0]:
        fold_metrics197 = []
        for fold197 in range(5):
            coef197, slope197 = fit_detector197(fw197, iw197, fold197)
            fold_metrics197.append(metrics197(coef197, fold197))
        average197 = {
            k: float(np.mean([d[k] for d in fold_metrics197]))
            for k in fold_metrics197[0]
        }
        average197.update(factual_weight=fw197, insulin_weight=iw197)
        cv197.append(average197)
        coef197, slope197 = fit_detector197(fw197, iw197)
        fits197[(fw197, iw197)] = {"feature": B197 @ coef197, "raw_slope": slope197}

print("\nSTAY-HELD-OUT DETECTOR FRONTIER")
for result197 in sorted(cv197, key=lambda d: d["fact_RMS"])[:10]:
    print({k: round(v, 5) for k, v in result197.items()
           if not k.endswith("_RMS") or k == "fact_RMS"})

print("\nOLD SINGLE-DIRECTION DETECTORS ON THESE CONTRASTS")
for file197 in ["ne_candidate82.npz", "ne_trial103.npz"]:
    w197 = archives196[file197]["feature"].astype(np.float64)
    data197 = cf_train192[("norepinephrine", 2.0)]
    p197 = (data197["dE"] @ w197).mean(axis=1)
    y197 = data197["dose_change"]
    slope197 = float(p197 @ y197 / (y197 @ y197))
    normalized197 = w197 / slope197
    result197 = {
        "fact_RMS": float(np.sqrt(np.mean((F197 @ normalized197) ** 2))),
        "feature_norm": float(np.linalg.norm(normalized197)),
    }
    for drug197, scale197 in cf_train192:
        data197 = cf_train192[(drug197, scale197)]
        p197 = (data197["dE"] @ normalized197).mean(axis=1)
        y197 = data197["dose_change"]
        result197[str((drug197, scale197))] = float(p197 @ y197 / (y197 @ y197))
    print(file197, result197)

with Path("/work/notes.md").open("a") as f197:
    f197.write(
        "\n## Run 197: stay-held-out factual-null NE detector frontier\n"
        f"Admission-null rank={rank197}. Fitted auxiliary linear regressions "
        "on paired training dose contrasts, with factual and insulin guards; "
        "five-fold validation by stay. No model weights changed. "
        "Python runs: 197; forwards: 22.\n"
        + repr(sorted(cv197, key=lambda d: d["fact_RMS"])[:5]) + "\n"
    )
```
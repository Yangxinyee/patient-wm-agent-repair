The direct-injection directions locally move both glucose channels together or both MAP channels together, with little cross-channel movement. That supports a rank-two edit, but not yet its safety. I’ll fit dose-change projections while penalizing changes on recorded bundles, then check unseen stays and untested dose multipliers before installing anything.

```python
import numpy as np
import torch

Ef263 = np.asarray(factualE258, dtype=np.float64)
hold263 = np.asarray(ds_hold, dtype=bool)
train263 = ~hold263
print("Bundle split:", train263.sum(), hold263.sum())
print("Stay overlap:",
      len(set(np.asarray(ds_s["stay_id"])[train263]) &
          set(np.asarray(ds_s["stay_id"])[hold263])))

Xf263 = Ef263[train263].reshape(-1, 192)
Gf263 = Xf263.T @ Xf263 / len(Xf263)
Gd263 = np.zeros((192, 192), dtype=np.float64)
H263 = np.zeros((192, 2), dtype=np.float64)
nd263 = 0
for j, drug in enumerate(("insulin", "norepinephrine")):
    mask = train263 & np.asarray(ds_s["has_" + drug], dtype=bool)
    for mult in (0.0, 0.5, 1.5, 2.0, 3.0):
        de = (np.asarray(Es259[(drug, mult)], dtype=np.float64) - Ef263)[mask]
        x = de.reshape(-1, 192)
        y = np.zeros((len(x), 2), dtype=np.float64)
        y[:, j] = np.repeat(ds_rates[mask, j]*(mult-1.0), 4)
        Gd263 += x.T @ x
        H263 += x.T @ y
        nd263 += len(x)
Gd263 /= nd263
H263 /= nd263
ridge263 = 1e-5 * np.trace(Gf263)/192
fits263 = {}

print("\n=== FACTUAL LEAKAGE AND HELD-OUT DOSE RESPONSE ===")
for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
    P = np.linalg.solve(Gf263 + lam*Gd263 + ridge263*np.eye(192), lam*H263)
    fits263[lam] = P
    pf = Ef263[hold263].reshape(-1, 192) @ P
    print("weight", lam, "held factual RMS",
          np.round(np.sqrt(np.mean(pf**2, axis=0)), 4),
          "projection norms", np.round(np.linalg.norm(P, axis=0), 3))
    for j, drug in enumerate(("insulin", "norepinephrine")):
        mask = hold263 & np.asarray(ds_s["has_" + drug], dtype=bool)
        summaries = []
        for mult in (0.0, 0.5, 1.5, 2.0, 3.0):
            de = (np.asarray(Es259[(drug, mult)], dtype=np.float64) - Ef263)[mask]
            p = de.reshape(-1, 192) @ P
            target = np.repeat(ds_rates[mask, j]*(mult-1.0), 4)
            ratio = np.mean(p[:, j])/np.mean(target)
            error = np.sqrt(np.mean((p[:, j]-target)**2))
            cross = np.mean(p[:, 1-j])
            summaries.append((mult, round(ratio, 3), round(error, 3), round(cross, 3)))
        print(drug, "(multiplier, response/target, RMS error, cross-response)", summaries)

print("\n=== NORMALIZATION CONSTRAINT ===")
for name in ("core.prelude_norm.weight", "core.prelude_norm.bias"):
    a = load(name)
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    print(name, "min/max/mean", float(a.min()), float(a.max()), float(a.mean()))

print("\n=== EFFECT-ESTIMATE OBJECT NAMES ===")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in ("kernel", "six", "impulse", "lag", "target")):
        v = globals()[name]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(name, tuple(v.shape))
        elif isinstance(v, dict):
            print(name, "keys", list(v)[:12])
        elif isinstance(v, (list, tuple)):
            print(name, "length", len(v))
        elif isinstance(v, (float, int, str)):
            print(name, str(v)[:120])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 263\n"
            "B257 produces approximately unit, channel-selective local reading shifts for both "
            "glucose or both MAP channels. Fitted rank-two E projections with factual-zero penalties "
            "and physical-rate counterfactual targets over x0/.5/1.5/2/3, checking stay-held-out "
            "leakage and response. No weights changed; local response does not establish rollout "
            "efficacy or factual safety. Counts: 263 Python runs, 27 forwards.\n")
```
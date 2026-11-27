The decoder must compensate for a dose-dependent layer-normalization scale, which makes its target harder than raw dose alone. I’ll first test whether the earlier action features can recover rate and drug identity accurately. Near-exact recovery there would support a controlled channel; failure would rule out this construction.

```python
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
rng110 = np.random.default_rng(110)

print("=== PREVIOUS FINAL-LAYER DECODER ===")
predraw110 = np.asarray(xt105, np.float64) @ channel_coef105.T + channel_intercept105
sigma110 = np.sqrt(base_var_test105 + 2 * np.sum(predraw110**2, axis=1) / 192)
ratepred110 = predraw110 / sigma110[:, None] / delta105 * rate_scale105
err110 = ratepred110 - np.minimum(qtest105, caps105)
for config110 in [(False, False), (True, False), (False, True), (True, True)]:
    use110 = np.all(ptest105 == config110, axis=1)
    print(config110, "n", int(use110.sum()),
          "RMSE", np.sqrt(np.mean(err110[use110]**2, axis=0)).round(5),
          "max", np.max(np.abs(err110[use110]), axis=0).round(4))

print("\n=== DIRECT ATOM AND FIRST-HIDDEN DECODERS ===")
atomtrain110, atomtest110, targettrain110, targettest110 = [], [], [], []
for j110, drug110 in enumerate(["insulin", "norepinephrine"]):
    cap110 = float(caps105[j110])
    rt110 = np.r_[np.linspace(0, cap110, 2600), rng110.uniform(0, cap110, 2600)]
    rv110 = rng110.uniform(0, cap110, 1800)
    for rates110, atomlist110, targetlist110 in [
        (rt110, atomtrain110, targettrain110),
        (rv110, atomtest110, targettest110)
    ]:
        atoms110 = action_atom_exact(drug110, rates110).clone()
        atoms110[:, tracked105] = 0
        target110 = np.zeros((len(rates110), 4))
        target110[:, j110] = rates110
        target110[:, 2 + j110] = 1
        atomlist110.append(as_np(atoms110))
        targetlist110.append(target110)
    print(drug110, "rate normalization", norm_meta[(drug110, "rate")])

atrain110 = np.vstack(atomtrain110).astype(np.float64)
atest110 = np.vstack(atomtest110).astype(np.float64)
ytrain110 = np.vstack(targettrain110)
ytest110 = np.vstack(targettest110)
htrain110 = as_np(F.gelu(lin0(torch.tensor(atrain110, dtype=torch.float32),
                                 "encode_action.mlp1.0"))).astype(np.float64)
htest110 = as_np(F.gelu(lin0(torch.tensor(atest110, dtype=torch.float32),
                                "encode_action.mlp1.0"))).astype(np.float64)

firstfits110 = {}
for label110, x110, xv110 in [
    ("atom", atrain110, atest110),
    ("mlp1_hidden", htrain110, htest110)
]:
    mean110 = x110.mean(axis=0)
    sd110 = np.maximum(x110.std(axis=0), 0.02)
    candidates110 = []
    for alpha110 in [1e-7, 1e-5, 1e-3, 0.1]:
        fit110 = Ridge(alpha=alpha110).fit((x110-mean110)/sd110, ytrain110)
        e110 = fit110.predict((xv110-mean110)/sd110) - ytest110
        metrics110 = {
            "alpha": alpha110,
            "rmse": np.sqrt(np.mean(e110**2, axis=0)).round(6).tolist(),
            "max_abs": np.max(np.abs(e110), axis=0).round(5).tolist()
        }
        print(label110, metrics110)
        score110 = np.mean((e110 / np.array([1., .1, 1., 1.]))**2)
        candidates110.append((score110, fit110, metrics110))
    bestfirst110 = min(candidates110, key=lambda z: z[0])
    firstfits110[label110] = {
        "fit": bestfirst110[1],
        "coef": bestfirst110[1].coef_ / sd110,
        "intercept": bestfirst110[1].intercept_ -
                     (bestfirst110[1].coef_ / sd110) @ mean110,
        "metrics": bestfirst110[2]
    }

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 110: private-channel decoder audit\n"
            "- Audited the final-layer decoder by treatment configuration.\n"
            "- Tested direct-atom and first-hidden rate/identity recovery on independent continuous doses.\n"
            "- No additional weight edits; encoding accuracy alone is not causal or factual validation.\n")
print("Current edits:", edited())
```

The local directions work at small amplitudes, but larger injections become asymmetric and affect the other drug’s channels. I won’t install them unchecked. First I’ll test whether the relay’s remaining features can subtract factual marker signals without subtracting dose-edit contrasts. Success would mean small held-out factual residuals and preserved responses across all tested multipliers.

```python
import numpy as np

def arr222(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

q222 = arr222(load("core.prelude.0.attn.q.weight"))
kv222 = arr222(load("core.prelude.0.attn.kv.weight"))
out222 = arr222(load("core.prelude.0.attn.out.weight"))
headrows222 = np.flatnonzero(np.linalg.norm(q222, axis=1) < 1e-10)
print("uniform-head rows:", headrows222)
print("vocab config:", VOCAB["config"])
print("relay arrays:",
      {k: arr222(relay206[k]).shape
       for k in ["norm_tokens", "head_features", "target", "pool", "train", "hold"]})

assert len(headrows222) in [24, 32, 48, 64]
reserved222 = np.array([len(headrows222)-2, len(headrows222)-1])
other222 = np.setdiff1d(np.arange(len(headrows222)), reserved222)
WV222 = kv222[192 + headrows222].copy()
WV222[reserved222] = np.asarray(V215, dtype=np.float32)

Pfact222 = np.asarray(marker215["fact_pool"], dtype=np.float64)
Pbase222 = np.asarray(relay206["pool"], dtype=np.float64)
Phi222 = Pfact222 @ WV222.T
F222 = Phi222[:, reserved222]
Q222 = Phi222[:, other222]
train222 = np.asarray(relay206["train"], dtype=bool)
hold222 = np.asarray(relay206["hold"], dtype=bool)
assert train222.shape == (len(Q222),)
baseline222 = (
    Pbase222 @ kv222[192 + headrows222].T
    @ out222[:, headrows222].T
)

contrast222 = {}
Dlist222 = []
for key222, obj222 in marker215["cf"].items():
    pcf222 = np.asarray(obj222["pool"], dtype=np.float64)
    dphi222 = (pcf222 - Pfact222) @ WV222.T
    mask222 = np.asarray(obj222["mask"], dtype=bool)
    contrast222[key222] = (dphi222, mask222)
    dm222 = dphi222[train222 & mask222][:, other222]
    if len(dm222):
        Dlist222.append(dm222 / np.sqrt(len(dm222)))
D222 = np.concatenate(Dlist222, axis=0) / np.sqrt(len(Dlist222))

Qtr222 = Q222[train222]
Ftr222 = F222[train222]
gram222 = Qtr222.T @ Qtr222 / len(Qtr222)
rhs222 = Qtr222.T @ Ftr222 / len(Qtr222)
ridge222 = 1e-6 * np.trace(gram222) / len(other222)
basecoef222 = np.linalg.solve(
    gram222 + ridge222 * np.eye(len(other222)),
    Qtr222.T @ baseline222[train222] / len(Qtr222)
)
print("baseline reconstruction held-out RMSE:",
      np.sqrt(np.mean((Q222[hold222] @ basecoef222
                       - baseline222[hold222]) ** 2)))

comp222 = {}
for eta222 in [0., 0.01, 0.1, 1., 10., 100.]:
    cc222 = np.linalg.solve(
        gram222 + eta222 * (D222.T @ D222)
        + ridge222 * np.eye(len(other222)),
        rhs222
    )
    rr222 = F222 - Q222 @ cc222
    comp222[eta222] = {"coef": cc222, "residual": rr222}
    print("\neta", eta222,
          "held-out factual marker RMS",
          np.sqrt(np.mean(rr222[hold222] ** 2, axis=0)))
    for key222 in [
        ("insulin", 0.5), ("insulin", 2.0),
        ("norepinephrine", 0.5), ("norepinephrine", 2.0),
        ("norepinephrine", 3.0)
    ]:
        dd222, mm222 = contrast222[key222]
        use222 = hold222 & mm222
        retained222 = (
            dd222[:, reserved222] - dd222[:, other222] @ cc222
        )
        target222 = np.asarray(marker215["cf"][key222]["dq"])
        drugcol222 = 0 if key222[0] == "insulin" else 1
        print(key222,
              "target", round(float(target222[use222, drugcol222].mean()), 4),
              "retained", round(float(retained222[use222, drugcol222].mean()), 4),
              "cross", round(float(retained222[use222, 1-drugcol222].mean()), 4))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 222\n"
            "Finite-amplitude attention directions are nonlinear and cross-couple. "
            "Testing closed-form factual marker compensation using remaining "
            "uniform-head features, with dose-contrast-invariance penalties. "
            "No weights changed. Counts: 222 Python runs, 20 forwards.\n")
```
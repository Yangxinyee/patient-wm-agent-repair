The transition’s large state scale helps explain why small latent changes disappear. Before replacing an attention head, I’ll test a simpler option: whether the prelude representation already retains the gated insulin innovation. A decoder that generalizes across stays would support a targeted recurrent-matrix edit; poor held-out decoding would argue against that shortcut.

```python
import sys, numpy as np, torch

def capture_transition102(indices, action_override=None):
    captured = {}
    previous = sys.getprofile()
    def trace(frame, event, arg):
        if event == "return" and frame.f_code is transition73.__code__:
            for key in ("ctx", "hf", "ek", "uk", "zk", "tok", "ep", "hr", "hc", "out"):
                value = frame.f_locals.get(key)
                if torch.is_tensor(value):
                    captured[key] = value.detach().cpu().numpy().copy()
    sys.setprofile(trace)
    try:
        result = transition73(indices, action_override=action_override)
    finally:
        sys.setprofile(previous)
    return result, captured

indices102 = np.flatnonzero(np.asarray(bundle_to_row82) >= 0)
print("bundle-to-row mapping:", len(indices102), "target length:", len(g94),
      "row range:", np.asarray(bundle_to_row82)[indices102].min(),
      np.asarray(bundle_to_row82)[indices102].max())
assert len(indices102) == len(g94)
rows102 = np.asarray(bundle_to_row82)[indices102].astype(int)

Eparts102 = []
for start102 in range(0, len(indices102), 192):
    _, part102 = capture_transition102(indices102[start102:start102+192])
    Eparts102.append(part102["ep"])
Eslots102 = np.concatenate(Eparts102)
print("prelude shape:", Eslots102.shape,
      "gated innovation range:", float(np.min(g94)), float(np.max(g94)))
print("original posterior reserved units:", reserved94)

train102 = np.asarray(train94, bool)[rows102]
test102 = np.asarray(test94, bool)[rows102]
target102 = np.asarray(g94)[rows102]
active102 = np.asarray(actual_insulin95, bool)[rows102]
decoder_fits102 = {}
table102 = []

for mode102 in ("mean", "shared_slots"):
    if mode102 == "mean":
        feat102 = Eslots102.mean(axis=1)
        fit_feat102, fit_target102 = feat102[train102], target102[train102]
    else:
        feat102 = Eslots102.mean(axis=1)
        fit_feat102 = Eslots102[train102].reshape(-1, 192)
        fit_target102 = np.repeat(target102[train102], 4)
    mean102 = fit_feat102.mean(axis=0)
    std102 = np.maximum(fit_feat102.std(axis=0), 1e-5)
    A102 = np.column_stack(((fit_feat102-mean102)/std102, np.ones(len(fit_feat102))))
    V102 = np.column_stack(((feat102-mean102)/std102, np.ones(len(feat102))))
    gram102, rhs102 = A102.T@A102, A102.T@fit_target102
    for ridge102 in (0.0001, 0.01, 1.0, 100.0):
        penalty102 = np.eye(193)*ridge102
        penalty102[-1,-1] = 1e-8
        coef102 = np.linalg.solve(gram102+penalty102, rhs102)
        pred102 = V102@coef102
        beta102 = coef102[:-1]/std102
        bias102 = coef102[-1]-mean102@beta102
        decoder_fits102[(mode102, ridge102)] = (beta102, bias102)
        a102 = test102 & active102
        z102 = test102 & ~active102
        table102.append({
            "mode": mode102, "ridge": ridge102,
            "test_R2_all": 1-np.mean((pred102[test102]-target102[test102])**2)/
                           max(np.var(target102[test102]),1e-12),
            "test_R2_insulin": 1-np.mean((pred102[a102]-target102[a102])**2)/
                               max(np.var(target102[a102]),1e-12),
            "non_insulin_RMS": np.sqrt(np.mean(pred102[z102]**2)),
            "coefficient_norm": np.linalg.norm(beta102)
        })
print(__import__("pandas").DataFrame(table102).to_string(index=False))

print("intervention recipe dependencies:", intervention_bank74.__code__.co_names)
print("recipe string constants:",
      [v for v in intervention_bank74.__code__.co_consts if isinstance(v,str)])
print("bank keys:", list(repair_bank.keys()))
print("E65/U65 shapes:", tuple(E65.shape), tuple(U65.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 102\nCaptured prelude slot representations for the retained posterior-training bundles and tested stay-held-out linear decoding of the gated insulin innovation. This tests a simpler rank-one recurrent-matrix repair before modifying attention. No weights edited.\n")
```

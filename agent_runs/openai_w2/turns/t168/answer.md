The admission features have a large false insulin signal. A simple projection removes it but badly increases residuals on ordinary bundles, so I won’t install that version. I’ll instead find the correction that least changes factual and dose-probe predictions. It would be useful if admission responses reach zero while factual RMS and dose contrasts remain near their current values.

```python
import numpy as np, torch, inspect, pandas as pd

print("Attention helper:", inspect.signature(attention104))
F147 = np.asarray(prelude_E111, dtype=np.float64)
mu147 = np.asarray(fit128["mu"], dtype=np.float64)
sd147 = np.asarray(fit128["sd"], dtype=np.float64)
B147 = np.asarray(fit128["B"], dtype=np.float64)
X147 = np.column_stack([np.ones(F147.shape[0]*4),
                       ((F147-mu147)/sd147).reshape(-1, 192)])
C147 = np.column_stack([np.ones(4), (C146-mu147)/sd147])
Dparts147, Yparts147 = [], []
for key147 in prelude_cf_E111:
    dt147 = np.asarray(residual_targets102[key147], dtype=np.float64)
    active147 = np.any(np.abs(dt147) > 1e-10, axis=1)
    delta147 = (np.asarray(prelude_cf_E111[key147], dtype=np.float64)-F147)/sd147
    Dparts147.append(np.column_stack([np.zeros(active147.sum()*4),
                                     delta147[active147].reshape(-1,192)]))
    Yparts147.append(np.repeat(dt147[active147],4,axis=0))
D147 = np.concatenate(Dparts147)
Y147 = np.concatenate(Yparts147)
print("Factual / contrast designs:", X147.shape, D147.shape,
      "target RMS:", np.sqrt(np.mean(Y147**2, axis=0)))
print("Baseline affine vs folded max error:",
      np.max(np.abs(X147 @ B147 - F147.reshape(-1,192) @ Wdose134)))

def fold_decoder147(B):
    raw = B[1:]/sd147[:,None]
    intercept = B[0]-mu147@raw
    gamma = np.asarray(load_original("core.prelude_norm.weight"), dtype=np.float64)
    beta = np.asarray(load_original("core.prelude_norm.bias"), dtype=np.float64)
    return raw + np.outer(1/gamma, intercept/np.sum(beta/gamma))

def report_decoder147(w):
    fact = np.einsum("nsd,dk->nsk", F147, w)
    report = {"ins_rms": np.sqrt(np.mean(fact[:,:,0]**2)),
              "ne_rms": np.sqrt(np.mean(fact[:,:,1]**2)),
              "initial_max": np.max(np.abs(C146@w))}
    for key in prelude_cf_E111:
        drug, mult = key
        channel = 0 if drug == "insulin" else 1
        dose_delta = np.asarray(residual_targets102[key])[:,channel]
        pred_delta = np.einsum("nsd,dk->nsk",
            np.asarray(prelude_cf_E111[key], dtype=np.float64)-F147,w)
        report[("ins" if channel == 0 else "ne")+("_half_gain" if mult == .5 else "_x2_gain")] = (
            np.sum(pred_delta[:,:,channel]*dose_delta[:,None]) /
            (4*np.sum(dose_delta**2)))
    return report

Hfact147 = X147.T@X147/len(X147)
Hprobe147 = D147.T@D147/len(D147)
constraint_candidates147 = {}
reports147 = [{"candidate":"baseline", **report_decoder147(Wdose134)}]
for pw147 in [0.0, .03, .3, 3.0]:
    for ridge147 in [1e-6, 1e-4, .01]:
        reg147 = np.eye(193)*ridge147
        reg147[0,0] = 1e-12
        H147 = Hfact147 + pw147*Hprobe147 + reg147
        HC147 = np.linalg.solve(H147,C147.T)
        constrained147 = B147 - HC147 @ np.linalg.solve(C147@HC147,C147@B147)
        w147 = fold_decoder147(constrained147)
        key147 = (pw147,ridge147)
        constraint_candidates147[key147] = w147
        reports147.append({"candidate":str(key147), **report_decoder147(w147)})
print(pd.DataFrame(reports147).to_string(index=False))
```

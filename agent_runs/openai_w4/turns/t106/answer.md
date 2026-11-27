The existing posterior hidden features barely encode dose innovations, so changing only its output layer is unlikely to work. I’ll test a pair of gated units built from the better raw-input insulin decoder, compensating for the units they replace. This is still only a construction test: small held-out posterior error and negligible non-action leakage would justify testing its glucose effect.

```python
import torch
import torch.nn.functional as F
torch.set_num_threads(8)

W0_94 = np.asarray(load_original("z_post.0.weight"), dtype=np.float64)
b0_94 = np.asarray(load_original("z_post.0.bias"), dtype=np.float64)
W2_94 = np.asarray(load_original("z_post.2.weight"), dtype=np.float64)
b2_94 = np.asarray(load_original("z_post.2.bias"), dtype=np.float64)

def gelu94(x):
    return F.gelu(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(np.float64)

H94 = gelu94(raw87 @ W0_94.T + b0_94)
train94, test94 = np.asarray(fit82, bool), np.asarray(test82, bool)
Hc94 = H94[train94] - H94[train94].mean(axis=0)
cov94 = Hc94.T @ Hc94 / train94.sum()
precision94 = np.linalg.inv(cov94 + 1e-4 * np.eye(192))
importance94 = np.sum(W2_94[:32] ** 2, axis=0) / np.diag(precision94)
reserved94 = np.argsort(importance94)[:2]
kept94 = np.setdiff1d(np.arange(192), reserved94)

D94 = np.column_stack([H94[:, kept94], np.ones(len(H94))])
removed94 = H94[:, reserved94] @ W2_94[:, reserved94].T
pen94 = np.eye(D94.shape[1]) * 0.01
pen94[-1, -1] = 0
comp94 = np.linalg.solve(
    D94[train94].T @ D94[train94] + pen94,
    D94[train94].T @ removed94[train94]
)
W2base94 = W2_94.copy()
W2base94[:, kept94] += comp94[:-1].T
W2base94[:, reserved94] = 0
b2base94 = b2_94 + comp94[-1]
comp_error94 = D94 @ comp94 - removed94

decoder94 = raw_candidates87[("insulin", 0.1, 10.0)].copy()
gate_coef94 = np.zeros(576)
gate_coef94[384:] = gate_models88["insulin"].coef_[0]
gate_bias94 = float(gate_models88["insulin"].intercept_[0])
gate_scale94 = 24.0
W0base94, b0base94 = W0_94.copy(), b0_94.copy()
for sign, row in zip([1.0, -1.0], reserved94):
    W0base94[row] = gate_scale94 * gate_coef94 + sign * 0.5 * decoder94[:-1]
    b0base94[row] = gate_scale94 * gate_bias94 + sign * 0.5 * decoder94[-1]

def gated_innovation94(raw):
    h = gelu94(np.asarray(raw) @ W0base94[reserved94].T + b0base94[reserved94])
    return h[:, 0] - h[:, 1]

g94 = gated_innovation94(raw87)
active94 = np.asarray(repair_bank["has_insulin"])[
    np.flatnonzero(np.isclose(raw87[:, 384:], raw87[:, 384:]).all(axis=1))
] if False else (weights82["insulin"] > 0)
# labels82 is populated only where the record-based innovation is available.
labelled94 = active94 & test94 & np.isfinite(labels82["insulin"])
print("Reserved posterior units:", reserved94.tolist())
print("Compensation held-out mean-latent RMSE:",
      np.sqrt(np.mean(comp_error94[test94, :32] ** 2)))
print("Compensation held-out maximum absolute error:",
      np.max(np.abs(comp_error94[test94, :32])))
print("Gated held-out innovation RMSE:",
      np.sqrt(np.mean((g94[labelled94] - labels82["insulin"][labelled94]) ** 2)))
print("Inactive gated RMS:", np.sqrt(np.mean(g94[test94 & ~active94] ** 2)))
for factor in [0.5, 2.0]:
    delta = gated_innovation94(raw_cf87[("insulin", factor)]) - g94
    dr = cfdr82[("insulin", factor)]
    use = test94 & (np.abs(dr) > 1e-8)
    print("Controlled insulin", factor, "slope",
          float(delta[use] @ dr[use] / (dr[use] @ dr[use])))

print("\nHead datasets:")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and {"prev", "dt", "code"}.issubset(obj.columns):
        print(name, obj.shape, list(obj.columns), "prev range",
              (obj["prev"].min(), obj["prev"].max()))
print("State-bank names:",
      [(n, getattr(o["x"], "shape", None)) for n, o in list(globals().items())
       if isinstance(o, dict) and "x" in o and "stay_id" in o][:12])
print("Transition index arrays:",
      [(n, o.shape) for n, o in list(globals().items())
       if isinstance(o, np.ndarray) and o.shape == (4059,)
       and o.dtype.kind in "iu"][:15])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 94\n"
            "Constructed (not saved) two gated insulin-innovation posterior units. "
            "Original removed-unit outputs are compensated by a closed-form regression "
            "on retained hidden units. Testing held-out latent fidelity and non-action leakage "
            "before selecting an outcome-response direction. No edits.\n")
```

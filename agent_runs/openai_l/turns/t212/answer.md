The counterfactual inputs are available for all five dose factors. The check stopped at a float64/float32 mismatch, again without changing weights. I’ll now fit linear dose decoders and a glucose-versus-MAP gate, testing whether they generalize to held-out training stays before constructing a new head.

```python
import numpy as np, torch
from scipy.linalg import solve

torch.set_num_threads(8)
def arr192(x):
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)

I192 = arr192(I180)
H192 = arr192(H179)
W0192 = arr192(load_original("attr_heads.value_bin.0.weight"))
b0192 = arr192(load_original("attr_heads.value_bin.0.bias"))
W2192 = arr192(load_original("attr_heads.value_bin.2.weight"))
b2192 = arr192(load_original("attr_heads.value_bin.2.bias"))
hcheck192 = torch.nn.functional.gelu(torch.tensor(
    I192 @ W0192.T + b0192, dtype=torch.float32)).numpy()
print("Original hidden reconstruction RMS:",
      np.sqrt(np.mean((hcheck192 - H192)**2)))

importance192 = H192.reshape(-1, 192).std(0) * np.linalg.norm(
    W2192 - W2192.mean(0, keepdims=True), axis=0)
quiet192 = np.argsort(importance192)[:16]
print("Quiet units:", quiet192.tolist())
print("Importance:", np.round(importance192[quiet192], 5).tolist())

mu192 = I192[:, fitmask179].reshape(-1, 192).mean(0)
sd192 = np.maximum(I192[:, fitmask179].reshape(-1, 192).std(0), 0.05)
XF192 = np.concatenate([(I192 - mu192) / sd192,
                        np.ones(I192.shape[:-1] + (1,))], axis=-1)
Ffit192 = XF192[:, fitmask179].reshape(-1, 193)
CF192 = Ffit192.T @ Ffit192 / len(Ffit192)

Dparts192, Yparts192, maskparts192, slices192 = [], [], [], {}
offset192 = 0
for key, case in cf_features81.items():
    drug, factor = key
    if factor == 1.0:
        continue
    base = cf_features81[(drug, 1.0)]
    r = case["rows"]
    d = (arr192(case["input"]) - arr192(base["input"])) / sd192
    d = np.concatenate([d, np.zeros(d.shape[:-1] + (1,))], axis=-1).reshape(-1, 193)
    y = np.zeros((len(d), 2))
    j = 0 if drug == "insulin" else 1
    unit = 1.0 if j == 0 else 0.1
    y[:, j] = np.tile(np.asarray(case["delta_rate"]), 8) / unit
    mask = np.tile(fitmask179[r], 8)
    Dparts192.append(d)
    Yparts192.append(y)
    maskparts192.append(mask)
    slices192[key] = slice(offset192, offset192 + len(d))
    offset192 += len(d)
    if factor == 2.0:
        print("Base/input agreement", drug,
              np.sqrt(np.mean((arr192(base["input"]) - I192[:, r])**2)),
              "events", np.unique(events179[r], return_counts=True))

D192 = np.concatenate(Dparts192)
Y192 = np.concatenate(Yparts192)
maskD192 = np.concatenate(maskparts192)
CD192 = D192[maskD192].T @ D192[maskD192] / maskD192.sum()
RD192 = D192[maskD192].T @ Y192[maskD192] / maskD192.sum()

directions192 = {}
for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
    coef = solve(CD192 + lam * CF192 + 1e-10 * np.eye(193),
                 RD192, assume_a="pos")
    directions192[lam] = coef
    factual = XF192[:, ~fitmask179].reshape(-1, 193) @ coef
    pred = D192 @ coef
    report = {}
    for drug in ["insulin", "norepinephrine"]:
        j = 0 if drug == "insulin" else 1
        sl = slices192[(drug, 2.0)]
        keep = ~maskD192[sl]
        yt = Y192[sl, j][keep]
        yp = pred[sl, j][keep]
        report[drug] = round(float(yt @ yp / max(yt @ yt, 1e-12)), 4)
    print("lambda", lam,
          "factual RMS (dose units 1, .1)",
          np.round(np.sqrt(np.mean(factual**2, axis=0)), 4).tolist(),
          "contrast RMSE", np.round(np.sqrt(np.mean(
              (pred[~maskD192] - Y192[~maskD192])**2, axis=0)), 4).tolist(),
          "x2 slopes", report)

gate_y192 = np.tile(np.where(
    np.char.startswith(np.asarray(events179, dtype=str), "glucose"), 1., -1.),
    (8, 1))
gate192 = solve(CF192 + 0.01 * CD192 + 1e-6 * np.eye(193),
                Ffit192.T @ gate_y192[:, fitmask179].reshape(-1) / len(Ffit192),
                assume_a="pos")
gatepred192 = XF192 @ gate192
for label in [1., -1.]:
    vals = gatepred192[:, ~fitmask179][gate_y192[:, ~fitmask179] == label]
    print("Gate label", label, "quantiles",
          np.round(np.quantile(vals, [.01, .05, .5, .95, .99]), 3).tolist())
print("Gate validation classification error:",
      np.mean(np.sign(gatepred192[:, ~fitmask179]) != gate_y192[:, ~fitmask179]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 192 / pre-GELU feature diagnosis\n"
            "Fitting factual-anchored linear dose decoders across x0, "
            "half, x1.5, x2, x3 and a dose-invariant sensor-family gate. "
            "No new weights saved; current conservative edit retained.\n")
```
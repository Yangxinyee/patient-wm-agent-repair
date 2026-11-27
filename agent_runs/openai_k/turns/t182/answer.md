Both drug contrast sets are now aligned, with no missing bundle/drug keys. The computation then stopped because the local core requires a Torch tensor rather than a NumPy array. I’ll correct that interface and finish the assay. Accurate held-out slopes, little cross-drug leakage, and low baseline noise would support the decoder route.

```python
import numpy as np, torch
from scipy.linalg import null_space

u166cf = torch.as_tensor(u164cf, dtype=torch.float32)
oc164, hc164 = local_token154(token164(u166cf), u=u166cf)
q164base = np.asarray(dynamic_views154(o164, h164)["query"], dtype=float)
q164cf = np.asarray(dynamic_views154(oc164, hc164)["query"], dtype=float)
dq164 = q164cf - q164base

gamma164 = np.asarray(load_original("query_norm.weight"), dtype=float)
nuisance164 = np.column_stack([
    np.asarray(load_original("last_numeric.weight"), dtype=float),
    np.asarray(load_original("embed.code.weight"), dtype=float).T,
    np.asarray(load_original("embed.bins.value.weight"), dtype=float).T,
    np.asarray(load_original("embed.bins.rate.weight"), dtype=float).T,
    np.asarray(load_original("last_status.weight"), dtype=float).T,
    gamma164[:, None] * np.asarray(load_original("z_head.weight"), dtype=float),
])
basis164 = null_space(nuisance164.T, rcond=1e-7)
qmean164 = q163o.mean(axis=0)
noise164 = (q163o-qmean164) @ basis164
A164 = dq164 @ basis164
Y164 = np.zeros((len(dc164), 2))
for j in (0, 1):
    m = np.asarray(drug151) == j
    Y164[m, j] = dc164[m]

tr164 = np.asarray(training158, dtype=bool)
va164 = np.asarray(validation158, dtype=bool)
decoders164 = {}
print("register-independent dimensions:", basis164.shape[1])
print("local-query dose-change RMS:", np.sqrt(np.mean(dq164**2)))

print("\n=== STAY-HELD-OUT DOSE DECODING ===")
for penalty in (10., 1., .1, .01, .001):
    G = A164[tr164].T @ A164[tr164] / tr164.sum()
    G += penalty * noise164.T @ noise164 / len(noise164)
    G += 1e-8 * np.eye(G.shape[0])
    coef = np.linalg.solve(G, A164[tr164].T @ Y164[tr164] / tr164.sum())
    decoder = basis164 @ coef
    pred = dq164[va164] @ decoder
    slopes = []
    for j in (0, 1):
        m = va164 & (np.asarray(drug151) == j)
        slopes.append(np.mean((dq164[m] @ decoder) / dc164[m, None], axis=0))
    print("penalty", penalty,
          "held-out RMSE", np.round(np.sqrt(np.mean((pred-Y164[va164])**2, axis=0)), 3),
          "slopes", np.round(slopes, 3).tolist(),
          "original-noise SD",
          np.round(((q163o-qmean164) @ decoder).std(axis=0), 4),
          "decoder norms", np.round(np.linalg.norm(decoder, axis=0), 2))
    decoders164[penalty] = decoder

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Runs 164–166: complete register-independent decoder assay\n"
            "- Recovered both drug contrast sets, aligned by bundle and drug; fixed NumPy/Torch action interface.\n"
            "- Decoders trained on exact local contrasts with a penalty on original-query noise; held-out diagnostics printed.\n"
            "- No weights changed. Python runs: 166; forward passes: 10.\n")
```

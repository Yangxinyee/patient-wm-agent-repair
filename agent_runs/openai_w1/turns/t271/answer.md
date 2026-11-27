The latest output is still run 238; the projection test has not been reported. I’ll keep the validated weights unchanged and run that test. The decisive result is held-out dose separation relative to factual leakage, with sensible scaling beyond x2.

```python
import numpy as np
from pathlib import Path

tr239 = np.asarray(train213, dtype=bool)
te239 = np.asarray(hold213, dtype=bool)
cf239 = {}
for key, x in rp_cf_input.items():
    z, token = posterior238(x, safe236)
    action = array238(x)[:, 384:576]
    dp = (
        array238(norm215(action)) - array238(norm215(u205))
        + array238(norm215(token)) - array238(norm215(ts238))
    ) / 7.0
    cf239[key] = {
        "z": z, "token": token, "dp": dp,
        "pool": pools238 + dp,
        "mask": np.asarray(rp_cf_masks[key[0]], dtype=bool)
    }

print("HELD-OUT LATENT CONTRASTS")
for drug in ["insulin", "norepinephrine"]:
    for mult in [0.0, 0.5, 1.5, 2.0, 3.0]:
        c = cf239[(drug, mult)]
        m = c["mask"] & te239
        dz = c["z"][m][:, [9, 16]] - zs238[m][:, [9, 16]]
        print(drug, mult, "n", int(m.sum()),
              "dz", np.round(dz.mean(axis=0), 3),
              "pool RMS", round(float(np.sqrt(np.mean(c["dp"][m]**2))), 4))

P239 = pools238.astype(np.float64)
C239 = P239[tr239].T @ P239[tr239] / tr239.sum()
D239 = np.stack([
    cf239[(d, 2.0)]["dp"][cf239[(d, 2.0)]["mask"] & tr239].mean(axis=0)
    for d in ["insulin", "norepinephrine"]
]).astype(np.float64)

projections239 = {}
print("\nLINEAR PROJECTIONS")
for alpha in [1e-6, 1e-4, 1e-2, 1e-1]:
    V = np.linalg.solve(C239 + alpha*np.eye(192), D239.T)
    V = V @ np.linalg.pinv(D239 @ V)
    projections239[alpha] = V.astype(np.float32)
    f = P239[te239] @ V
    print("ridge", alpha, "norm", np.linalg.norm(V, axis=0).round(2),
          "factual RMS", np.sqrt(np.mean(f*f, axis=0)).round(4))
    for drug in ["insulin", "norepinephrine"]:
        results = []
        for mult in [0.0, 0.5, 1.5, 2.0, 3.0]:
            c = cf239[(drug, mult)]
            m = c["mask"] & te239
            results.append(np.round((c["dp"][m] @ V).mean(axis=0), 3).tolist())
        print(drug, results)

print("\nIMPULSE KERNEL OBJECTS")
for name in ["he_kernels", "eh_kernels", "kernel_recovery"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name)
        for k, v in obj.items():
            if isinstance(v, dict):
                print(k, {str(a): describe236(b) for a, b in list(v.items())[:5]})
            else:
                a = array238(v)
                print(k, a.round(4).tolist() if a.size <= 24 else a.shape)

p239 = Path("/work/notes.md")
if "\n## Run 239\n" not in p239.read_text():
    with p239.open("a") as f:
        f.write(
            "\n\n## Run 239\n"
            "Built all ten safe-carrier dose contrasts. Tested linear projection "
            "separation, held-out factual leakage, and x0/.5/1.5/2/3 scaling. "
            "No weights changed. Counts: 239 Python runs, 23 forwards.\n"
        )
```
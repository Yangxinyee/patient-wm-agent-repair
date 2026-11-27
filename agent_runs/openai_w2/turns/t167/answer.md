The repeated output confirms that admission bundles were excluded from the feature fit. It does not yet test the correction. I’ll now reconstruct their features and measure how enforcing zero dose signal changes the factual residuals. The correction is promising only if it removes the admission response without substantially increasing residuals elsewhere.

```python
import numpy as np, torch, re
from pathlib import Path

initmask146 = np.r_[True, S_object120["stay_id"][1:] != S_object120["stay_id"][:-1]]
initidx146 = np.flatnonzero(initmask146)
print("Initial bundles:", len(initidx146), "hours:", np.unique(S_object120["hour"][initidx146]))
empty_u146 = encode_variable_sets_repair([[]], "encode_action", zero_empty=False)
init_h146 = torch.zeros((len(initidx146), 4, 192), dtype=torch.float32)
init_e146 = all_e104[initidx146].clone()
init_u146 = empty_u146.repeat(len(initidx146), 1)
init_z146 = torch.tensor(S_object120["x"][initidx146, 960:], dtype=torch.float32)
init_features146 = prelude_features111(init_h146, init_e146, init_u146, init_z146)
if isinstance(init_features146, torch.Tensor):
    init_features146 = (init_features146,)
print("Prelude return shapes:", [tuple(x.shape) for x in init_features146])
print("Initial e variation:", float(torch.max(torch.abs(init_e146 - init_e146[:1]))))
print("Initial z variation:", float(torch.max(torch.abs(init_z146 - init_z146[:1]))))

gamma146 = torch.as_tensor(load_original("core.prelude_norm.weight"))
beta146 = torch.as_tensor(load_original("core.prelude_norm.bias"))
lnchecks146 = []
for j146, x146 in enumerate(init_features146):
    if tuple(x146.shape[1:]) == (4, 192):
        n146 = (x146 - beta146) / gamma146
        err146 = float(n146.mean(-1).abs().mean() +
                       (n146.square().mean(-1) - 1).abs().mean())
        lnchecks146.append((err146, j146))
print("Final-normalization checks:", lnchecks146)
init_E146 = init_features146[min(lnchecks146)[1]].detach().numpy().astype(np.float64)
C146 = init_E146[0]
print("Initial dose-decoder outputs by slot:\n", C146 @ Wdose134)
print("Constraint singular values:", np.linalg.svd(C146, compute_uv=False))
Wzero146 = Wdose134 - np.linalg.pinv(C146, rcond=1e-9) @ (C146 @ Wdose134)
print("Projected initial outputs:\n", C146 @ Wzero146)
for label146, w146 in [("current", Wdose134), ("Euclidean zero-admission", Wzero146)]:
    pred146 = np.einsum("nsd,dk->nsk", np.asarray(prelude_E111), w146)
    print(label146, "factual RMS:", np.sqrt(np.mean(pred146**2, axis=(0, 1))),
          "factual mean:", pred146.mean(axis=(0, 1)))

print("=== FIT AND CONTRAST HELPERS ===")
print([k for k in globals() if callable(globals()[k]) and
       any(s in k.lower() for s in ["fit", "direct", "probe"])])
print("=== FEATURE DICTIONARIES ===")
for k146, v146 in list(globals().items()):
    if isinstance(v146, dict) and any(s in k146.lower() for s in ["prelude", "cf", "fit112", "fit128"]):
        print(k146, {str(k): getattr(v, "shape", type(v).__name__) for k, v in v146.items()})
print("=== NOTES ON DECODER CONSTRUCTION ===")
for s146 in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if any(t in s146.split("\n")[0] for t in ["111", "112", "113", "114", "115", "116", "128"]):
        print(s146[:1700])
```

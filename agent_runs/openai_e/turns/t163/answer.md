The raw norepinephrine dose units and reading-bin centres check out. However, the historical projected features differ from the recomputed features, and the sensitivity map differs by about 1.7%. I won’t reuse those arrays blindly. I’ll check for a basis rotation or omitted scale constant and construct fresh admission-null features; unexplained discrepancies would stop the fit.

```python
import numpy as np
import torch
import types
from pathlib import Path

print("HISTORICAL PROJECTION AUDIT")
sample142 = np.linspace(0, len(Fslot141) - 1, 128, dtype=int)
oldF142 = arr141(XF95)[sample142]
rawF142 = EF141.reshape(-1, 192)[sample142]
matches142 = []
for name142, obj142 in list(globals().items()):
    if name142.startswith("_") or not isinstance(obj142, (np.ndarray, torch.Tensor)):
        continue
    if tuple(obj142.shape) == (192, 188):
        b142 = arr141(obj142).astype(np.float64)
        err142 = float(np.sqrt(np.mean((rawF142 @ b142 - oldF142)**2)))
        matches142.append((err142, name142))
print("closest historical bases", sorted(matches142)[:5])

print("\nRECURRENT SCALE AUDIT")
unmapped142 = np.linalg.solve(Cout141, Praw141)
rowden142 = np.maximum(np.sum(Braw141**2, axis=1), 1e-15)
inferred_delta142 = np.sum(unmapped142 * Braw141, axis=1) / rowden142
print("diagonal-map reconstruction relative error",
      float(np.linalg.norm(unmapped142 - inferred_delta142[:, None] * Braw141)
            / np.linalg.norm(unmapped142)))
print("inferred minus softplus quantiles",
      np.round(np.quantile(inferred_delta142 - delta141.reshape(-1), [0, .5, 1]), 8))
map_errors142 = {}
for offset142 in [0.0, 1e-4, 1e-3, 1e-2]:
    p142 = Cout141 @ ((delta141.reshape(-1) + offset142)[:, None] * Braw141)
    map_errors142[offset142] = float(np.linalg.norm(p142 - Praw141) / np.linalg.norm(Praw141))
print("candidate additive scale errors", map_errors142)
for name142, obj142 in sorted(list(globals().items())):
    if isinstance(obj142, types.FunctionType) and "core.delta_in" in obj142.__code__.co_consts:
        print("scale helper", name142, "numeric constants",
              [x for x in obj142.__code__.co_consts if isinstance(x, (int, float))][:15])

print("\nUNCHANGED-DOSE CONTRAST CHECK")
contrast_errors142 = {}
for key142 in E_edit95:
    action142 = "has_insulin" if key142[0] == "insulin" else "has_norepinephrine"
    absent142 = ~cf_frame[action142].to_numpy(dtype=bool)
    deltaE142 = arr141(E_edit95[key142]) - EF141[cfrows141]
    contrast_errors142[key142] = float(np.max(np.abs(deltaE142[absent142])))
    print(key142, "max contrast without that drug", contrast_errors142[key142])

first_rows142 = state_df.sort_values(["stay_id", "hour"]).groupby("stay_id", sort=False).first()
first_idx142 = first_rows142["state_row"].to_numpy(dtype=int)
admission142 = EF141[first_idx142].reshape(-1, 192)
_, adm_s142, adm_Vt142 = np.linalg.svd(admission142, full_matrices=True)
adm_rank142 = int(np.sum(adm_s142 > adm_s142[0] * 1e-5))
Binput142 = adm_Vt142[adm_rank142:].T.copy()
Fmean142 = EF141.mean(axis=1) @ Binput142
Dmean142 = {
    key: (arr141(value) - EF141[cfrows141]).mean(axis=1) @ Binput142
    for key, value in E_edit95.items()
}
print("\nFRESH ADMISSION-NULL FEATURES")
print("admission rank", adm_rank142, "input dimensions", Binput142.shape[1],
      "initial residual RMS", float(np.sqrt(np.mean((admission142 @ Binput142)**2))))
print("leading admission singular values", np.round(adm_s142[:8], 6))
ready142 = (
    min(map_errors142.values()) < 1e-5
    and max(contrast_errors142.values()) < 1e-4
    and adm_rank142 <= 8
    and centres141.shape == prob141.shape
)
print("ready for consistent output-constrained fit:", ready142)

with open("/work/notes.md", "a") as f142:
    f142.write(
        "\n## Run 142: resolve projection and recurrent-scale discrepancies\n"
        f"Historical basis matches: {sorted(matches142)[:2]}. "
        f"Recurrent scale-map errors: {map_errors142}. "
        f"Unchanged-dose contrast maxima: {contrast_errors142}. "
        f"Fresh initial-prelude null rank={adm_rank142}; ready={ready142}. "
        "No weights changed. Python runs: 142; forwards: 15.\n"
    )
```
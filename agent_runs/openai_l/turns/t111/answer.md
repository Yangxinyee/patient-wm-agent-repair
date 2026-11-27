The run stopped at a sequence-boundary assertion before producing any fit. I’ll exclude bundles without a preceding state and verify the reconstructed prelude against the existing transition code. Only a matching reconstruction and a selective dose-deviation signal would justify testing an edit here.

```python
import numpy as np, torch
from pathlib import Path

ctx_all102.fill_(float("nan"))
hf_all102.fill_(float("nan"))
E_all102.fill_(float("nan"))
valid_list103, invalid_list103 = [], []

def prepare103(ids):
    try:
        ctx, hf, _, _, _ = bundle_inputs74(ids)
    except AssertionError:
        if len(ids) == 1:
            invalid_list103.append(int(ids[0]))
        else:
            mid = len(ids) // 2
            prepare103(ids[:mid])
            prepare103(ids[mid:])
        return
    with torch.no_grad():
        ctx_all102[ids] = ctx
        hf_all102[ids] = hf
        E_all102[ids] = prelude_features102(
            ctx, hf, e_valid60[ids], u_valid60[ids]
        )
    valid_list103.extend(map(int, ids))

for start103 in range(0, n102, 96):
    prepare103(np.arange(start103, min(start103 + 96, n102)))
valid_indices103 = np.asarray(sorted(valid_list103), dtype=int)
valid_mask103 = np.zeros(n102, dtype=bool)
valid_mask103[valid_indices103] = True
print("Valid / boundary bundles:", len(valid_indices103), len(invalid_list103))

captured103 = {}
norm_backup103 = norm63
def capture_norm103(x, prefix):
    result = norm_backup103(x, prefix)
    if prefix == "core.prelude_norm":
        captured103["E"] = result.clone()
    return result

ids103 = valid_indices103[:8]
ctx103, hf103, mem103, age103, pad103 = bundle_inputs74(ids103)
with torch.no_grad():
    z103 = mlp63(torch.cat([
        ctx103, e_valid60[ids103], u_valid60[ids103]
    ], -1), "z_post")[:, :32]
    norm63 = capture_norm103
    try:
        transition64(hf103, e_valid60[ids103], u_valid60[ids103],
                     z103, mem103, age103, ~pad103)
    finally:
        norm63 = norm_backup103
prelude_error103 = float(
    (captured103["E"] - E_all102[ids103]).square().mean().sqrt()
)
print("Prelude reconstruction RMS:", prelude_error103)

dose_by_bundle102, E_cf102 = {}, {}
for drug103 in ("insulin", "norepinephrine"):
    chosen103 = np.asarray(drug_names81) == drug103
    dose_by_bundle102[drug103] = np.bincount(
        np.asarray(owners72, dtype=int),
        weights=np.where(chosen103, np.asarray(rates65), 0.0),
        minlength=n102
    )
    ids103 = np.flatnonzero(
        (dose_by_bundle102[drug103] > 0) & valid_mask103
    )
    for factor103 in (0.0, 0.5, 2.0, 3.0):
        rr103 = np.asarray(rates65).copy()
        dd103 = np.asarray(doses65).copy()
        rr103[chosen103] *= factor103
        dd103[chosen103] *= factor103
        with torch.no_grad():
            uc103 = pool_actions49(
                action_embedding67(drugs65, rr103, dd103),
                owners72, counts72
            )
            pieces103 = []
            for start103 in range(0, len(ids103), 96):
                jj103 = ids103[start103:start103 + 96]
                pieces103.append(prelude_features102(
                    ctx_all102[jj103], hf_all102[jj103],
                    e_valid60[jj103], uc103[jj103]
                ))
        E_cf102[(drug103, factor103)] = {
            "indices": ids103,
            "features": torch.cat(pieces103),
            "delta_rate": (factor103 - 1) * dose_by_bundle102[drug103][ids103]
        }

Ef103 = E_all102[valid_indices103].reshape(-1, 192).numpy().astype(np.float64)
Emean102 = Ef103.mean(0)
Ec103 = Ef103 - Emean102
Sf103 = Ec103.T @ Ec103 / len(Ec103)
fit_summary103, extractors103 = {}, {}

for drug103 in ("insulin", "norepinephrine"):
    D103, y103 = [], []
    for (d103, factor103), c103 in E_cf102.items():
        dx103 = (
            c103["features"] - E_all102[c103["indices"]]
        ).reshape(-1, 192).numpy().astype(np.float64)
        target103 = c103["delta_rate"] if d103 == drug103 else np.zeros(len(c103["indices"]))
        D103.append(dx103)
        y103.append(np.repeat(target103, 4))
    D103, y103 = np.vstack(D103), np.concatenate(y103)
    Sc103 = D103.T @ D103 / len(D103)
    bc103 = D103.T @ y103 / len(D103)
    fit_summary103[drug103], extractors103[drug103] = {}, {}
    for penalty103 in (0.001, 0.01, 0.1, 1.0):
        v103 = np.linalg.solve(
            Sc103 + penalty103 * Sf103 + 1e-6 * np.eye(192), bc103
        )
        extractors103[drug103][penalty103] = v103
        fit_summary103[drug103][penalty103] = {
            "contrast_RMSE": float(np.sqrt(np.mean((D103 @ v103 - y103)**2))),
            "target_RMS": float(np.sqrt(np.mean(y103**2))),
            "factual_signal_SD": float(np.std(Ec103 @ v103)),
            "coefficient_norm": float(np.linalg.norm(v103)),
        }
print("Dose-residual extraction:", fit_summary103)
print("State metadata:", type(sx63).__name__, getattr(sx63, "shape", None))
if hasattr(sx63, "columns"):
    print("State columns:", list(sx63.columns))

with Path("/work/notes.md").open("a") as f103:
    f103.write(
        "\n## Runs 102–103 / boundary-safe prelude probe\n"
        f"Run 102 stopped at boundary assertion; excluded {len(invalid_list103)} bundles. "
        f"Prelude reconstruction RMS={prelude_error103}. Extractor diagnostics={fit_summary103}. "
        "No weights edited.\n"
    )
```

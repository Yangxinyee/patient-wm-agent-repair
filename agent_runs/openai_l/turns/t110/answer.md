The factor-one check passes: the retained counterfactual computations reproduce factual predictions to about 0.001 in mean units. That rules out a major alignment error. I’ll test whether the prelude contains a linear dose-deviation signal that is much stronger under intervention than on factual histories. If not, this edit location is unsuitable.

```python
import numpy as np, torch
from pathlib import Path

torch.set_num_threads(8)

def prelude_features102(ctx, hf, e, u):
    z = mlp63(torch.cat([ctx, e, u], dim=-1), "z_post")[:, :32]
    tokens = torch.cat([
        hf + w63("core.slots"),
        e[:, None, :],
        u[:, None, :],
        linear63(z, "z_in")[:, None, :]
    ], dim=1)
    return norm63(layer63(tokens, "core.prelude.0")[:, :4, :],
                  "core.prelude_norm")

n102 = len(e_valid60)
ctx_all102 = torch.empty((n102, 192))
hf_all102 = torch.empty((n102, 4, 192))
E_all102 = torch.empty((n102, 4, 192))

with torch.no_grad():
    for start102 in range(0, n102, 96):
        ids102 = np.arange(start102, min(start102 + 96, n102))
        ctx102, hf102, _, _, _ = bundle_inputs74(ids102)
        ctx_all102[ids102] = ctx102
        hf_all102[ids102] = hf102
        E_all102[ids102] = prelude_features102(
            ctx102, hf102, e_valid60[ids102], u_valid60[ids102]
        )

dose_by_bundle102 = {}
E_cf102 = {}
for drug102 in ("insulin", "norepinephrine"):
    chosen102 = np.asarray(drug_names81) == drug102
    dose_by_bundle102[drug102] = np.bincount(
        np.asarray(owners72, dtype=int),
        weights=np.where(chosen102, np.asarray(rates65), 0.0),
        minlength=n102
    )
    ids102 = np.flatnonzero(dose_by_bundle102[drug102] > 0)
    for factor102 in (0.0, 0.5, 2.0, 3.0):
        rr102 = np.asarray(rates65).copy()
        dd102 = np.asarray(doses65).copy()
        rr102[chosen102] *= factor102
        dd102[chosen102] *= factor102
        with torch.no_grad():
            u102 = pool_actions49(
                action_embedding67(drugs65, rr102, dd102),
                owners72, counts72
            )
            parts102 = []
            for start102 in range(0, len(ids102), 96):
                jj102 = ids102[start102:start102 + 96]
                parts102.append(prelude_features102(
                    ctx_all102[jj102], hf_all102[jj102],
                    e_valid60[jj102], u102[jj102]
                ))
            ec102 = torch.cat(parts102)
        E_cf102[(drug102, factor102)] = {
            "indices": ids102,
            "features": ec102,
            "delta_rate": (factor102 - 1) * dose_by_bundle102[drug102][ids102]
        }

Ef102 = E_all102.mean(1).numpy().astype(np.float64)
Emean102 = Ef102.mean(0)
Ec102 = Ef102 - Emean102
Sf102 = Ec102.T @ Ec102 / len(Ec102)

fit_summary102 = {}
extractors102 = {}
for drug102 in ("insulin", "norepinephrine"):
    D102, y102 = [], []
    for (d102, factor102), c102 in E_cf102.items():
        ids102 = c102["indices"]
        dx102 = c102["features"].mean(1).numpy().astype(np.float64) - Ef102[ids102]
        D102.append(dx102)
        y102.append(c102["delta_rate"] if d102 == drug102
                    else np.zeros(len(ids102)))
    D102 = np.vstack(D102)
    y102 = np.concatenate(y102)
    Sc102 = D102.T @ D102 / len(D102)
    bc102 = D102.T @ y102 / len(D102)
    fit_summary102[drug102] = {}
    extractors102[drug102] = {}
    for penalty102 in (0.001, 0.01, 0.1, 1.0):
        v102 = np.linalg.solve(
            Sc102 + penalty102 * Sf102 + 1e-6 * np.eye(192), bc102
        )
        extractors102[drug102][penalty102] = v102
        fit_summary102[drug102][penalty102] = {
            "contrast_RMSE": float(np.sqrt(np.mean((D102 @ v102 - y102)**2))),
            "target_RMS": float(np.sqrt(np.mean(y102**2))),
            "factual_signal_SD": float(np.std(Ec102 @ v102)),
            "coefficient_norm": float(np.linalg.norm(v102)),
        }

print("Prelude feature shape / finite:", tuple(E_all102.shape),
      bool(torch.isfinite(E_all102).all()))
print("Dose-residual extraction:", fit_summary102)
print("Per-drug active bundles:",
      {d: int(np.count_nonzero(a)) for d, a in dose_by_bundle102.items()})

with Path("/work/notes.md").open("a") as f102:
    f102.write(
        "\n## Run 102 / prelude dose-residual extraction\n"
        f"Factor-one equivalence passed. Prelude extractor diagnostics: {fit_summary102}. "
        "These are feasibility diagnostics, not causal estimates or validated edits. No edits.\n"
    )
```

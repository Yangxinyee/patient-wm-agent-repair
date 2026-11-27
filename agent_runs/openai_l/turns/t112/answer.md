The prelude reconstruction is exact, but its dose extractor loses much of the intervention signal. I won’t edit that compressed representation yet. I’ll test the earlier posterior inputs, where treatment and history remain separate. A substantially cleaner separation there would support a more targeted repair.

```python
import numpy as np, torch
from pathlib import Path

ids104 = valid_indices103
with torch.no_grad():
    X104 = torch.cat([
        ctx_all102[ids104], e_valid60[ids104], u_valid60[ids104]
    ], -1)
    H104 = torch.nn.functional.gelu(linear63(X104, "z_post.0"))

raw104 = X104.numpy().astype(np.float64)
hidden104 = H104.numpy().astype(np.float64)
mean_raw104 = raw104.mean(0)
mean_hidden104 = hidden104.mean(0)
Rc104 = raw104 - mean_raw104
Hc104 = hidden104 - mean_hidden104
Sf_raw104 = Rc104.T @ Rc104 / len(Rc104)
Sf_hidden104 = Hc104.T @ Hc104 / len(Hc104)

contrasts104 = []
with torch.no_grad():
    for drug104 in ("insulin", "norepinephrine"):
        chosen104 = np.asarray(drug_names81) == drug104
        jj104 = np.flatnonzero(
            (dose_by_bundle102[drug104] > 0) & valid_mask103
        )
        base_x104 = torch.cat([
            ctx_all102[jj104], e_valid60[jj104], u_valid60[jj104]
        ], -1)
        base_h104 = torch.nn.functional.gelu(linear63(base_x104, "z_post.0"))
        for factor104 in (0.0, 0.5, 1.5, 2.0, 3.0):
            rr104 = np.asarray(rates65).copy()
            dd104 = np.asarray(doses65).copy()
            rr104[chosen104] *= factor104
            dd104[chosen104] *= factor104
            uc104 = pool_actions49(
                action_embedding67(drugs65, rr104, dd104),
                owners72, counts72
            )
            xc104 = base_x104.clone()
            xc104[:, 384:] = uc104[jj104]
            hc104 = torch.nn.functional.gelu(linear63(xc104, "z_post.0"))
            contrasts104.append({
                "drug": drug104, "factor": factor104, "indices": jj104,
                "raw": (xc104 - base_x104).numpy().astype(np.float64),
                "hidden": (hc104 - base_h104).numpy().astype(np.float64),
                "delta_rate": (factor104 - 1) * dose_by_bundle102[drug104][jj104]
            })

extractor_results104, extractors104 = {}, {}
for space104, factual104, sf104 in (
    ("raw", Rc104, Sf_raw104), ("hidden", Hc104, Sf_hidden104)
):
    D104 = np.vstack([c[space104] for c in contrasts104])
    sc104 = D104.T @ D104 / len(D104)
    for drug104 in ("insulin", "norepinephrine"):
        y104 = np.concatenate([
            c["delta_rate"] if c["drug"] == drug104
            else np.zeros(len(c["indices"]))
            for c in contrasts104
        ])
        b104 = D104.T @ y104 / len(D104)
        for penalty104 in (0.01, 0.1, 1.0):
            v104 = np.linalg.solve(
                sc104 + penalty104 * sf104
                + 1e-6 * np.eye(D104.shape[1]), b104
            )
            key104 = (space104, drug104, penalty104)
            extractors104[key104] = v104
            per_factor104 = {}
            for c104 in contrasts104:
                if c104["drug"] == drug104:
                    pred104 = c104[space104] @ v104
                    dy104 = c104["delta_rate"]
                    per_factor104[c104["factor"]] = round(
                        float(pred104 @ dy104 / (dy104 @ dy104)), 3
                    )
            extractor_results104[key104] = {
                "contrast_RMSE": round(float(np.sqrt(np.mean((D104 @ v104-y104)**2))), 4),
                "factual_SD": round(float(np.std(factual104 @ v104)), 4),
                "dose_slopes": per_factor104
            }

for key104, result104 in extractor_results104.items():
    print(key104, result104)

print("\nBundle-input dependencies:", bundle_inputs73.__code__.co_names)
for name104 in bundle_inputs73.__code__.co_names:
    obj104 = globals().get(name104)
    if isinstance(obj104, dict):
        print(name104, "keys:", list(obj104)[:15])
    elif hasattr(obj104, "columns"):
        print(name104, "columns:", list(obj104.columns))
    elif isinstance(obj104, np.ndarray):
        print(name104, obj104.shape, obj104.dtype)

with Path("/work/notes.md").open("a") as f104:
    f104.write(
        "\n## Run 104 / posterior-input residual feasibility\n"
        f"Compared raw and hidden posterior features: {extractor_results104}. "
        "Prelude-only extraction is too lossy; earlier features may retain treatment/history "
        "separation. No weights edited.\n"
    )
```

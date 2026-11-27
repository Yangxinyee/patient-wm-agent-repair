The archive reconstructs exactly after removing glucose. However, all four MAP directions have similar factual cost, so changing direction alone is unlikely to solve the problem. I’ll measure how each slot actually affects the output probabilities and use that to fit a more efficient dose feature. Lower held-out distortion at the same dose response would confirm the improvement.

```python
import numpy as np, pandas as pd, torch

# Recover record-based targets and validation identifiers without changing weights.
r275_notes = open("/work/notes.md").read()
for heading in ["## Run 6:", "## Run 7:"]:
    start = r275_notes.find(heading)
    if start >= 0:
        end = r275_notes.find("\n## ", start + len(heading))
        print(r275_notes[start:end if end >= 0 else start+3000][:3000])

print("=== RATE-FIT METADATA ===")
for name, value in list(globals().items()):
    if name.startswith("r245_") and not callable(value):
        print(name, r272_summary(value))

r275_factual24 = pd.read_csv("/work/fwd_24/factual_edited.csv")
print("Factual CSV columns:", r275_factual24.columns.tolist())
r275_staycol24 = next(c for c in r275_factual24.columns if "stay" in c.lower())
r275_old_ids = set(r275_factual24[r275_staycol24].astype(int).unique())
r275_header = pd.read_csv("/data/tuning.csv", nrows=1)
r275_staycol = next(c for c in r275_header.columns if "stay" in c.lower())
r275_all_ids = sorted(
    pd.read_csv("/data/tuning.csv", usecols=[r275_staycol])[r275_staycol]
      .astype(int).unique().tolist()
)
r275_new_ids = [i for i in r275_all_ids if i not in r275_old_ids][:100]
print("Untested factual validation stays:", r275_new_ids)

# Finite differences of scalar injection into each slot, not model gradients.
r275_ridge = min(r267_routes, key=lambda k: abs(float(k)-1.0))
r275_direction = np242(r267_routes[r275_ridge]).astype(np.float64)
r275_eps = 0.05
r275_pre = r274_pre(r274_case, r272_glucose)
r275_mean0, r275_prob0, _ = r267_full_pre(r275_pre, indices151)
r275_P = np.stack([np242(p) for p in r275_prob0], axis=1).astype(np.float64)
r275_Jslot_mean = np.empty((len(indices151), 4, 4), dtype=np.float64)
r275_Jslot_log = np.empty((len(indices151), 4, 17, 4), dtype=np.float64)

for slot in range(4):
    step = torch.zeros_like(r275_pre)
    step[:, slot, :] = torch.as_tensor(
        r275_eps*r275_direction, dtype=torch.float32
    )
    plus_m, plus_p, _ = r267_full_pre(r275_pre+step, indices151)
    minus_m, minus_p, _ = r267_full_pre(r275_pre-step, indices151)
    plus_P = np.stack([np242(p) for p in plus_p], axis=1)
    minus_P = np.stack([np242(p) for p in minus_p], axis=1)
    r275_Jslot_mean[:, :, slot] = (
        np242(plus_m)-np242(minus_m)
    )/(2*r275_eps)
    r275_Jslot_log[:, :, :, slot] = (
        np.log(np.maximum(plus_P, 1e-12)) -
        np.log(np.maximum(minus_P, 1e-12))
    )/(2*r275_eps)

r275_E = np242(r274_case["E"]).astype(np.float64)
r275_Jfeature_log = np.einsum(
    "brks,bsi->brki", r275_Jslot_log, r275_E
)
r275_Jfeature_mean = np.einsum(
    "brs,bsi->bri", r275_Jslot_mean, r275_E
)
r275_W = r275_P*r274_weights[None, :, None]/len(indices151)
r275_design = (
    r275_Jfeature_log*np.sqrt(r275_W)[:, :, :, None]
).reshape(-1, 192)
r275_F = r275_design.T @ r275_design
r275_Af = r275_Jfeature_mean.mean(axis=0)

print("Feature-aware Fisher diagonal mean:", float(np.trace(r275_F)/192))
print("Old feature's quadratic KL estimate:",
      float(0.5*r274_v @ r275_F @ r274_v))
print("Slot-aware mean derivative times old feature:",
      (r275_Af @ r274_v).round(6).tolist())
print("Submitted weights remain:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 275: feature-aware factual distortion\n"
        "- Direction-only Fisher routes had similar held-out KL; half strength cost approximately 0.0009 categorical nats locally.\n"
        "- Computed scalar-slot finite differences and the induced probability Fisher matrix for a linear dose feature.\n"
        "- Recovered record-based target metadata and a second factual-validation batch.\n"
        "- No backpropagation, weight optimizer, or weight changes.\n"
        "- Python runs: 275; forward passes: 24.\n"
    )
```
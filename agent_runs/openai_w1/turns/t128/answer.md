The older helper comparison only reproduced known unit offsets; it did not explain the reading mismatch. I’ll align the fresh and validated arrays using the exported posterior vectors, then compare context, observation, and action inputs separately. This should isolate the faulty component without another parameter sweep.

```python
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

rr117_dist, rr117_match = cKDTree(
    r79_target.detach().cpu().numpy().astype(np.float64)
).query(rr_zexport4.detach().cpu().numpy().astype(np.float64))
rr117_exact = rr117_dist < 2e-5
print("Exact export/cache matches:", int(rr117_exact.sum()), "/", len(rr117_exact),
      "distance quantiles:", np.quantile(rr117_dist, [0, .5, .9, 1]))

rr117_e = rr_encode_atoms112(rr_atoms4, rr_obs_mask112, "encode", False)
rr117_u = rr_encode_atoms112(rr_atoms4, rr_act_mask112, "encode_action", True)
rr117_refidx = torch.as_tensor(rr117_match, dtype=torch.long)

def rr117_compare(label, a, b):
    d = (torch.as_tensor(a) - torch.as_tensor(b)[rr117_refidx]).float()
    err = d.reshape(len(d), -1).square().mean(1).sqrt().numpy()
    for subset, mask in (
        ("reading", rr117_exact & np.asarray(rr_has_reading4, dtype=bool)),
        ("active drug", rr117_exact & np.asarray(rr_has_active4, dtype=bool)),
        ("other", rr117_exact & ~np.asarray(rr_has_reading4, dtype=bool)
         & ~np.asarray(rr_has_active4, dtype=bool))
    ):
        if mask.any():
            print(label, subset, np.quantile(err[mask], [0, .5, .9, 1]))

print("\n=== ALIGNED COMPONENT DIFFERENCES ===")
rr117_compare("context", rr_best_ctx111, r88_ctx_full)
rr117_compare("observation", rr117_e, r88_e_full)
rr117_compare("action", rr117_u, r95_u)
rr117_compare("old reconstructed posterior", rr_zexport4, r88_z_full)

print("\n=== ALIGNED READING-ATOM DIFFERENCES ===")
rr117_diffs = []
for j in np.flatnonzero(rr117_exact & np.asarray(rr_has_reading4, dtype=bool))[:80]:
    old_j = int(rr117_match[j])
    old_atoms = r88_atoms_full[torch.as_tensor(r95_owners == old_j)]
    new_atoms = rr_atoms4[torch.as_tensor(rr_owner4 == j)]
    if len(old_atoms) != len(new_atoms) or not len(old_atoms):
        continue
    dist = torch.cdist(new_atoms, old_atoms).numpy()
    aa, bb = linear_sum_assignment(dist)
    rr117_diffs.extend((new_atoms[aa] - old_atoms[bb]).numpy())
if rr117_diffs:
    rr117_diffs = np.asarray(rr117_diffs)
    print("matched atoms", len(rr117_diffs), "norm quantiles",
          np.quantile(np.linalg.norm(rr117_diffs, axis=1), [0, .5, .9, 1]))
    print("difference singular values",
          np.linalg.svd(rr117_diffs, compute_uv=False)[:12])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 117\nComparing fresh and validated components only after alignment by exported "
            "posterior vectors. This isolates reconstruction errors from cache-order errors. "
            "No edits. Counts: 117 Python runs, 6 forward passes.\n")
```

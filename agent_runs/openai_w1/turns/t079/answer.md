The inspection stopped on a pandas indexing error, so it added no validation evidence. I’ll fix that and compare atom counts, independently encoded observations, and posterior means directly. Close agreement with the exported posterior would confirm the reconstruction; a substantial mismatch would mean the latent correction is still unsafe to use.

```python
import torch
import torch.nn.functional as F

def r71_observations(bundle_rows, glucose_delta=0.0, map_delta=0.0):
    atoms, owners = [], []
    for j, rows in enumerate(bundle_rows):
        for row in rows:
            if row.event in ("insulin", "norepinephrine"):
                continue
            value = float(row.value)
            if row.event in ("glucose_finger", "glucose_lab"):
                value += glucose_delta
            elif row.event in ("map_arterial", "map_cuff"):
                value += map_delta
            atoms.append(r36_atom(row.event, value=value))
            owners.append(j)
    owner = torch.tensor(owners, dtype=torch.long)
    vectors = torch.stack(atoms)
    pooled = torch.zeros(len(bundle_rows), 192)
    pooled.index_add_(0, owner, r33_mlp(vectors, "encode.mlp1", "gelu"))
    count = torch.bincount(owner, minlength=len(bundle_rows)).float()
    pooled = pooled / count.clamp_min(1).sqrt()[:, None]
    pooled += r33_linear(torch.log1p(count)[:, None], "encode.size")
    return r33_ln(r33_mlp(pooled, "encode.mlp2", "gelu"), "encode.norm")

r71_owner = r36_owners.numpy()
r71_action = r36_action_flags.bool()
print("=== EXACT ATOM COUNTS ===")
for j, i in enumerate(np.asarray(rr58_mi)[:4]):
    atom_ix = np.flatnonzero(r71_owner == i)
    world_ix = atom_ix[~r71_action[atom_ix].numpy()]
    identity = np.asarray(r33_meta)[int(r36_indices[i]), :2]
    print(int(i), identity, "cache all/world", len(atom_ix), len(world_ix),
          "reconstructed world", len(rr58_mrows[j]))

r71_e = r71_observations(rr58_mrows)
print("\n=== OBSERVATION ENCODER COMPARISON ===")
for label, other in (("old helper", rr58_e0),
                     ("cached", r36_e[np.asarray(rr58_mi)])):
    diff = r71_e - torch.as_tensor(other)
    print(label, "RMSE", float(diff.square().mean().sqrt()),
          "max", float(diff.abs().max()))

r71_e_all = r36_set_encode("encode", ~r71_action, False)
r71_u_all = r36_set_encode("encode_action", r71_action, True)
print("\n=== POSTERIOR RECONSTRUCTION ===")
for label, e in (("cached", r36_e), ("world-only", r71_e_all)):
    inp = torch.cat((r36_ctx, e, r71_u_all), dim=1)
    for act in ("gelu", "silu"):
        hidden = r33_act(r33_linear(inp, "z_post.0"), act)
        z = r33_linear(hidden, "z_post.2")[:, :32]
        err = (z - r36_latent_target).square().mean(1).sqrt()
        print(label, act, "mean/median/p95",
              np.round(np.quantile(err.numpy(), [0.5, 0.95]), 6),
              round(float(err.mean()), 6),
              "drug mean", round(float(err[r36_has_start.any(1)].mean()), 6))
print("target latent RMS", float(r36_latent_target.square().mean().sqrt()))
print("CURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 70-71\n"
            "Run 70 stopped on DataFrame column/row indexing; no validation claim follows. "
            "Run 71 fixes indexing and checks independent observation encoding and posterior agreement. "
            "No weights changed. Counts: 71 Python runs, 6 forward passes.\n")
```
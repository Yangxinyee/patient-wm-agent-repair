The helpers expose the required reconstruction steps, although the diagnostic output was truncated. I’ll rebuild the bundle inputs and compare them with the cached features before using the new states. Matching counts and cached encodings, with small posterior error, would validate the reconstruction; mismatches would halt refitting.

```python
import torch
import pandas as pd
import numpy as np

torch.set_grad_enabled(False)
torch.set_num_threads(8)

def bundle_inputs188(records, meta):
    starts, ends = {}, {}
    for row in records.to_dict("records"):
        key = (int(row["stay_id"]), int(round(float(row["hour"]) * 60)))
        starts.setdefault(key, []).append(row)
        if row["event"] in ("insulin", "norepinephrine"):
            key_end = (int(row["stay_id"]),
                       int(round(float(row["end_hour"]) * 60)))
            ends.setdefault(key_end, []).append(row)
    obs_sets, action_sets, action_records = [], [], []
    for row in meta.itertuples(index=False):
        key = (int(row.stay_id), int(round(float(row.hour) * 60)))
        world = starts.get(key, [])
        actions = [r for r in world
                   if r["event"] in ("insulin", "norepinephrine")]
        obs = [embed_record_atom_v2(r, 0, "same") for r in world
               if r["event"] not in ("insulin", "norepinephrine")]
        obs += [embed_record_atom_v2(r, 2, "same")
                for r in ends.get(key, [])]
        obs_sets.append(obs)
        action_sets.append([embed_record_atom_v2(r, 0, "same")
                            for r in actions])
        action_records.append(actions)
    return obs_sets, action_sets, action_records

obs_old188, acts_old188, rows_old188 = bundle_inputs188(
    reference_records_repair, sm_repair)
counts_old188 = np.array([len(o) + len(a)
                         for o, a in zip(obs_old188, acts_old188)])
print("OLD bundle count mismatches:",
      np.sum(counts_old188 != sm_repair["n_atoms"].to_numpy()))
checks188 = np.unique(np.r_[posterior_rows_repair[:16],
                            posterior_rows_repair[::170]])[:48]
e_check188 = encode_variable_sets_repair(
    [obs_old188[i] for i in checks188], "encode")
u_check188 = encode_variable_sets_repair(
    [acts_old188[i] for i in checks188], "encode_action", zero_empty=True)
lookup188 = {int(r): j for j, r in enumerate(posterior_rows_repair)}
positions188 = np.array([lookup188[int(i)] for i in checks188])
print("OLD cached e/u RMSE:",
      float(torch.sqrt(torch.mean(
          (e_check188 - posterior_e_repair[positions188]) ** 2))),
      float(torch.sqrt(torch.mean(
          (u_check188 - posterior_u_repair[positions188]) ** 2))))

meta188 = pd.DataFrame({k: S187[k] for k in
                       ["stay_id", "hour", "has_insulin",
                        "has_norepinephrine", "n_atoms"]})
sx188 = torch.tensor(S187["x"], dtype=torch.float32)
records188 = train[train.stay_id.isin(meta188.stay_id.unique())].copy()
obs188, acts188, action_rows188 = bundle_inputs188(records188, meta188)
counts188 = np.array([len(o) + len(a) for o, a in zip(obs188, acts188)])
print("NEW bundle count mismatches:",
      np.sum(counts188 != meta188["n_atoms"].to_numpy()))
print("Count-difference distribution:",
      dict(zip(*np.unique(counts188 - meta188["n_atoms"].to_numpy(),
                          return_counts=True))))

rows188 = np.flatnonzero(
    np.r_[False, meta188.stay_id.to_numpy()[1:] ==
                 meta188.stay_id.to_numpy()[:-1]])
e188 = encode_variable_sets_repair([obs188[i] for i in rows188], "encode")
u188 = encode_variable_sets_repair(
    [acts188[i] for i in rows188], "encode_action", zero_empty=True)

old_meta188, old_sx188 = sm_repair, sx_repair
try:
    sm_repair, sx_repair = meta188, sx188
    ctx188, hf188 = previous_context_repair(rows188)
finally:
    sm_repair, sx_repair = old_meta188, old_sx188

print("Posterior weight names:",
      [n for n in Wrepair if n.startswith("z_post")])
z_fit188 = rmlp(torch.cat([ctx188, e188, u188], dim=1), "z_post")[:, :32]
z188 = sx188[rows188, 960:992]
print("NEW posterior RMSE / maximum:",
      float(torch.sqrt(torch.mean((z_fit188-z188)**2))),
      float(torch.max(torch.abs(z_fit188-z188))))
E188 = prelude_features111(hf188, e188, u188, z188).numpy()
print("NEW prelude shape:", E188.shape)
print("Action flag mismatches:",
      {drug: int(np.sum(
          np.array([any(r["event"] == drug for r in action_rows188[i])
                    for i in rows188]) !=
          meta188.iloc[rows188]["has_" + drug].to_numpy()))
       for drug in ["insulin", "norepinephrine"]})
print("Source-z agreement in OLD cached features:",
      float(torch.sqrt(torch.mean(
          (posterior_z_repair -
           sx_repair[posterior_rows_repair, 960:992]) ** 2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run188: independent feature reconstruction\n"
            f"New prelude shape {E188.shape}; old e/u and new posterior "
            "agreement printed for validation. Current edit unchanged.\n")
```
The learned empty measurement encoding reduces posterior error from about 0.135 to 0.00032. That strongly supports the missing-empty-encoder explanation. I’ll now resolve the remaining categorical and bin conventions for both drugs. Near-machine-precision agreement would establish a reliable action-input reconstruction; a persistent error would reveal another missing detail.

```python
import itertools, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

t0 = time.time()
print("numeric convention:", VOCAB["numeric_note"])
atom_recipe_repair = {}
action_atom_vectors_repair = {}
empty_e_repair = empty_encodings_repair["mlp2_size_bias"]

for drug in ["insulin", "norepinephrine"]:
    pick = np.flatnonzero(action_fixture_repair["event"].to_numpy() == drug)
    rec = action_fixture_repair.iloc[pick]
    cc = torch.as_tensor(action_ctx_repair[pick], dtype=torch.float32)
    yy = torch.as_tensor(posterior_target_repair[pick], dtype=torch.float32)
    nz = torch.zeros(len(pick), 6)
    for attr, col in [("dose", 2), ("rate", 4)]:
        spec = numeric_lookup_repair[(drug, attr)]
        nz[:, col] = torch.as_tensor(
            (rec[attr].to_numpy() - spec["median"]) / spec["iqr"],
            dtype=torch.float32
        )
        nz[:, col + 1] = 1
    base = (
        rW["embed.code.weight"][code_lookup_repair[drug]]
        + rW["embed.kind.weight"][0] + rW["embed.ekind.weight"][2]
        + rp_linear(nz, "embed.numeric")
    )
    candidates, av = [], []
    for side in ["left", "right"]:
        bins_term = torch.zeros(len(pick), 192)
        for attr in ["dose", "rate"]:
            spec = numeric_lookup_repair[(drug, attr)]
            bb = np.searchsorted(spec["fine_bin_edges"],
                                 rec[attr].to_numpy(), side=side) + 1
            bins_term += rW[f"embed.bins.{attr}.weight"][torch.as_tensor(bb)]
        for unit, runit, kq in itertools.product(range(6), range(4), [-1, 0, 1, 2, 3]):
            atom = (base + bins_term
                    + rW["embed.categorical.unit.weight"][unit]
                    + rW["embed.categorical.rate_unit.weight"][runit])
            if kq >= 0:
                atom = atom + rW["embed.kq.weight"][kq]
            candidates.append((unit, runit, kq, side))
            av.append(atom)
    av = torch.stack(av)
    hidden = rp_mlp(av, "encode_action.mlp1")
    size = rp_linear(torch.full((1, 1, 1), np.log(2), dtype=torch.float32),
                     "encode_action.size")
    uv = rp_norm(rp_mlp(hidden + size, "encode_action.mlp2"),
                 "encode_action.norm")
    inp = torch.cat([
        cc.unsqueeze(0).expand_as(uv),
        empty_e_repair.reshape(1, 1, 192).expand_as(uv),
        uv
    ], dim=-1)
    zz = rp_mlp(inp, "z_post")[..., :32]
    errs = (zz - yy.unsqueeze(0)).square().mean(dim=(1, 2)).sqrt().numpy()
    order = np.argsort(errs)
    print("\n", drug, "best candidates:")
    for j in order[:5]:
        print(candidates[j], "posterior RMSE", float(errs[j]))
    best = int(order[0])
    atom_recipe_repair[drug] = candidates[best]
    action_atom_vectors_repair[drug] = av[best].clone()

print("selected recipes:", atom_recipe_repair)
print("elapsed seconds:", round(time.time() - t0, 2))
note("Run 39: empty measurement encoder resolved the major posterior mismatch. Searched exact drug categorical and bin conventions; retained best atom recipes. No weights edited.")
```

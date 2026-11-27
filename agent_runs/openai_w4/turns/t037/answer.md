The last run exposed the needed tensor names but stopped on missing unit columns in the retained dataframe. No weights changed. I’ll infer the embedding conventions by matching isolated drug bundles to exported posterior means. A near-exact match would validate this calculation before any edit.

```python
import itertools, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
torch.set_grad_enabled(False)
rW = {name: torch.as_tensor(np.asarray(load_original(name)).copy(), dtype=torch.float32)
      for name in NAMES}

def rp_linear(x, prefix):
    return F.linear(x, rW[prefix + ".weight"], rW.get(prefix + ".bias"))

def rp_norm(x, prefix):
    return F.layer_norm(x, (x.shape[-1],),
                        rW[prefix + ".weight"], rW[prefix + ".bias"], eps=1e-5)

def rp_mlp(x, prefix):
    return rp_linear(F.gelu(rp_linear(x, prefix + ".0")), prefix + ".2")

def rp_action_encoder(atoms):
    hidden = rp_mlp(atoms, "encode_action.mlp1")
    hidden = hidden + rp_linear(torch.full((len(atoms), 1), np.log(2),
                                          dtype=torch.float32), "encode_action.size")
    return rp_norm(rp_mlp(hidden, "encode_action.mlp2"), "encode_action.norm")

fixture = action_fixture_repair.copy()
si = fixture["state_index"].to_numpy(dtype=int)
assert np.all(si > 0)
assert np.all(state_train_repair["stay_id"][si] ==
              state_train_repair["stay_id"][si - 1])
px = torch.as_tensor(state_train_repair["x"][si - 1].copy(), dtype=torch.float32)
dt = torch.as_tensor(
    (state_train_repair["hour"][si] - state_train_repair["hour"][si - 1]) * 3600,
    dtype=torch.float32)
flow = px[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -dt[:, None, None] * torch.exp(rW["core.flow.a"])[None, None, :])
action_ctx_repair = (px[:, :192] + rp_norm(flow.mean(1), "core.out_norm")
                     + rp_mlp(torch.log1p(dt[:, None]) / 10, "gap"))
posterior_target_repair = torch.as_tensor(
    state_train_repair["x"][si, 960:992].copy(), dtype=torch.float32)

numeric_specs_repair = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
code_ids_repair = {x["name"]: x["id"] for x in VOCAB["codes"]}
search_rows = []
started = time.time()

for drug in ["insulin", "norepinephrine"]:
    ix = np.flatnonzero(fixture["event"].eq(drug).to_numpy())
    if not len(ix):
        continue
    ctx = action_ctx_repair[ix]
    target = posterior_target_repair[ix]
    candidates = list(itertools.product(range(6), range(4), [-1, 0, 1, 2, 3],
                                        ["left", "right"]))
    bases = {}
    for side in ["left", "right"]:
        numbers = np.zeros((len(ix), 6), dtype=np.float32)
        bins = {}
        for attribute, zcol, flagcol in [("dose", 2, 3), ("rate", 4, 5)]:
            spec = numeric_specs_repair[(drug, attribute)]
            vals = fixture.iloc[ix][attribute].to_numpy()
            numbers[:, zcol] = np.clip((vals - spec["median"]) / spec["iqr"], -10, 10)
            numbers[:, flagcol] = 1
            bins[attribute] = torch.as_tensor(
                1 + np.searchsorted(spec["fine_bin_edges"], vals, side=side),
                dtype=torch.long)
        bases[side] = (
            rW["embed.code.weight"][code_ids_repair[drug]]
            + rW["embed.kind.weight"][0] + rW["embed.ekind.weight"][2]
            + rW["embed.bins.value.weight"][0]
            + rW["embed.bins.dose.weight"][bins["dose"]]
            + rW["embed.bins.rate.weight"][bins["rate"]]
            + rp_linear(torch.as_tensor(numbers), "embed.numeric")
        )
    for start in range(0, len(candidates), 24):
        chunk = candidates[start:start+24]
        atoms = torch.stack([
            bases[side] + rW["embed.categorical.unit.weight"][unit]
            + rW["embed.categorical.rate_unit.weight"][rate_unit]
            + (rW["embed.kq.weight"][kq] if kq >= 0 else 0)
            for unit, rate_unit, kq, side in chunk
        ])
        u = rp_action_encoder(atoms.reshape(-1, 192)).reshape(len(chunk), len(ix), 192)
        contexts = ctx[None].expand(len(chunk), -1, -1)
        post_input = torch.cat([contexts, torch.zeros_like(u), u], dim=-1)
        prediction = rp_mlp(post_input.reshape(-1, 576), "z_post")[:, :32]
        prediction = prediction.reshape(len(chunk), len(ix), 32)
        errors = torch.sqrt(((prediction - target[None]) ** 2).mean((1, 2))).numpy()
        for params, error in zip(chunk, errors):
            search_rows.append((drug, *params, float(error)))

posterior_search_repair = pd.DataFrame(
    search_rows, columns=["drug", "unit", "rate_unit", "kq", "bin_side", "z_RMSE"])
print("fixture counts:", fixture["event"].value_counts().to_dict())
print("posterior target RMS:", float(torch.sqrt((posterior_target_repair**2).mean())))
for drug, table in posterior_search_repair.groupby("drug"):
    print("\n", drug)
    print(table.nsmallest(6, "z_RMSE").to_string(index=False))
print("seconds:", round(time.time() - started, 2))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 30–31\n"
            "Run 30 stopped on absent rate_unit/dose_unit columns; no edits. "
            "Reconstructed isolated action embeddings, action encoder and "
            "z_post using GELU, testing unit/rate-unit/KQ/bin conventions "
            "against 133 exported posterior means. Best errors: "
            + str(posterior_search_repair.groupby("drug")["z_RMSE"].min().to_dict())
            + ". No weights edited.\n")
```
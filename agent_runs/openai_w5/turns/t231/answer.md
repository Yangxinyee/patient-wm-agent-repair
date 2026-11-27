The existing forward results confirm correct directions but substantial doubling-versus-halving asymmetry. I’ll first test whether the action encoder itself carries rates proportionally. If it does, the next suspects are attention routing and the dose-centering term; if it does not, increasing the head strength would be premature.

```python
import numpy as np, torch
import torch.nn.functional as F

code_id = {r["name"]: r["id"] for r in V["codes"]}
num_spec = {(r["code"], r["attribute"]): r for r in V["numeric"]}

def linear205(x, p, weights=W):
    return F.linear(x, weights[p + ".weight"], weights.get(p + ".bias"))

def norm205(x, p, weights=W):
    return F.layer_norm(x, (192,), weights[p + ".weight"],
                        weights[p + ".bias"], 1e-5)

def drug_atom205(drug, rate, weights=W):
    rate = np.asarray(rate, dtype=np.float32).reshape(-1)
    dose = rate if drug == "insulin" else rate * 4.8
    n = len(rate)
    x = (weights["embed.code.weight"][code_id[drug]]
         + weights["embed.kind.weight"][0]
         + weights["embed.ekind.weight"][2]
         + weights["embed.kq.weight"][0]
         + weights["embed.text.weight"][0])
    for name in ["action", "discharged_to", "route", "status"]:
        x = x + weights["embed.categorical." + name + ".weight"][0]
    x = x + weights["embed.categorical.unit.weight"][3 if drug == "insulin" else 4]
    x = x + weights["embed.categorical.rate_unit.weight"][2 if drug == "insulin" else 3]
    x = x.reshape(1, -1).expand(n, -1).clone()
    nums = torch.zeros((n, 6), dtype=torch.float32)
    x += weights["embed.bins.value.weight"][0]
    for attr, vals, zidx, flagidx in [("dose", dose, 2, 3), ("rate", rate, 4, 5)]:
        sp = num_spec[(drug, attr)]
        bins = 1 + np.searchsorted(np.asarray(sp["fine_bin_edges"]), vals, side="left")
        x += weights["embed.bins." + attr + ".weight"][torch.as_tensor(bins)]
        nums[:, zidx] = torch.from_numpy(np.clip((vals-sp["median"])/sp["iqr"], -10, 10))
        nums[:, flagidx] = 1
    return x + linear205(nums, "embed.numeric", weights)

def set205(atom_sets, p="encode_action", weights=W):
    n = atom_sets.shape[1]
    if n:
        h = linear205(F.gelu(linear205(atom_sets, p+".mlp1.0", weights)),
                      p+".mlp1.2", weights).sum(1) / np.sqrt(n)
    else:
        h = torch.zeros((atom_sets.shape[0], 192), dtype=torch.float32)
    h += linear205(torch.full((atom_sets.shape[0], 1), np.log1p(n)), p+".size", weights)
    return norm205(linear205(F.gelu(linear205(h, p+".mlp2.0", weights)),
                             p+".mlp2.2", weights), p+".norm", weights)

print("=== FINITE ACTION CHANNELS ===")
rate_grid205 = np.array([0, .25, .5, 1, 2, 4, 8, 11, 16, 24], dtype=np.float32)
for drug in ["insulin", "norepinephrine"]:
    grid = rate_grid205 if drug == "insulin" else rate_grid205 / 8
    uu = set205(drug_atom205(drug, grid)[:, None, :])
    print(drug)
    print(pd.DataFrame({"rate": grid,
                        "I_pair": (uu[:,15]-uu[:,66]).numpy(),
                        "NE_pair": (uu[:,21]-uu[:,61]).numpy(),
                        "ordinary_RMS": uu[:, [i for i in range(192) if i not in priv]].square().mean(1).sqrt().numpy()
                       }).round(7).to_string(index=False))
print("Empty encoded action private:",
      set205(torch.empty((1, 0, 192)))[:, priv].tolist())

print("\n=== CORE AND HEAD TENSOR NAMES ===")
print([(n, tuple(W[n].shape)) for n in NAMES
       if n.startswith(("core.prelude.", "core.prelude_norm.", "core.b_in",
                        "core.c_out", "core.coda.", "z_prior.", "z_post.",
                        "gap.", "last_", "type_emb.", "embed_transition."))])
```
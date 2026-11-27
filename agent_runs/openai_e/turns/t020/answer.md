The available output is still the layer metadata, not a posterior reconstruction result. I’ll run a compact matching test now, without changing weights. A close match would validate the local action inputs; otherwise I’ll correct the reconstruction before attempting any repair.

```python
import numpy as np
import torch
import torch.nn.functional as F
import itertools

torch.set_num_threads(8)

def recon_lin(x, p):
    return F.linear(x, W0[p+".weight"], W0.get(p+".bias"))

def recon_mlp(x, p, act="gelu"):
    fn = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[act]
    return recon_lin(fn(recon_lin(x, p+".0")), p+".2")

def recon_norm(x, p):
    return F.layer_norm(x, (192,), W0[p+".weight"], W0[p+".bias"], 1e-5)

def recon_ctx(indices, act="gelu", gap_style="log1p"):
    indices = np.asarray(indices, dtype=int)
    previous = torch.as_tensor(S0["x"][indices-1], dtype=torch.float32)
    dt = torch.as_tensor(
        (S0["hour"][indices]-S0["hour"][indices-1])*3600,
        dtype=torch.float32
    ).reshape(-1, 1)
    hh = previous[:, 192:960].reshape(-1, 4, 192)
    hh *= torch.exp(-dt[:, :, None]*torch.exp(W0["core.flow.a"])[None, None, :])
    lg = torch.log1p(dt) if gap_style == "log1p" else torch.log(dt.clamp_min(1))
    return previous[:, :192] + recon_norm(hh.mean(1), "core.out_norm") + recon_mlp(lg/10, "gap", act)

drug_starts, drug_ends = {}, {}
for row in tune[tune.event.isin(DRUGS)].itertuples(index=False):
    drug_starts.setdefault((int(row.stay_id), int(round(row.hour*60))), []).append(row)
    drug_ends.setdefault((int(row.stay_id), int(round(row.end_hour*60))), []).append(row)

pure_action_rows = {}
for drug in DRUGS:
    indices, records = [], []
    for j in np.flatnonzero(S0["has_insulin"] | S0["has_norepinephrine"]):
        key = (int(S0["stay_id"][j]), int(round(S0["hour"][j]*60)))
        rows = drug_starts.get(key, [])
        if (j > 0 and S0["stay_id"][j-1] == S0["stay_id"][j]
            and len(rows) == 1 and not drug_ends.get(key)
            and int(S0["n_atoms"][j]) == 1 and rows[0].event == drug):
            indices.append(j)
            records.append(rows[0])
    pure_action_rows[drug] = (np.array(indices[:30], dtype=int), records[:30])
    print(drug, "eligible single-start bundles:", len(indices))

posterior_recon_search = []
choices = list(itertools.product(range(6), range(4)))
for drug, (indices, records) in pure_action_rows.items():
    if not len(indices):
        continue
    values = {"dose": np.array([r.dose for r in records]),
              "rate": np.array([r.rate for r in records])}
    numeric = np.zeros((len(records), 6), dtype=np.float32)
    for col, attr in ((2, "dose"), (4, "rate")):
        spec = NUM_SPEC[(drug, attr)]
        numeric[:, col] = np.clip((values[attr]-spec["median"])/spec["iqr"], -10, 10)
        numeric[:, col+1] = 1
    base_atom = (
        W0["embed.code.weight"][CODE_ID[drug]] +
        W0["embed.kind.weight"][0] + W0["embed.ekind.weight"][2] +
        F.linear(torch.from_numpy(numeric), W0["embed.numeric.weight"],
                 W0["embed.numeric.bias"])
    )
    target = torch.as_tensor(S0["x"][indices, 960:992])
    for side in ("left", "right"):
        atom = base_atom.clone()
        for attr in ("dose", "rate"):
            bins = np.searchsorted(NUM_SPEC[(drug, attr)]["fine_bin_edges"],
                                   values[attr], side=side)+1
            atom += W0["embed.bins."+attr+".weight"][bins]
        atoms = torch.stack([
            atom + W0["embed.categorical.unit.weight"][ui] +
            W0["embed.categorical.rate_unit.weight"][ri]
            for ui, ri in choices
        ])
        for act in ("gelu", "relu", "silu"):
            uu = recon_mlp(atoms, "encode_action.mlp1", act)
            uu += recon_lin(torch.full((*uu.shape[:-1], 1), np.log(2),
                                       dtype=torch.float32), "encode_action.size")
            uu = recon_norm(recon_mlp(uu, "encode_action.mlp2", act), "encode_action.norm")
            for gap_style in ("log1p", "log"):
                ctx = recon_ctx(indices, act, gap_style)
                inputs = torch.cat((ctx.expand(len(choices), -1, -1),
                                    torch.zeros_like(uu), uu), dim=-1)
                zp = recon_mlp(inputs, "z_post", act)[..., :32]
                errors = ((zp-target[None])**2).mean((1, 2)).sqrt()
                for ci, (ui, ri) in enumerate(choices):
                    posterior_recon_search.append({
                        "drug": drug, "rmse": float(errors[ci]),
                        "unit_id": ui, "rate_unit_id": ri,
                        "bin_side": side, "activation": act, "gap_style": gap_style
                    })

posterior_recon_best = {}
print("\n=== BEST MATCHES ===")
for drug in DRUGS:
    matches = sorted([r for r in posterior_recon_search if r["drug"] == drug],
                     key=lambda r: r["rmse"])
    if matches:
        posterior_recon_best[drug] = matches[0]
        for result in matches[:3]:
            print(result)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 17: single-start posterior matching\n"
        "Reconstructed action-only posterior inputs; searched bin boundaries, unit "
        "mappings, activations and gap transforms. Best candidates: "
        + repr(posterior_recon_best) +
        "\nNo weights edited. Python runs: 17. Forward requests: 3.\n"
    )
```
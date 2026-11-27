The new snapshots cover 100 training stays. I’ll reconstruct the action encoder and posterior on the first infusion bundles, where there are no earlier infusion endings to complicate the input. Near-zero agreement with the exported posterior means would validate this pathway; a mismatch would make an edit based on it premature.

```python
import itertools
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
torch.set_grad_enabled(False)

weights37 = {}
def wc37(name):
    if name not in weights37:
        weights37[name] = torch.as_tensor(load_original(name)).float()
    return weights37[name]

def linear37(x, prefix):
    bias = wc37(prefix + ".bias") if prefix + ".bias" in NAMES else None
    return F.linear(x, wc37(prefix + ".weight"), bias)

def activation37(x, name="gelu"):
    return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[name](x)

def mlp37(x, prefix, activation="gelu"):
    return linear37(activation37(linear37(x, prefix + ".0"), activation),
                    prefix + ".2")

def norm37(x, prefix):
    return F.layer_norm(x, (192,), wc37(prefix + ".weight"),
                        wc37(prefix + ".bias"), 1e-5)

state_train37 = dict(np.load("/work/fwd_4/states_original.npz"))
numeric37 = {(r["code"], r["attribute"]): r for r in VOCAB["numeric"]}
code_ids37 = {r["name"]: r["id"] for r in VOCAB["codes"]}

def contexts37(states, indices):
    indices = np.asarray(indices)
    prev = indices - 1
    assert np.all(states["stay_id"][prev] == states["stay_id"][indices])
    dt = torch.tensor((states["hour"][indices] - states["hour"][prev]) * 3600,
                      dtype=torch.float32)
    x = torch.tensor(states["x"][prev])
    h = x[:, 192:960].reshape(-1, 4, 192)
    flowed = h * torch.exp(-dt[:, None, None] *
                          torch.exp(wc37("core.flow.a"))[None, None, :])
    return (x[:, :192] + norm37(flowed.mean(1), "core.out_norm")
            + mlp37(torch.log1p(dt)[:, None] / 10, "gap"))

raw_lookup37 = train.set_index(["stay_id", "hour", "event"])
action_candidates37 = {}
for drug in ("insulin", "norepinephrine"):
    own = "has_insulin" if drug == "insulin" else "has_norepinephrine"
    other = "has_norepinephrine" if drug == "insulin" else "has_insulin"
    eligible = np.flatnonzero(
        state_train37[own] & ~state_train37[other]
        & (state_train37["n_atoms"] == 1)
        & np.isclose(state_train37["hour"], 0.0833)
    )[:12]
    if not len(eligible):
        raise RuntimeError("No uncomplicated first infusion bundles for " + drug)

    rows = raw_lookup37.loc[
        [(int(state_train37["stay_id"][i]),
          float(state_train37["hour"][i]), drug) for i in eligible]
    ]
    ctx = contexts37(state_train37, eligible)
    target = torch.tensor(state_train37["x"][eligible, 960:992])
    ns = len(eligible)

    records = []
    atoms = []
    for side, unit, rate_unit, action, route, status in itertools.product(
            ("left", "right"), range(6), range(4), range(2), range(2), range(2)):
        nums = torch.zeros(ns, 6)
        for attr, zcol, flagcol in (("dose", 2, 3), ("rate", 4, 5)):
            spec = numeric37[(drug, attr)]
            values = rows[attr].to_numpy(float)
            nums[:, zcol] = torch.tensor(
                (values - spec["median"]) / spec["iqr"], dtype=torch.float32)
            nums[:, flagcol] = 1
        atom = (wc37("embed.code.weight")[code_ids37[drug]]
                + wc37("embed.kind.weight")[0]
                + wc37("embed.ekind.weight")[2]
                + linear37(nums, "embed.numeric"))
        for attr in ("dose", "rate"):
            bins = np.searchsorted(numeric37[(drug, attr)]["fine_bin_edges"],
                                   rows[attr].to_numpy(float), side=side) + 1
            atom = atom + wc37("embed.bins." + attr + ".weight")[bins]
        for field, ix in (("unit", unit), ("rate_unit", rate_unit),
                          ("action", action), ("route", route), ("status", status)):
            atom = atom + wc37("embed.categorical." + field + ".weight")[ix]
        records.append((side, unit, rate_unit, action, route, status))
        atoms.append(atom)

    atoms = torch.stack(atoms)
    ctx_batch = ctx.expand(len(records), -1, -1)
    e_batch = torch.zeros_like(ctx_batch)
    ranked = []
    for act in ("gelu", "silu", "relu"):
        mid = mlp37(atoms, "encode_action.mlp1", act)
        sized = mid + linear37(torch.full((len(records), ns, 1),
                                         np.log(2), dtype=torch.float32),
                               "encode_action.size")
        u = norm37(mlp37(sized, "encode_action.mlp2", act), "encode_action.norm")
        post_input = torch.cat((ctx_batch, e_batch, u), dim=-1)
        for post_act in ("gelu", "silu", "relu"):
            mean = mlp37(post_input, "z_post", post_act)[..., :32]
            errors = ((mean - target) ** 2).mean((1, 2)).sqrt().numpy()
            for j in np.argsort(errors)[:3]:
                ranked.append((float(errors[j]), act, post_act, records[j]))
    action_candidates37[drug] = sorted(ranked)
    print(drug, "n =", ns, "rates =", rows.rate.to_list())
    print("best posterior reconstructions:", action_candidates37[drug][:6])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 37\n"
            "Forward 4 exported 4491 states for 100 random training stays. "
            "Validating action encoder and posterior exactly on first, single-drug "
            "infusions by categorical/bin/activation enumeration. Candidate results "
            "retained in action_candidates37; utilities wc37, contexts37, mlp37. "
            "No edits.\n")
```
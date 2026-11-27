The lookup failed before any reconstruction was tested, so it tells us nothing about the model. I’ll match event times using canonical integer keys and rerun the check. Exact posterior agreement remains the criterion for using this action pathway in an edit.

```python
import itertools
import numpy as np
import pandas as pd
import torch

print("Original time dtype:", train["hour"].dtype)
inputs38 = train.loc[train.stay_id.isin(state_train_ids36),
                     ["stay_id", "hour", "event", "rate", "dose"]].copy()
inputs38["hour_key"] = np.rint(
    pd.to_numeric(inputs38["hour"]).to_numpy(float) * 10000).astype(np.int64)
raw_lookup38 = inputs38.set_index(["stay_id", "hour_key", "event"])
action_candidates37 = {}

for drug in ("insulin", "norepinephrine"):
    own = "has_insulin" if drug == "insulin" else "has_norepinephrine"
    other = "has_norepinephrine" if drug == "insulin" else "has_insulin"
    eligible = np.flatnonzero(
        state_train37[own] & ~state_train37[other]
        & (state_train37["n_atoms"] == 1)
        & np.isclose(state_train37["hour"], 0.0833)
    )[:12]
    keys = pd.MultiIndex.from_tuples(
        [(int(state_train37["stay_id"][i]),
          int(round(state_train37["hour"][i] * 10000)), drug) for i in eligible],
        names=raw_lookup38.index.names)
    rows = raw_lookup38.reindex(keys)
    assert rows.rate.notna().all(), (
        drug, keys.tolist(), raw_lookup38.head().to_string())
    ctx = contexts37(state_train37, eligible)
    target = torch.tensor(state_train37["x"][eligible, 960:992])
    ns = len(eligible)
    assert ns > 0

    records, atoms = [], []
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
    f.write("\n## Runs 37–38\n"
            "Run 37 stopped at a raw-event MultiIndex lookup before testing "
            "the encoder. Run 38 uses canonical integer time keys and retries. "
            "Candidate results in action_candidates37. No edits; forward 4/50.\n")
```
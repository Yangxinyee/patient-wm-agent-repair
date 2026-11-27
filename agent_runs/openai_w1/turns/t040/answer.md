Infusions occur five minutes after the readings, and later drug bundles include END atoms. The exported state immediately before an infusion is therefore useful for estimating its expected rate. I’ll test reconstruction of the latent posterior on these bundles; close agreement would validate the inputs needed for a centered-dose edit.

```python
import itertools
import torch
import torch.nn.functional as F
import pandas as pd

torch.set_num_threads(8)

# Combine the two original state exports without duplicating stay-hours.
r33_files = sorted(Path("/work").glob("fwd_*/states_original.npz"))
r33_parts = []
for r33_path in r33_files:
    with np.load(r33_path) as r33_z:
        r33_parts.append({k: r33_z[k].copy() for k in r33_z.files})
r33_raw = {k: np.concatenate([p[k] for p in r33_parts])
           for k in r33_parts[0]}
r33_meta = pd.DataFrame({
    "stay_id": r33_raw["stay_id"],
    "hour": np.round(r33_raw["hour"].astype(float), 4),
    "raw_index": np.arange(len(r33_raw["stay_id"]))
}).sort_values(["stay_id", "hour"]).drop_duplicates(["stay_id", "hour"])
r33_take = r33_meta.raw_index.to_numpy()
r33_states = {k: v[r33_take] for k, v in r33_raw.items()}
r33_states["hour"] = r33_meta.hour.to_numpy()
r33_ids = r33_meta.stay_id.to_numpy()
r33_hours = r33_meta.hour.to_numpy()
r33_records = train[train.stay_id.isin(np.unique(r33_ids))]
r33_starts, r33_ends = {}, {}
r33_nonactions = set()
for r33_row in r33_records.itertuples(index=False):
    r33_key = (int(r33_row.stay_id), round(float(r33_row.hour), 4))
    if r33_row.event in drugs:
        r33_atom = (r33_row.event, float(r33_row.rate), float(r33_row.dose))
        r33_starts.setdefault(r33_key, []).append(r33_atom)
        r33_ends.setdefault(
            (int(r33_row.stay_id), round(float(r33_row.end_hour), 4)), []
        ).append(r33_atom)
    else:
        r33_nonactions.add(r33_key)

r33_pairs = []
for r33_i in range(1, len(r33_ids)):
    r33_key = (int(r33_ids[r33_i]), round(float(r33_hours[r33_i]), 4))
    r33_atoms = (r33_starts.get(r33_key, []) +
                 r33_ends.get(r33_key, []))
    if (r33_ids[r33_i] == r33_ids[r33_i-1]
        and 0 < r33_hours[r33_i] < 24
        and r33_key not in r33_nonactions
        and len(r33_atoms) > 0
        and len(r33_atoms) == int(r33_states["n_atoms"][r33_i])):
        r33_pairs.append(r33_i)
r33_pairs = np.asarray(r33_pairs)
r33_rng = np.random.default_rng(330)
r33_sample = r33_rng.choice(r33_pairs, min(160, len(r33_pairs)), replace=False)
r33_sample.sort()

print("Exports:", [str(p) for p in r33_files])
print("Combined stays:", len(np.unique(r33_ids)),
      "reconstructable pure-action bundles:", len(r33_pairs))
print("Categorical row norms:")
for r33_name in NAMES:
    if r33_name.startswith("embed.categorical."):
        print(r33_name, np.round(np.linalg.norm(load_original(r33_name), axis=1), 3))
print("Gap tensors:", [(n, SHAPES[n]) for n in NAMES if n.startswith("gap.")])

r33_weights = {n: torch.as_tensor(load_original(n)) for n in NAMES}
r33_numeric = {(d["code"], d["attribute"]): d for d in VOCAB["numeric"]}
r33_codes = {d["name"]: d["id"] for d in VOCAB["codes"]}

def r33_linear(x, prefix):
    return F.linear(x, r33_weights[prefix + ".weight"],
                    r33_weights.get(prefix + ".bias"))

def r33_act(x, kind):
    return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[kind](x)

def r33_mlp(x, prefix, kind):
    return r33_linear(r33_act(r33_linear(x, prefix + ".0"), kind),
                      prefix + ".2")

def r33_ln(x, prefix):
    return F.layer_norm(x, (192,), r33_weights[prefix + ".weight"],
                        r33_weights[prefix + ".bias"], 1e-5)

def r33_action_embedding(end_mode, flags, use_unit, side):
    vectors, owners = [], []
    for j, i in enumerate(r33_sample):
        key = (int(r33_ids[i]), round(float(r33_hours[i]), 4))
        atoms = [(a, False) for a in r33_starts.get(key, [])]
        atoms += [(a, True) for a in r33_ends.get(key, [])]
        for (code, rate, dose), is_end in atoms:
            present = not is_end or end_mode == "copy"
            numeric = np.zeros(6, np.float32)
            b_rate = b_dose = 0
            if present:
                for value, attr, col in [(dose, "dose", 2), (rate, "rate", 4)]:
                    meta = r33_numeric[(code, attr)]
                    numeric[col] = np.clip(
                        (value-meta["median"])/meta["iqr"], -10, 10)
                    numeric[col+1] = 1
                    b = 1 + np.searchsorted(meta["fine_bin_edges"], value, side=side)
                    if attr == "dose":
                        b_dose = int(b)
                    else:
                        b_rate = int(b)
            v = (r33_weights["embed.code.weight"][r33_codes[code]]
                 + r33_weights["embed.kind.weight"][2 if is_end else 0]
                 + r33_weights["embed.ekind.weight"][2]
                 + r33_weights["embed.bins.value.weight"][0]
                 + r33_weights["embed.bins.rate.weight"][b_rate]
                 + r33_weights["embed.bins.dose.weight"][b_dose]
                 + r33_linear(torch.as_tensor(numeric), "embed.numeric"))
            for cat in ["status", "unit", "route", "action", "rate_unit",
                        "discharged_to"]:
                index = 0
                if present:
                    if cat == "unit" and use_unit:
                        index = 4 if code == "insulin" else 5
                    elif cat == "rate_unit":
                        index = 2 if code == "insulin" else 3
                    elif cat in flags:
                        index = 1
                v = v + r33_weights[f"embed.categorical.{cat}.weight"][index]
            for table in ["embed.kq.weight", "embed.text.weight"]:
                if table in r33_weights:
                    v = v + r33_weights[table][0]
            vectors.append(v)
            owners.append(j)
    return torch.stack(vectors), torch.tensor(owners)

r33_previous = torch.as_tensor(r33_states["x"][r33_sample-1])
r33_target = torch.as_tensor(r33_states["x"][r33_sample, -32:])
r33_dt = torch.tensor(
    3600*(r33_hours[r33_sample]-r33_hours[r33_sample-1]),
    dtype=torch.float32)
r33_flow = r33_previous[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -r33_dt[:, None, None] * torch.exp(r33_weights["core.flow.a"])[None, None, :])
r33_flow_out = r33_ln(r33_flow.mean(1), "core.out_norm")

# Cache candidate contexts and their contributions to the posterior hidden layer.
r33_contexts = {}
for r33_ga, r33_gscale, r33_base in itertools.product(
        ["silu", "gelu", "relu"], [.1, 1.0],
        ["out_plus_flow", "flow_only", "out_only"]):
    r33_gap = r33_mlp((r33_dt.clamp_min(1).log()*r33_gscale)[:, None],
                     "gap", r33_ga)
    r33_ctx = {
        "out_plus_flow": r33_previous[:, :192] + r33_flow_out,
        "flow_only": r33_flow_out,
        "out_only": r33_previous[:, :192]
    }[r33_base] + r33_gap
    r33_contexts[(r33_ga, r33_gscale, r33_base)] = F.linear(
        r33_ctx, r33_weights["z_post.0.weight"][:, :192],
        r33_weights["z_post.0.bias"])

r33_results = []
r33_counts = torch.tensor(
    [int(r33_states["n_atoms"][i]) for i in r33_sample],
    dtype=torch.float32)
for r33_end, r33_flags, r33_unit, r33_side in itertools.product(
        ["none", "copy"], [(), ("route", "action"),
                           ("status", "route", "action")],
        [False, True], ["left", "right"]):
    r33_emb, r33_owner = r33_action_embedding(
        r33_end, r33_flags, r33_unit, r33_side)
    for r33_ea in ["silu", "gelu", "relu"]:
        r33_atom_hidden = r33_mlp(r33_emb, "encode_action.mlp1", r33_ea)
        r33_sum = torch.zeros(len(r33_sample), 192)
        r33_sum.index_add_(0, r33_owner, r33_atom_hidden)
        r33_u = r33_ln(r33_mlp(
            r33_sum / r33_counts.sqrt()[:, None]
            + r33_linear(r33_counts.log1p()[:, None], "encode_action.size"),
            "encode_action.mlp2", r33_ea), "encode_action.norm")
        r33_u_term = F.linear(r33_u, r33_weights["z_post.0.weight"][:, 384:])
        for r33_ck, r33_c_term in r33_contexts.items():
            for r33_za in ["silu", "gelu", "relu"]:
                r33_hidden = r33_act(r33_c_term + r33_u_term, r33_za)
                r33_pred = r33_linear(r33_hidden, "z_post.2")[:, :32]
                r33_rmse = float((r33_pred-r33_target).square().mean().sqrt())
                r33_results.append((
                    r33_rmse, r33_end, r33_flags, r33_unit, r33_side,
                    r33_ea, *r33_ck, r33_za))
r33_results.sort(key=lambda x: x[0])
print("Posterior target RMS:", float(r33_target.square().mean().sqrt()))
print("Best reconstruction candidates:")
for r33_result in r33_results[:12]:
    print(r33_result)

with open("/work/notes.md", "a") as r33_file:
    r33_file.write(
        "\n## Runs 31–33\n"
        "Inspected normalized dose units and pure-action timing. Infusions follow "
        "hourly readings by five minutes; subsequent drug bundles include END "
        "atoms. Testing posterior reconstruction against exported original "
        "latent means before any posterior-output edit.\n"
        f"Best posterior reconstruction candidates: {r33_results[:3]}\n"
        "Counts: 33 Python runs, 6 forward passes. No edits.\n")
```

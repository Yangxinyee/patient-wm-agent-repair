The original model’s factual loss on these training stays is 1.63805 nats per reading. That is a reference for checking an edit, not evidence of correct drug effects. I’ll reconstruct the value head from the saved states and compare its means and losses with the forward tool. Close agreement would support using that reconstruction for statistical edits.

```python
import torch.nn.functional as F
import itertools

state_pack = np.load("/work/fwd_3/states_original.npz")
print("=== STATE ARRAYS ===")
for key in state_pack.files:
    arr = state_pack[key]
    print(key, arr.shape, arr.dtype, repr(arr[:3])[:240])
print("Event/atom kinds:", VOCAB["event_kinds"], VOCAB["atom_kinds"])
print("Finger centres:", VOCAB["value_bin_centres"]["glucose_finger"])
print("Last-numeric column norms:",
      torch.as_tensor(load_original("last_numeric.weight")).norm(dim=0).numpy().round(3))

head_reference = pd.read_csv("/work/fwd_4/factual_original.csv")
state_x = np.asarray(state_pack["x"])
state_ids = np.asarray(state_pack["stay_id"])
state_hours = np.asarray(state_pack["hour"])
head_rows = head_reference.copy()
head_rows["previous_state"] = -1
for sid, indices in head_rows.groupby("stay_id").groups.items():
    si = np.flatnonzero(state_ids == sid)
    order = np.argsort(state_hours[si])
    si = si[order]
    times = head_rows.loc[indices, "hour"].to_numpy()
    previous = np.searchsorted(state_hours[si], times - 1e-7, side="left") - 1
    valid = previous >= 0
    head_rows.loc[np.asarray(indices)[valid], "previous_state"] = si[previous[valid]]
head_rows = head_rows[head_rows.previous_state >= 0].reset_index(drop=True)

numeric_specs = {(s["code"], s["attribute"]): s for s in VOCAB["numeric"]}
code_ids = {c["name"]: c["id"] for c in VOCAB["codes"]}
def fine_bin(event, attribute, values):
    edges = numeric_specs[(event, attribute)]["fine_bin_edges"]
    return 1 + np.searchsorted(edges, values, side="left")

head_rows["last_value"] = np.nan
head_rows["last_hour"] = np.nan
for (sid, event), indices in head_rows.groupby(["stay_id", "event"]).groups.items():
    records = train[(train.stay_id == sid) & (train.event == event)].sort_values("hour")
    times = records.hour.to_numpy()
    vals = records.value.to_numpy()
    wanted = head_rows.loc[indices, "hour"].to_numpy()
    pos = np.searchsorted(times, wanted - 1e-7, side="left") - 1
    valid = pos >= 0
    indices = np.asarray(indices)
    head_rows.loc[indices[valid], "last_value"] = vals[pos[valid]]
    head_rows.loc[indices[valid], "last_hour"] = times[pos[valid]]

head_rows["last_bin"] = 0
head_rows["last_z"] = 0.0
for event, indices in head_rows.groupby("event").groups.items():
    spec = numeric_specs[(event, "value")]
    vals = head_rows.loc[indices, "last_value"].to_numpy()
    valid = np.isfinite(vals)
    indices = np.asarray(indices)
    head_rows.loc[indices[valid], "last_bin"] = fine_bin(event, "value", vals[valid])
    head_rows.loc[indices[valid], "last_z"] = np.clip(
        (vals[valid] - spec["median"]) / spec["iqr"], -10, 10)

head_weight_names = [
    "core.flow.a", "core.out_norm.weight", "core.out_norm.bias",
    "gap.0.weight", "gap.0.bias", "gap.2.weight", "gap.2.bias",
    "z_prior.0.weight", "z_prior.0.bias", "z_prior.2.weight", "z_prior.2.bias",
    "z_head.weight", "z_head.bias", "query_norm.weight", "query_norm.bias",
    "type_emb.weight", "embed.code.weight", "last_numeric.weight", "last_numeric.bias",
    "embed.bins.value.weight", "embed.bins.rate.weight", "last_status.weight",
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias",
    "embed_transition.value_bin.weight"
]
head_weights = {name: torch.as_tensor(load_original(name), dtype=torch.float32)
                for name in head_weight_names}
def linear_named(v, prefix, weights=head_weights):
    return F.linear(v, weights[prefix+".weight"], weights.get(prefix+".bias"))
def mlp_named(v, prefix, weights=head_weights):
    return linear_named(F.gelu(linear_named(v, prefix+".0", weights)),
                        prefix+".2", weights)
def norm_named(v, prefix, weights=head_weights):
    return F.layer_norm(v, (192,), weights[prefix+".weight"], weights[prefix+".bias"])

def reconstruct_head(rows, age_unit="seconds", layout="interleaved",
                     status=0, type_index=1, gap_scale=0.1,
                     z_samples=0, seed=821):
    previous = rows.previous_state.to_numpy(dtype=int)
    sx = torch.from_numpy(state_x[previous].astype(np.float32))
    dt = torch.tensor((rows.hour.to_numpy() - state_hours[previous]) * 3600,
                      dtype=torch.float32).clamp_min(1.0)
    h = sx[:,192:960].reshape(-1,4,192)
    flowed = h * torch.exp(-dt[:,None,None] * head_weights["core.flow.a"].exp())
    gap = mlp_named(dt.log()[:,None] * gap_scale, "gap")
    base = sx[:,:192] + norm_named(flowed.mean(1), "core.out_norm") + gap
    prior = mlp_named(base, "z_prior")
    mean_z, logsd_z = prior[:,:32], prior[:,32:]
    cid = torch.tensor([code_ids[e] for e in rows.event], dtype=torch.long)
    last_bin = torch.tensor(rows.last_bin.to_numpy(), dtype=torch.long)
    has = torch.tensor(rows.last_value.notna().to_numpy(), dtype=torch.float32)
    age = (rows.hour.to_numpy() - rows.last_hour.fillna(rows.hour).to_numpy())
    if age_unit == "seconds":
        age = age * 3600
    age_feature = torch.tensor(np.log1p(np.maximum(age, 0)) / 10, dtype=torch.float32)
    last_z = torch.tensor(rows.last_z.to_numpy(), dtype=torch.float32)
    last_features = torch.zeros(len(rows),8)
    last_features[:,0] = has
    last_features[:,1] = age_feature
    last_features[:,2] = last_z
    last_features[:,3 if layout == "interleaved" else 5] = has
    register = (
        head_weights["embed.code.weight"][cid]
        + linear_named(last_features, "last_numeric")
        + head_weights["embed.bins.value.weight"][last_bin]
        + head_weights["embed.bins.rate.weight"][0]
        + head_weights["last_status.weight"][status]
    )
    transition = head_weights["embed_transition.value_bin.weight"][cid*17+last_bin]
    centres = torch.tensor(np.array([VOCAB["value_bin_centres"][e] for e in rows.event]),
                           dtype=torch.float32)
    if centres.shape[1] == 16:
        centres = torch.cat([torch.zeros(len(rows),1), centres], dim=1)
    generator = torch.Generator().manual_seed(seed)
    probs, activations = [], []
    for draw in range(max(1, z_samples)):
        z = mean_z
        if z_samples:
            noise = torch.randn(mean_z.shape, generator=generator)
            z = mean_z + logsd_z.clamp(-6,2).exp() * noise
        view = base + linear_named(z, "z_head")
        q = norm_named(view + head_weights["type_emb.weight"][type_index], "query_norm")
        with_code = q + register
        hidden = F.gelu(linear_named(with_code, "attr_heads.value_bin.0"))
        logits = linear_named(hidden, "attr_heads.value_bin.2") + transition
        probs.append(logits.softmax(-1))
        activations.append(hidden)
    p = torch.stack(probs).mean(0)
    return (p*centres).sum(1).numpy(), p.numpy(), torch.stack(activations).mean(0).numpy()

grid_rows = head_rows[head_rows.hour >= 1].sample(
    n=min(500, int((head_rows.hour >= 1).sum())), random_state=821)
grid_scores = []
for age_unit, layout, status, type_index, gap_scale in itertools.product(
        ("seconds","hours"), ("interleaved","grouped"), (0,1), (1,0), (0.1,1.0)):
    predicted, _, _ = reconstruct_head(
        grid_rows, age_unit, layout, status, type_index, gap_scale)
    err = predicted - grid_rows.predicted_mean.to_numpy()
    grid_scores.append({
        "age_unit": age_unit, "layout": layout, "status": status,
        "type_index": type_index, "gap_scale": gap_scale,
        "mean_abs_difference": np.mean(np.abs(err)),
        "rms_difference": np.sqrt(np.mean(err**2))
    })
head_grid = pd.DataFrame(grid_scores).sort_values("rms_difference")
print("\n=== BEST RECONSTRUCTIONS ===")
print(head_grid.head(8).round(4).to_string(index=False))
best_head_settings = head_grid.iloc[0][
    ["age_unit","layout","status","type_index","gap_scale"]].to_dict()
best_head_settings["status"] = int(best_head_settings["status"])
best_head_settings["type_index"] = int(best_head_settings["type_index"])
local_mean, local_probs, local_hidden = reconstruct_head(
    head_rows, **best_head_settings, z_samples=32)
head_rows["local_mean"] = local_mean
print("\n=== FULL HEAD AGREEMENT (32 PRIOR DRAWS) ===")
for event, rows in head_rows[head_rows.hour >= 1].groupby("event"):
    diff = rows.local_mean - rows.predicted_mean
    print(event, "n", len(rows), "MAE", round(diff.abs().mean(),4),
          "RMS", round(np.sqrt(np.mean(diff**2)),4),
          "bias", round(diff.mean(),4))
print("Best settings:", best_head_settings)

with open("/work/notes.md", "a") as f:
    f.write("\n## Local head reconstruction — Python 9; forward 4\n"
            "- Captured original states and factual predictions for 100 training stays (original NLL 1.63805).\n")
    f.write(head_grid.head(8).to_string(index=False) + "\n")
    f.write("Best head settings: " + repr(best_head_settings) + "\n")
```

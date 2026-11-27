Flexible history adjustment puts norepinephrine’s one-hour association near +3.3 mmHg per unit of rate across lag choices, but that is not yet a causal result. Meanwhile, insulin’s record-derived sustained target is about −10.7 on the first tuning sample. I’ll reconstruct the value head from saved states; agreement with GPU predictions will tell us whether it is safe to compute a targeted weight edit locally.

```python
import torch.nn.functional as F
from itertools import product
import time

head_prefixes = ("attr_heads.value_bin.", "embed.code.", "embed.bins.",
                 "last_numeric.", "last_status.", "query_norm.", "type_emb.",
                 "core.flow.", "core.out_norm.", "gap.", "z_prior.", "z_head.")
W = {name: load_original(name) for name in NAMES if name.startswith(head_prefixes)}
W = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in W.items()}
code_ids = {entry["name"]: entry["id"] for entry in VOCAB["codes"]}
norms = {(entry["code"], entry["attribute"]): entry for entry in VOCAB["numeric"]}
def lin(x, prefix, weights=W):
    return F.linear(x, weights[prefix + ".weight"], weights.get(prefix + ".bias"))
def mlp(x, prefix, weights=W):
    return lin(F.gelu(lin(x, prefix + ".0", weights)), prefix + ".2", weights)
def ln(x, prefix, weights=W):
    return F.layer_norm(x, (x.shape[-1],),
                        weights[prefix + ".weight"], weights[prefix + ".bias"], 1e-5)

def reading_metadata(records, states):
    measured = records[records.event.isin(VOCAB["value_bin_centres"])].copy()
    measured = measured.sort_values(["stay_id", "hour", "event"])
    grouped = measured.groupby(["stay_id", "event"], sort=False)
    measured["last_value"] = grouped.value.shift(1)
    measured["last_hour"] = grouped.hour.shift(1)
    state_lookup = {}
    for sid in np.unique(states["stay_id"]):
        idx = np.flatnonzero(states["stay_id"] == sid)
        state_lookup[int(sid)] = (states["hour"][idx], idx)
    rows, previous, delta = [], [], []
    for i, row in measured.iterrows():
        if int(row.stay_id) not in state_lookup:
            continue
        hrs, idx = state_lookup[int(row.stay_id)]
        j = np.searchsorted(hrs, row.hour, side="left") - 1
        if j < 0:
            continue
        rows.append(i)
        previous.append(idx[j])
        delta.append((row.hour - hrs[j]) * 3600.)
    measured = measured.loc[rows].reset_index(drop=True)
    measured["state_row"] = previous
    measured["dt_seconds"] = delta
    measured["code"] = measured.event.map(code_ids).astype(int)
    measured["has_last"] = measured.last_value.notna().astype(float)
    measured["last_age_seconds"] = ((measured.hour - measured.last_hour) * 3600.).fillna(0)
    measured["last_z"] = 0.
    measured["last_bin"] = 0
    measured["observed_bin"] = 0
    for code, norm in [(c, norms[(c, "value")]) for c in VOCAB["value_bin_centres"]]:
        mask = measured.event == code
        values = measured.loc[mask, "last_value"].to_numpy()
        ok = np.isfinite(values)
        z = np.zeros(len(values))
        bins = np.zeros(len(values), dtype=int)
        z[ok] = np.clip((values[ok] - norm["median"]) / norm["iqr"], -10, 10)
        bins[ok] = 1 + np.searchsorted(norm["fine_bin_edges"], values[ok], side="left")
        measured.loc[mask, "last_z"] = z
        measured.loc[mask, "last_bin"] = bins
        measured.loc[mask, "observed_bin"] = 1 + np.searchsorted(
            norm["fine_bin_edges"], measured.loc[mask, "value"], side="left")
    return measured

head_meta_tune = reading_metadata(tuning, state_files[3])
head_meta_train = reading_metadata(train_small, state_files[4])
for key in ("embed_transition.value_bin.weight",):
    W[key] = torch.as_tensor(load_original(key), dtype=torch.float32)

def make_head_features(meta, states, gap_style="seconds_div10",
                       register_style="interleaved", status=0, type_id=1,
                       age_style="seconds", view_style="mean_prior"):
    indices = meta.state_row.to_numpy(int)
    x = torch.from_numpy(states["x"][indices])
    out, h = x[:, :192], x[:, 192:960].reshape(-1, 4, 192)
    dt = torch.tensor(meta.dt_seconds.to_numpy(), dtype=torch.float32)
    hflow = h * torch.exp(-dt[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :])
    if gap_style == "seconds_div10":
        gt = torch.log1p(dt) / 10
    elif gap_style == "seconds":
        gt = torch.log1p(dt)
    elif gap_style == "hours":
        gt = torch.log1p(dt / 3600)
    else:
        gt = torch.log1p(dt / 60) / 10
    base = out + ln(hflow.mean(1), "core.out_norm") + mlp(gt[:, None], "gap")
    prior = mlp(base, "z_prior")
    z = prior[:, :32] if view_style == "mean_prior" else x[:, 960:992]
    view = base + lin(z, "z_head")
    q = ln(view + W["type_emb.weight"][type_id], "query_norm")
    last = torch.tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    age = torch.tensor(meta.last_age_seconds.to_numpy(), dtype=torch.float32)
    age = torch.log1p(age if age_style == "seconds" else age / 3600) / 10
    value = torch.tensor(meta.last_z.to_numpy(), dtype=torch.float32)
    reg = torch.zeros(len(meta), 8)
    if register_style == "interleaved":
        reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 3] = last, age, value, last
    elif register_style == "numbers_first":
        reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 5] = last, age, value, last
    elif register_style == "has_last_only":
        reg[:, 0], reg[:, 1], reg[:, 2] = last, age, value
    elif register_style == "flags_first":
        reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 3] = last, age, last, value
    code = torch.tensor(meta.code.to_numpy(), dtype=torch.long)
    bins = torch.tensor(meta.last_bin.to_numpy(), dtype=torch.long)
    wc = (q + W["embed.code.weight"][code] + lin(reg, "last_numeric")
          + W["embed.bins.value.weight"][bins]
          + W["embed.bins.rate.weight"][0] + W["last_status.weight"][status])
    hidden = F.gelu(lin(wc, "attr_heads.value_bin.0"))
    logits = lin(hidden, "attr_heads.value_bin.2")
    logits += W["embed_transition.value_bin.weight"][code * 17 + bins]
    probs = logits.softmax(-1)
    centres = torch.zeros(len(meta), 17)
    for event, vals in VOCAB["value_bin_centres"].items():
        mask = (meta.event == event).to_numpy()
        centres[mask, 1:] = torch.tensor(vals, dtype=torch.float32)
    means = (probs * centres).sum(-1).numpy()
    nll = -probs[torch.arange(len(meta)), torch.tensor(meta.observed_bin.to_numpy(), dtype=torch.long)].log().numpy()
    return hidden.numpy(), logits.numpy(), means, nll, prior.numpy()

reference = head_meta_tune.merge(factual_baseline, on=["stay_id", "hour", "event"])
lookup_meta = head_meta_tune.set_index(["stay_id", "hour", "event"])
reference_meta = lookup_meta.loc[list(zip(reference.stay_id, reference.hour, reference.event))].reset_index()
refmean = reference.predicted_mean.to_numpy()
reconstruction_scores = []
for gap_style, register_style, status in product(
        ["seconds_div10", "seconds", "hours"],
        ["interleaved", "numbers_first", "has_last_only", "flags_first"], [0, 1]):
    _, _, mean, nll, prior = make_head_features(
        reference_meta, state_files[3], gap_style, register_style, status)
    reconstruction_scores.append({
        "gap": gap_style, "register": register_style, "status": status,
        "mean_rmse_vs_gpu": float(np.sqrt(np.mean((mean - refmean)**2))),
        "nll": float(np.mean(nll)), "mean_corr": float(np.corrcoef(mean, refmean)[0, 1])})
reconstruction_scores.sort(key=lambda a: a["mean_rmse_vs_gpu"])
print("Reading counts train / tuning:", len(head_meta_train), len(head_meta_tune))
print("Best head reconstructions:")
print(json.dumps(reconstruction_scores[:10], indent=2))
print("Original GPU factual NLL:", factual_baseline.nll.mean())
print("Last-numeric column norms:", torch.linalg.norm(W["last_numeric.weight"], dim=0).numpy())
```

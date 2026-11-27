The training-state sample has factual NLL 1.6432; this is a separate baseline, not a degradation. I’ll now test the head reconstruction against its means and bin likelihoods. Small discrepancies across all four reading types would validate it; systematic discrepancies would expose an incorrect implementation assumption.

```python
import itertools, time
import torch
import torch.nn.functional as TF
from scipy.special import softmax, logsumexp

torch.set_grad_enabled(False)
torch.set_num_threads(4)
t0 = time.time()

head_data = pd.read_csv("/work/fwd_4/factual_original.csv")
r0 = train[train.stay_id.isin(state_train_ids) & train.event.isin(measurement_events)].copy()
r0 = r0.sort_values(["stay_id", "hour", "event"])
r0["last_value"] = r0.groupby(["stay_id", "event"]).value.shift()
r0["last_hour"] = r0.groupby(["stay_id", "event"]).hour.shift()
head_data = head_data.merge(
    r0[["stay_id", "hour", "event", "last_value", "last_hour"]],
    on=["stay_id", "hour", "event"], how="left", validate="one_to_one"
)

state_groups = {
    int(sid): np.flatnonzero(S["stay_id"] == sid)
    for sid in np.unique(S["stay_id"])
}
prior_indices = []
for row in head_data.itertuples():
    ii = state_groups[int(row.stay_id)]
    j = np.searchsorted(S["hour"][ii], row.hour, side="left") - 1
    prior_indices.append(int(ii[j]) if j >= 0 else -1)
head_data["state_index"] = prior_indices
head_data = head_data[head_data.state_index >= 0].reset_index(drop=True)
head_data["dt_s"] = (
    head_data.hour.to_numpy() - S["hour"][head_data.state_index.to_numpy()]
) * 3600
head_data["has_last"] = head_data.last_value.notna().astype(float)
head_data["last_age_s"] = (
    (head_data.hour - head_data.last_hour).fillna(0).to_numpy() * 3600
)

code_ids = {c["name"]: c["id"] for c in VOCAB["codes"]}
numeric_specs = {(v["code"], v["attribute"]): v for v in VOCAB["numeric"]}
head_data["code_id"] = head_data.event.map(code_ids)
head_data["z_last"] = 0.0
head_data["last_bin"] = 0
head_data["observed_bin"] = 0
for event in head_data.event.unique():
    m = head_data.event.eq(event)
    spec = numeric_specs[event, "value"]
    vals = head_data.loc[m, "last_value"].to_numpy()
    good = np.isfinite(vals)
    zz = np.zeros(len(vals), np.float32)
    zz[good] = np.clip((vals[good] - spec["median"]) / spec["iqr"], -10, 10)
    bb = np.zeros(len(vals), np.int64)
    bb[good] = 1 + np.searchsorted(spec["fine_bin_edges"], vals[good], side="left")
    head_data.loc[m, "z_last"] = zz
    head_data.loc[m, "last_bin"] = bb
    head_data.loc[m, "observed_bin"] = 1 + np.searchsorted(
        spec["fine_bin_edges"], head_data.loc[m, "observed"], side="left"
    )

head_weights = {}
def hw(name):
    if name not in head_weights:
        head_weights[name] = torch.as_tensor(np.asarray(load_original(name)), dtype=torch.float32)
    return head_weights[name]

def act_fn(x, activation):
    return {"gelu": TF.gelu, "silu": TF.silu, "relu": TF.relu}[activation](x)

def head_mlp(x, prefix, activation):
    a = TF.linear(x, hw(prefix + ".0.weight"), hw(prefix + ".0.bias"))
    return TF.linear(act_fn(a, activation), hw(prefix + ".2.weight"), hw(prefix + ".2.bias"))

def head_norm(x, prefix):
    return TF.layer_norm(x, (192,), hw(prefix + ".weight"), hw(prefix + ".bias"))

test_rows = np.random.default_rng(361).choice(len(head_data), size=min(400, len(head_data)), replace=False)
hd = head_data.iloc[test_rows]
sx = torch.tensor(S["x"][hd.state_index.to_numpy()])
dt = torch.tensor(hd.dt_s.to_numpy(), dtype=torch.float32)
flow_h = sx[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -dt[:, None, None] * torch.exp(hw("core.flow.a"))[None, None, :]
)
base_ctx = sx[:, :192] + head_norm(flow_h.mean(1), "core.out_norm")
cid = torch.tensor(hd.code_id.to_numpy(), dtype=torch.long)
lb = torch.tensor(hd.last_bin.to_numpy(), dtype=torch.long)
ob = torch.tensor(hd.observed_bin.to_numpy(), dtype=torch.long)
hl = torch.tensor(hd.has_last.to_numpy(), dtype=torch.float32)
zl = torch.tensor(hd.z_last.to_numpy(), dtype=torch.float32)
centres = torch.tensor(np.array([
    [0.0] + VOCAB["value_bin_centres"][e] for e in hd.event
]), dtype=torch.float32)
transition = hw("embed_transition.value_bin.weight")[cid * 17 + lb]
true_mean = hd.predicted_mean.to_numpy()
true_nll = hd.nll.to_numpy()

# (z-value column, age column, columns holding identical has-last/has-value flags)
layouts = [
    (2, 1, (0, 3)), (2, 1, (0, 4)), (2, 1, (0, 5)),
    (0, 7, (1, 6)), (2, 7, (0, 1)), (3, 1, (0, 2)),
    (5, 1, (0, 2)), (2, 1, (0,)), (0, 1, (6,)),
]
lnproj = hw("last_numeric.weight").T @ hw("attr_heads.value_bin.0.weight").T
search_rows = []
for activation, gap_mode, type_id, status_id in itertools.product(
    ("gelu", "silu", "relu"), ("seconds_div10", "seconds", "hours_div10"),
    (0, 1, 3), (0, 1)
):
    if gap_mode == "seconds_div10":
        gap_x = torch.log1p(dt)[:, None] / 10
    elif gap_mode == "seconds":
        gap_x = torch.log1p(dt)[:, None]
    else:
        gap_x = torch.log1p(dt / 3600)[:, None] / 10
    ctx = base_ctx + head_mlp(gap_x, "gap", activation)
    zp = head_mlp(ctx, "z_prior", activation)[:, :32]
    view = ctx + TF.linear(zp, hw("z_head.weight"), hw("z_head.bias"))
    qv = head_norm(view + hw("type_emb.weight")[type_id], "query_norm")
    fixed_input = (
        qv + hw("embed.code.weight")[cid] + hw("embed.bins.value.weight")[lb]
        + hw("embed.bins.rate.weight")[0] + hw("last_numeric.bias")
        + hw("last_status.weight")[status_id]
    )
    fixed_pre = TF.linear(
        fixed_input, hw("attr_heads.value_bin.0.weight"), hw("attr_heads.value_bin.0.bias")
    )
    for age_mode in ("seconds_div10", "hours_div10"):
        ag = torch.tensor(hd.last_age_s.to_numpy(), dtype=torch.float32)
        age_x = torch.log1p(ag if age_mode == "seconds_div10" else ag / 3600) / 10
        for zcol, acol, flags in layouts:
            pre = fixed_pre + zl[:, None] * lnproj[zcol] + age_x[:, None] * lnproj[acol]
            pre = pre + hl[:, None] * lnproj[list(flags)].sum(0)
            logits = TF.linear(
                act_fn(pre, activation),
                hw("attr_heads.value_bin.2.weight"),
                hw("attr_heads.value_bin.2.bias")
            ) + transition
            prob = torch.softmax(logits, -1)
            pm = (prob * centres).sum(-1).numpy()
            pn = (-torch.log(prob[torch.arange(len(hd)), ob].clamp_min(1e-12))).numpy()
            mr = float(np.sqrt(np.mean((pm - true_mean) ** 2)))
            nr = float(np.sqrt(np.mean((pn - true_nll) ** 2)))
            search_rows.append(dict(
                activation=activation, gap_mode=gap_mode, type_id=type_id,
                status_id=status_id, age_mode=age_mode, zcol=zcol, acol=acol,
                flags=flags, mean_rmse=mr, nll_rmse=nr, score=mr + 3 * nr
            ))
head_search_results = pd.DataFrame(search_rows).sort_values("score")
head_cfg = head_search_results.iloc[0].to_dict()
print("=== BEST HEAD RECONSTRUCTIONS ===")
print(head_search_results.head(12).to_string(index=False))
print("Aligned readings:", len(head_data), "seconds:", round(time.time() - t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 16: local value-head reconstruction\n"
            "Forward 4 provides the factual baseline on the 100 state-sampled training stays: "
            "NLL 1.64320, 4660 readings. Testing activation, query type, gap scaling, "
            "and last-register layout before relying on a head-only edit.\n"
            + head_search_results.head(5).to_json(orient="records") + "\n")
```
On these dose schedules, the provisional insulin model predicts roughly −13.6 for doubling and +6.8 for halving—much larger than the original model’s shifts. Norepinephrine’s target is also potentially substantial, but depends on persistence. Next I’ll reconstruct the reading decoder from the saved states. Close agreement with the forward tool is necessary before using local calculations to design a precise edit.

```python
import itertools
import time

CODE_ID = {x["name"]: x["id"] for x in VOCAB["codes"]}
NUM_SPEC = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}

def linear0(x, prefix):
    return F.linear(x, W0[prefix + ".weight"], W0.get(prefix + ".bias"))

def norm0(x, prefix):
    return F.layer_norm(x, (192,), W0[prefix + ".weight"],
                        W0[prefix + ".bias"], 1e-5)

def activation0(x, kind="gelu"):
    return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[kind](x)

def mlp0(x, prefix, kind="gelu"):
    return linear0(activation0(linear0(x, prefix + ".0"), kind), prefix + ".2")

# Align every factual prediction with the strictly preceding bundle state
# and the strictly preceding reading of the same code.
hm_rows = []
state_groups = {sid: g.sort_values("hour") for sid, g in state_df.groupby("stay_id")}
record_groups = {(sid, event): g.sort_values("hour")
                 for (sid, event), g in tune.groupby(["stay_id", "event"])}
for row in factual0.itertuples(index=False):
    sg = state_groups[row.stay_id]
    pos = np.searchsorted(sg.hour.to_numpy(), row.hour - 1e-6) - 1
    if pos < 0:
        continue
    prev_state = sg.iloc[pos]
    rg = record_groups[(row.stay_id, row.event)]
    lp = np.searchsorted(rg.hour.to_numpy(), row.hour - 1e-6) - 1
    spec0 = NUM_SPEC[(row.event, "value")]
    if lp >= 0:
        old = rg.iloc[lp]
        last_z = np.clip((old.value - spec0["median"]) / spec0["iqr"], -10, 10)
        last_bin = 1 + np.searchsorted(spec0["fine_bin_edges"], old.value, side="left")
        age = (row.hour - old.hour) * 3600
        has_last = 1.0
    else:
        last_z, last_bin, age, has_last = 0.0, 0, 0.0, 0.0
    hm_rows.append((row.stay_id, row.hour, row.event, CODE_ID[row.event],
                    int(prev_state.state_row),
                    (row.hour - prev_state.hour) * 3600,
                    last_z, last_bin, age, has_last, row.observed,
                    row.predicted_mean, row.nll))
head_meta = pd.DataFrame(hm_rows, columns=[
    "stay_id", "hour", "event", "code", "state_row", "dt",
    "last_z", "last_bin", "age", "has_last", "observed", "gpu_mean", "gpu_nll"])
state_x = torch.from_numpy(S0["x"][head_meta.state_row.to_numpy()])
head_codes = torch.tensor(head_meta.code.to_numpy(), dtype=torch.long)
head_bins = torch.tensor(head_meta.last_bin.to_numpy(), dtype=torch.long)
head_centres = torch.tensor([[0.] + VOCAB["value_bin_centres"][e]
                             for e in head_meta.event], dtype=torch.float32)
head_transition = W0["embed_transition.value_bin.weight"][head_codes * 17 + head_bins]
head_static = (W0["embed.code.weight"][head_codes]
               + W0["embed.bins.value.weight"][head_bins]
               + W0["embed.bins.rate.weight"][0]
               + W0["last_status.weight"][0]
               + W0["last_numeric.bias"])
dt0 = torch.tensor(head_meta.dt.to_numpy(), dtype=torch.float32)[:, None]
flow_h = state_x[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -dt0[:, None, :] * torch.exp(W0["core.flow.a"])[None, None, :])

def decoder_query(kind="gelu", gap_style="log1p", type_id=1):
    gap_x = (torch.log1p(dt0) if gap_style == "log1p"
             else torch.log(dt0.clamp_min(1))) / 10
    ctx = state_x[:, :192] + norm0(flow_h.mean(1), "core.out_norm") + mlp0(gap_x, "gap", kind)
    zp = mlp0(ctx, "z_prior", kind)
    view = ctx + linear0(zp[:, :32], "z_head")
    return norm0(view + W0["type_emb.weight"][type_id], "query_norm"), zp

def head_calc(input0, idx=None, kind="gelu"):
    if idx is None:
        trans0, ctr0 = head_transition, head_centres
    else:
        trans0, ctr0 = head_transition[idx], head_centres[idx]
    hidden0 = activation0(linear0(input0, "attr_heads.value_bin.0"), kind)
    logits0 = linear0(hidden0, "attr_heads.value_bin.2") + trans0
    prob0 = logits0.softmax(-1)
    return (prob0 * ctr0).sum(-1), hidden0, logits0

# Enumerate the sparse 8-field register layout rather than assume its order.
rng_decode = np.random.default_rng(1101)
idx0 = torch.tensor(rng_decode.choice(len(head_meta), 180, replace=False))
truth0 = torch.tensor(head_meta.gpu_mean.to_numpy(), dtype=torch.float32)[idx0]
has0 = torch.tensor(head_meta.has_last.to_numpy(), dtype=torch.float32)[idx0]
z0 = torch.tensor(head_meta.last_z.to_numpy(), dtype=torch.float32)[idx0]
age0 = torch.log1p(torch.tensor(head_meta.age.to_numpy(), dtype=torch.float32)[idx0]) / 10
layouts = []
for iz in range(8):
    for ia in range(8):
        if ia == iz:
            continue
        remaining = [j for j in range(8) if j not in (iz, ia)]
        for n_presence in [1, 2]:
            for presence in itertools.combinations(remaining, n_presence):
                layouts.append((iz, ia, presence))
layout_vecs = torch.zeros(len(layouts), len(idx0), 8)
for j, (iz, ia, presence) in enumerate(layouts):
    layout_vecs[j, :, iz] = z0
    layout_vecs[j, :, ia] = age0
    for ip in presence:
        layout_vecs[j, :, ip] = has0

layout_results = []
for act0 in ["gelu", "silu", "relu"]:
    q0, zp0 = decoder_query(act0)
    base0 = q0[idx0] + head_static[idx0]
    best0 = (float("inf"), None)
    for start0 in range(0, len(layouts), 40):
        lv = layout_vecs[start0:start0 + 40]
        inp = base0[None] + F.linear(lv, W0["last_numeric.weight"])
        mn, _, _ = head_calc(inp, idx0, act0)
        err = ((mn - truth0[None]) ** 2).mean(-1)
        j0 = int(err.argmin())
        if float(err[j0]) < best0[0]:
            best0 = (float(err[j0]), layouts[start0 + j0])
    layout_results.append((act0, *best0))
print("Best register layouts (MSE against GPU means):", layout_results)
best_act, _, best_layout = min(layout_results, key=lambda a: a[1])
register8 = torch.zeros(len(head_meta), 8)
iz, ia, presence = best_layout
register8[:, iz] = torch.tensor(head_meta.last_z.to_numpy(), dtype=torch.float32)
register8[:, ia] = torch.log1p(torch.tensor(head_meta.age.to_numpy(), dtype=torch.float32)) / 10
for ip in presence:
    register8[:, ip] = torch.tensor(head_meta.has_last.to_numpy(), dtype=torch.float32)

q_head, z_prior_head = decoder_query(best_act)
head_input = q_head + head_static + F.linear(register8, W0["last_numeric.weight"])
local_mean, head_hidden, local_logits = head_calc(head_input, kind=best_act)
head_meta["local_mean"] = local_mean.numpy()
head_meta["mean_error"] = head_meta.local_mean - head_meta.gpu_mean
print("Chosen activation/layout:", best_act, best_layout)
print("Decoder error by event:\n",
      head_meta.groupby("event").mean_error.agg(
          ["mean", "std", lambda a: np.sqrt(np.mean(a*a))]).round(4))
print("Prior log-SD quantiles:",
      np.quantile(z_prior_head[:, 32:].numpy(), [.01, .5, .99]))
print("Largest decoder errors:\n",
      head_meta.loc[head_meta.mean_error.abs().nlargest(5).index,
                    ["hour", "event", "has_last", "gpu_mean", "local_mean", "mean_error"]]
      .round(3).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 11: local reading decoder reconstruction\n"
            f"Enumerated register layouts and activation types. Best: {best_act}, {best_layout}; "
            f"decoder RMSE {np.sqrt(np.mean(head_meta.mean_error**2)):.4f}. "
            "Provisional insulin x2 target on recorded reached hours: -13.586; "
            "half target +6.793, assuming beta=-2.2778 and persistence=.8574.\n"
            "Python runs: 11. Forward requests: 3.\n")
```
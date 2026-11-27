The prediction files contain the means and observed-bin NLLs needed for a reconstruction check. Some implementation details are absent from the architecture summary, especially the eight register inputs. I’ll test a few explicit interpretations rather than assume one. A clearly better match across both glucose and MAP would identify the correct convention.

```python
from scipy.special import softmax

def wlin(x, name, weights=W0):
    return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

def wnorm(x, name, weights=W0):
    return F.layer_norm(x, (192,), weights[name + ".weight"], weights[name + ".bias"])

def actfn(x, kind):
    return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[kind](x)

def wmlp(x, name, kind="gelu", weights=W0):
    return wlin(actfn(wlin(x, name + ".0", weights), kind), name + ".2", weights)

code_ids = {c["name"]: c["id"] for c in VOCAB["codes"]}
numeric_specs = {(e["code"], e["attribute"]): e for e in VOCAB["numeric"]}
state_by_stay = {}
for sid in np.unique(states0["stay_id"]):
    ii = np.flatnonzero(states0["stay_id"] == sid)
    state_by_stay[int(sid)] = ii

head_records = []
for sid, ff in fact_train0.groupby("stay_id", sort=False):
    rr = train[train.stay_id == sid]
    si = state_by_stay[int(sid)]
    sh = states0["hour"][si]
    for row in ff.itertuples(index=False):
        pos = np.searchsorted(sh, row.hour, side="left") - 1
        assert pos >= 0
        idx = si[pos]
        earlier = rr[(rr.event == row.event) & (rr.hour < row.hour) & rr.value.notna()]
        if len(earlier):
            last = earlier.iloc[-1]
            lv, age, has = float(last.value), (row.hour - last.hour) * 3600.0, 1.0
            spec = numeric_specs[(row.event, "value")]
            lz = (lv - spec["median"]) / spec["iqr"]
            lb = np.searchsorted(spec["fine_bin_edges"], lv, side="right") + 1
        else:
            lv, age, has, lz, lb = 0., 0., 0., 0., 0
        head_records.append((sid, row.hour, row.event, row.observed,
                             row.predicted_mean, row.nll, idx,
                             (row.hour - states0["hour"][idx]) * 3600.,
                             has, age, lz, lb))
head_meta = pd.DataFrame(head_records, columns=[
    "stay_id", "hour", "event", "observed", "tool_mean", "tool_nll",
    "state_idx", "dt", "has_last", "age", "last_z", "last_bin"])
hm = head_meta.iloc[:1000].copy()
sx = torch.tensor(states0["x"][hm.state_idx.to_numpy()])
out0, h0 = sx[:, :192], sx[:, 192:960].reshape(-1, 4, 192)
dt0 = torch.tensor(hm.dt.to_numpy(), dtype=torch.float32).reshape(-1, 1)
flowh = h0 * torch.exp(-dt0[:, None, :] * torch.exp(W0["core.flow.a"])[None, None, :])
hnorm = wnorm(flowh.mean(1), "core.out_norm")
cid = torch.tensor(hm.event.map(code_ids).to_numpy(), dtype=torch.long)
lb = torch.tensor(hm.last_bin.to_numpy(), dtype=torch.long)
centres = torch.tensor(np.stack([np.r_[0., VOCAB["value_bin_centres"][ev]]
                                for ev in hm.event]), dtype=torch.float32)
target_means = hm.tool_mean.to_numpy()
registers = []
for layout in ("value_then_flag", "flag_then_value"):
    for age_unit in ("seconds", "hours"):
        nf = np.zeros((len(hm), 8), dtype=np.float32)
        nf[:, 0] = hm.has_last
        agev = hm.age.to_numpy() / (3600. if age_unit == "hours" else 1.)
        nf[:, 1] = np.log1p(agev) / 10.
        if layout == "value_then_flag":
            nf[:, 2], nf[:, 3] = hm.last_z, hm.has_last
        else:
            nf[:, 2], nf[:, 3] = hm.has_last, hm.last_z
        reg = (wlin(torch.tensor(nf), "last_numeric")
               + W0["embed.bins.value.weight"][lb]
               + W0["embed.bins.rate.weight"][0]
               + W0["last_status.weight"][0])
        registers.append((layout, age_unit, reg))

head_trials = []
for activation in ("gelu", "silu", "relu"):
    gapv = wmlp(torch.log1p(dt0) / 10., "gap", activation)
    ctx = out0 + hnorm + gapv
    zmean = wmlp(ctx, "z_prior", activation)[:, :32]
    view = ctx + wlin(zmean, "z_head")
    for typ in range(15):
        q = wnorm(view + W0["type_emb.weight"][typ], "query_norm")
        for layout, age_unit, reg in registers:
            wc = q + W0["embed.code.weight"][cid] + reg
            hidden = actfn(wlin(wc, "attr_heads.value_bin.0"), activation)
            logits = (wlin(hidden, "attr_heads.value_bin.2")
                      + W0["embed_transition.value_bin.weight"][cid * 17 + lb])
            probs = logits.softmax(-1)
            means = (probs * centres).sum(-1).numpy()
            err = means - target_means
            head_trials.append((float(np.sqrt(np.mean(err**2))),
                                activation, typ, layout, age_unit,
                                float(np.mean(err)), float(probs[:, 0].mean())))
head_trials.sort()
print("Best reconstruction candidates (RMSE, activation, type, layout, age, bias, p0):")
for item in head_trials[:12]:
    print(item)
print("Matched reading/state pairs:", len(head_meta))
print(head_meta[["event", "dt", "has_last", "age", "last_bin"]].head(8).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 10\nLoaded original tensors and matched training factual CSV schema. Established complete numeric normalisation and fine-bin tables. Preparing a forward-head reconstruction; no model edits or external information used.\n")
```

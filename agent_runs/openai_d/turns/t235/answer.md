The earlier controlled probe established the reading register’s layout: presence in field 0, log age in field 3, and last normalized value in field 6, with left-sided bins. I’ll use that layout to build the tuning cache and compare both reconstructed head inputs and paired NLL changes. Close agreement would make the cache useful for revision screening.

```python
import torch.nn.functional as F
torch.set_num_threads(8)
r201_W = {n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in NAMES}

def r201_lin(x, p, W=r201_W):
    return F.linear(x, W[p+".weight"], W.get(p+".bias"))

def r201_mlp(x, p, W=r201_W):
    return r201_lin(F.gelu(r201_lin(x, p+".0", W)), p+".2", W)

def r201_norm(x, p, W=r201_W):
    return F.layer_norm(x, (192,), W[p+".weight"], W[p+".bias"])

def r201_ctx(s, meta):
    a = torch.as_tensor(s["x"][meta.state_idx.to_numpy(np.int64)], dtype=torch.float32)
    dt = torch.tensor(meta.dt.to_numpy(), dtype=torch.float32).reshape(-1,1)
    h = a[:,192:960].reshape(-1,4,192)
    hf = h * torch.exp(-dt[:,:,None] * torch.exp(r201_W["core.flow.a"])[None,None,:])
    return a[:,:192] + r201_norm(hf.mean(1), "core.out_norm") + r201_mlp(torch.log(dt)/10, "gap")

def r201_queries(ctx, eps):
    pp = r201_mlp(ctx, "z_prior")
    z = pp[None,:,:32] + pp[None,:,32:].clamp(-6,2).exp() * eps
    return r201_norm(ctx[None] + r201_lin(z,"z_head") + r201_W["type_emb.weight"][1], "query_norm")

r201_codes = {v["name"]: v["id"] for v in VOCAB["codes"]}
r201_numeric = {(v["code"],v["attribute"]):v for v in VOCAB["numeric"]}

def r201_fixed(meta, status=True, rate=True):
    codes = torch.tensor([r201_codes[e] for e in meta.event], dtype=torch.long)
    lb = torch.tensor(meta.last_bin.to_numpy(), dtype=torch.long)
    nf = torch.zeros(len(meta),8)
    nf[:,0] = torch.tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    nf[:,3] = nf[:,0] * torch.log1p(torch.tensor(meta.age.to_numpy(),dtype=torch.float32))/10
    nf[:,6] = torch.tensor(meta.last_z.to_numpy(),dtype=torch.float32)
    x = r201_W["embed.code.weight"][codes] + r201_lin(nf,"last_numeric")
    x = x + r201_W["embed.bins.value.weight"][lb]
    if rate:
        x = x + r201_W["embed.bins.rate.weight"][0]
    if status:
        x = x + r201_W["last_status.weight"][0]
    return x, codes, lb

r201_sel = np.arange(64)
r201_tm = head_meta.iloc[r201_sel].copy()
r201_tc = r201_ctx(dict(np.load("/work/fwd_3/states_original.npz")), r201_tm)
r201_tq = r201_queries(r201_tc, eps98[:,r201_sel])
r201_options = []
for st in [False, True]:
    for ra in [False, True]:
        xx = r201_tq + r201_fixed(r201_tm, st, ra)[0][None]
        err = float((xx-r124_x[:,r201_sel]).abs().max())
        r201_options.append((err,st,ra))
print("train reconstruction options", r201_options)
r201_err,r201_st,r201_ra = min(r201_options)
assert r201_err < 0.002, "Resolve head-input discrepancy before screening."

r201_rows = []
for sid, df in r200_data.groupby("stay_id",sort=True):
    si = np.flatnonzero(r200_s["stay_id"] == sid)
    hours = r200_s["hour"][si]
    assert np.all(np.diff(hours)>0), "State bundles must have distinct increasing hours."
    last = {}
    for hour, group in df.groupby("hour",sort=True):
        meas = group[group.event.isin(VOCAB["value_bin_centres"])]
        if len(meas) == 0:
            continue
        j = np.searchsorted(hours, hour, side="left")-1
        assert j >= 0
        for row in meas.itertuples():
            event = row.event
            info = r201_numeric[(event,"value")]
            old = last.get(event)
            z = 0. if old is None else (old[1]-info["median"])/info["iqr"]
            lb = 0 if old is None else 1+np.searchsorted(info["fine_bin_edges"],old[1],side="left")
            r201_rows.append((sid,hour,event,row.value,int(si[j]),(hour-hours[j])*3600,
                              float(old is not None),0. if old is None else (hour-old[0])*3600,
                              z,int(lb)))
        for row in meas.itertuples():
            last[row.event]=(hour,row.value)
r201_meta = pd.DataFrame(r201_rows,columns=[
    "stay_id","hour","event","observed","state_idx","dt","has_last","age","last_z","last_bin"])
assert len(r201_meta)==len(r199_gpu)
r201_ctx_tune = r201_ctx(r200_s,r201_meta)
r201_rng = torch.Generator().manual_seed(201)
r201_half = torch.randn((16,len(r201_meta),32),generator=r201_rng)
r201_eps = torch.cat([r201_half,-r201_half])
r201_fixed_tune,r201_code_tune,r201_lb_tune = r201_fixed(r201_meta,r201_st,r201_ra)
r201_X = r201_queries(r201_ctx_tune,r201_eps) + r201_fixed_tune[None]
r201_trans = r201_W["embed_transition.value_bin.weight"][r201_code_tune*17+r201_lb_tune]
r201_y = torch.tensor([1+np.searchsorted(r201_numeric[(r.event,"value")]["fine_bin_edges"],
                                       r.observed,side="left") for r in r201_meta.itertuples()])
r201_centres = torch.tensor([[0.]+VOCAB["value_bin_centres"][e] for e in r201_meta.event])

def r201_eval(W):
    h = F.gelu(F.linear(r201_X,W["attr_heads.value_bin.0.weight"],W["attr_heads.value_bin.0.bias"]))
    logits = F.linear(h,W["attr_heads.value_bin.2.weight"],W["attr_heads.value_bin.2.bias"])
    p = (logits+r201_trans[None]).softmax(-1).mean(0)
    loss = -p[torch.arange(len(r201_y)),r201_y].clamp_min(1e-20).log()
    mean = (p*r201_centres).sum(-1)
    return loss.numpy(),mean.numpy()

r201_loss0,r201_mean0 = r201_eval(r201_W)
r201_current = {n:torch.as_tensor(load(n),dtype=torch.float32) for n in edited()}
r201_lossc,r201_meanc = r201_eval(r201_current)
r201_check = r201_meta[["stay_id","hour","event","observed"]].copy()
r201_check["local_original_mean"]=r201_mean0
r201_check["local_original_nll"]=r201_loss0
r201_check["local_delta"]=r201_lossc-r201_loss0
r201_gpu_c=pd.read_csv("/work/fwd_27/factual_edited.csv")
r201_check=r201_check.merge(r199_gpu,on=["stay_id","hour","event","observed"],validate="one_to_one")
r201_check=r201_check.merge(r201_gpu_c,on=["stay_id","hour","event","observed"],
                           suffixes=("_original","_edited"),validate="one_to_one")
print("rows",len(r201_meta),"head shape",tuple(r201_X.shape))
print("original mean RMSE",np.sqrt(np.mean((r201_check.local_original_mean-r201_check.predicted_mean_original)**2)))
print("original NLL: local/GPU",float(r201_loss0.mean()),float(r199_gpu.nll.mean()))
print("paired NLL delta: local/GPU",float((r201_lossc-r201_loss0).mean()),
      float((r201_check.nll_edited-r201_check.nll_original).mean()))
print(r201_check.assign(gpu_delta=r201_check.nll_edited-r201_check.nll_original)
      .groupby("event")[["local_delta","gpu_delta"]].mean().to_string())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 201: independent tuning-cache reconstruction\n"
            f"Train head-input reconstruction max error {r201_err:.8g}. "
            f"Tuning 100–199 paired local delta {(r201_lossc-r201_loss0).mean():.8g}; "
            f"GPU delta {(r201_check.nll_edited-r201_check.nll_original).mean():.8g}. "
            "No weights changed.\n")
```

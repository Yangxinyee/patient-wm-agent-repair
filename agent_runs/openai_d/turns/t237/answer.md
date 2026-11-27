Context, prior, and query reconstruction agree with the recovered original helpers. The mismatch with `r124_x` varies across latent draws, so that variable is not a reliable baseline reference here. I’ll check the other cached references and compare the fresh tuning cache directly with GPU results, especially the edited-minus-original NLL change.

```python
r203_expected = r201_tq + r201_fixed(r201_tm,True,True)[0][None]
print("=== ALTERNATIVE OLD-CACHE CHECKS ===")
for name in ["X99","XX99","X187","M99"]:
    if name in globals():
        v = globals()[name]
        if tuple(v.shape)==(32,4510,192):
            print(name,"max discrepancy",float((r203_expected-v[:,r201_sel]).abs().max()))
if "fixed96" in globals():
    print("fixed96 discrepancy",float((r201_fixed(r201_tm,True,True)[0]-fixed96[r201_sel]).abs().max()))

r201_rows = []
for sid, df in r200_data.groupby("stay_id",sort=True):
    si = np.flatnonzero(r200_s["stay_id"] == sid)
    hours = r200_s["hour"][si]
    assert np.all(np.diff(hours)>0)
    last = {}
    for hour, group in df.groupby("hour",sort=True):
        meas = group[group.event.isin(VOCAB["value_bin_centres"])]
        if len(meas) == 0:
            continue
        j = np.searchsorted(hours,hour,side="left")-1
        assert j >= 0
        for row in meas.itertuples():
            info = r201_numeric[(row.event,"value")]
            old = last.get(row.event)
            z = 0. if old is None else (old[1]-info["median"])/info["iqr"]
            lb = 0 if old is None else 1+np.searchsorted(info["fine_bin_edges"],old[1],side="left")
            r201_rows.append((sid,hour,row.event,row.value,int(si[j]),(hour-hours[j])*3600,
                              float(old is not None),0. if old is None else (hour-old[0])*3600,z,int(lb)))
        for row in meas.itertuples():
            last[row.event]=(hour,row.value)
r201_meta = pd.DataFrame(r201_rows,columns=[
    "stay_id","hour","event","observed","state_idx","dt","has_last","age","last_z","last_bin"])
assert len(r201_meta)==len(r199_gpu)
r201_ctx_tune=r201_ctx(r200_s,r201_meta)
r201_rng=torch.Generator().manual_seed(201)
r201_half=torch.randn((16,len(r201_meta),32),generator=r201_rng)
r201_eps=torch.cat([r201_half,-r201_half])
r201_fixed_tune,r201_code_tune,r201_lb_tune=r201_fixed(r201_meta,True,True)
r201_X=r201_queries(r201_ctx_tune,r201_eps)+r201_fixed_tune[None]
r201_trans=r201_W["embed_transition.value_bin.weight"][r201_code_tune*17+r201_lb_tune]
r201_y=torch.tensor([1+np.searchsorted(r201_numeric[(r.event,"value")]["fine_bin_edges"],
                                    r.observed,side="left") for r in r201_meta.itertuples()])
r201_centres=torch.tensor([[0.]+VOCAB["value_bin_centres"][e] for e in r201_meta.event])

def r201_eval(W):
    h=F.gelu(F.linear(r201_X,W["attr_heads.value_bin.0.weight"],W["attr_heads.value_bin.0.bias"]))
    logits=F.linear(h,W["attr_heads.value_bin.2.weight"],W["attr_heads.value_bin.2.bias"])
    p=(logits+r201_trans[None]).softmax(-1).mean(0)
    loss=-p[torch.arange(len(r201_y)),r201_y].clamp_min(1e-20).log()
    mean=(p*r201_centres).sum(-1)
    return loss.numpy(),mean.numpy()

r201_loss0,r201_mean0=r201_eval(r201_W)
r201_current={n:torch.as_tensor(load(n),dtype=torch.float32) for n in edited()}
r201_lossc,r201_meanc=r201_eval(r201_current)
r201_check=r201_meta[["stay_id","hour","event","observed"]].copy()
r201_check["local_original_mean"]=r201_mean0
r201_check["local_original_nll"]=r201_loss0
r201_check["local_delta"]=r201_lossc-r201_loss0
r201_gpu_c=pd.read_csv("/work/fwd_27/factual_edited.csv")
r201_check=r201_check.merge(r199_gpu,on=["stay_id","hour","event","observed"],validate="one_to_one")
r201_check=r201_check.merge(r201_gpu_c,on=["stay_id","hour","event","observed"],
                           suffixes=("_original","_edited"),validate="one_to_one")
print("=== INDEPENDENT TUNING VALIDATION ===")
print("rows",len(r201_meta),"head shape",tuple(r201_X.shape))
print("original mean RMSE",np.sqrt(np.mean((r201_check.local_original_mean-r201_check.predicted_mean_original)**2)))
print("original NLL local/GPU",float(r201_loss0.mean()),float(r199_gpu.nll.mean()))
print("paired NLL delta local/GPU",float((r201_lossc-r201_loss0).mean()),
      float((r201_check.nll_edited-r201_check.nll_original).mean()))
print(r201_check.assign(gpu_delta=r201_check.nll_edited-r201_check.nll_original)
      .groupby("event")[["local_delta","gpu_delta"]].mean().to_string())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 203: fresh independent tuning cache\n"
            f"Tuning 100–199 paired local delta {(r201_lossc-r201_loss0).mean():.8g}; "
            f"GPU delta {(r201_check.nll_edited-r201_check.nll_original).mean():.8g}. "
            "r124_x provenance is not accepted; context/prior/query reconstruction was verified. "
            "No weights changed.\n")
```

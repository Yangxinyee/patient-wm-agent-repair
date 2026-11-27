The GPU confirms that the current edit preserves factual fit on these training stays: value NLL improved by 0.00253 nats. That does not validate the local reconstruction or causal response. I’ll now match local predictions to these GPU outputs; the size and pattern of disagreement should identify what remains wrong.

```python
import torch.nn.functional as F
torch.set_num_threads(8)
W225 = {n: load(n) for n in NAMES}
WO225 = {n: load_original(n) for n in NAMES}
def lin225(x, prefix, weights):
    return F.linear(x, weights[prefix+".weight"], weights.get(prefix+".bias"))
def ln225(x, prefix, weights):
    return F.layer_norm(x, (x.shape[-1],), weights[prefix+".weight"],
                        weights[prefix+".bias"], eps=1e-5)

codes225 = torch.tensor([code_id[e] for e in head_rows.event], dtype=torch.long)
obs225 = np.zeros(len(head_rows), dtype=np.int64)
for ev in sensor_neurons:
    ix = np.flatnonzero(head_rows.event.to_numpy() == ev)
    obs225[ix] = 1 + np.searchsorted(num_spec[(ev,"value")]["fine_bin_edges"],
                                   head_rows.value.to_numpy()[ix], side="left")
obs225 = torch.from_numpy(obs225)
trname225 = next(n for n in NAMES if n.startswith("embed_transition.value_bin"))
print("transition", trname225, SHAPES[trname225])
print("centres schema", str(V["value_bin_centres"])[:650])
print("weight-cache maximum difference",
      max(float((W225[n]-W[n]).abs().max()) for n in NAMES))

def head225(weights, sf, noise=None, status=None, typeidx=1):
    xx = torch.from_numpy(np.asarray(sf["x"][head_prev]))
    hh = xx[:,192:960].reshape(-1,4,192)
    dt = torch.tensor((head_rows.hour.to_numpy()-sf["hour"][head_prev])*3600,
                      dtype=torch.float32).clamp_min(1)
    hf = hh * torch.exp(-dt[:,None,None]*torch.exp(weights["core.flow.a"])[None,None,:])
    gp = lin225(F.gelu(lin225(torch.log1p(dt)[:,None]/10,"gap.0",weights)),
                "gap.2",weights)
    ctx = xx[:,:192]+ln225(hf.mean(1),"core.out_norm",weights)+gp
    zp = lin225(F.gelu(lin225(ctx,"z_prior.0",weights)),"z_prior.2",weights)
    zz = zp[:,:32] if noise is None else zp[:,:32]+torch.exp(zp[:,32:].clamp(-7,2))*noise
    qq = ln225(ctx+lin225(zz,"z_head",weights)+weights["type_emb.weight"][typeidx],
               "query_norm",weights)
    wc = (qq+weights["embed.code.weight"][codes225]
          +lin225(head_ln,"last_numeric",weights)
          +weights["embed.bins.value.weight"][torch.from_numpy(head_vb)]
          +weights["embed.bins.rate.weight"][0])
    if status is not None:
        wc = wc+weights["last_status.weight"][status]
    logits = lin225(F.gelu(lin225(wc,"attr_heads.value_bin.0",weights)),
                    "attr_heads.value_bin.2",weights)
    logits += weights[trname225][codes225*17+torch.from_numpy(head_vb)]
    return torch.softmax(logits,dim=-1), wc, qq, zp

for label, weights, path in [
    ("original",WO225,"/work/fwd_15/states_original.npz"),
    ("edited",W225,"/work/fwd_15/states_edited.npz")]:
    sf225 = np.load(path)
    gpu225 = pd.read_csv(f"/work/fwd_20/factual_{label}.csv")
    for stat in [None,0,1]:
        pp,wc225,qq225,zp225 = head225(weights,sf225,status=stat)
        nn = -torch.log(pp[torch.arange(len(pp)),obs225].clamp_min(1e-30)).numpy()
        comparison225 = head_rows[["stay_id","hour","event"]].copy()
        comparison225["local_nll"] = nn
        comparison225 = comparison225.merge(gpu225,on=["stay_id","hour","event"],validate="one_to_one")
        print(label,"status",stat,"local NLL",round(comparison225.local_nll.mean(),6),
              "GPU",round(comparison225.nll.mean(),6),
              "NLL RMSE",round(np.sqrt(np.mean((comparison225.local_nll-comparison225.nll)**2)),6))
    print(label,"prior logsd min/mean/max",
          float(zp225[:,32:].min()),float(zp225[:,32:].mean()),float(zp225[:,32:].max()))
```

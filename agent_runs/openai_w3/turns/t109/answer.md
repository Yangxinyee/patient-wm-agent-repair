The readout search strongly selects measurement type 1, but the remaining error may reflect latent sampling or an incomplete register convention. The dose coefficients also need their original centring and scaling restored before use. I’ll check both now; this should reveal whether the proposed correction is genuinely small on factual inputs.

```python
print("DOSE-FIT SCALING")
for n98 in ("mean88", "scale88", "fact88", "fact_norm88", "fx89", "train_fact88",
            "chosen90", "means90", "contrast_meta88", "dose_target88"):
    if n98 not in globals():
        continue
    o98 = globals()[n98]
    if isinstance(o98, dict):
        print(n98, {str(k): (type(v).__name__, getattr(v, "shape", None),
                           str(v)[:120] if np.isscalar(v) else "")
                    for k,v in o98.items()})
    else:
        print(n98, type(o98).__name__, getattr(o98, "shape", None),
              np.quantile(as_numpy(o98), [0,.5,1]).tolist()
              if hasattr(o98,"shape") and np.asarray(as_numpy(o98)).size else str(o98)[:120])

def reading_probs98(h, out, dt_s, codes, last_values, age_s, samples=0, seed=98):
    h = torch.as_tensor(h, dtype=torch.float32)
    out = torch.as_tensor(out, dtype=torch.float32)
    dt_s = torch.as_tensor(dt_s, dtype=torch.float32)
    codes = np.asarray(codes, dtype=int)
    last_values = np.asarray(last_values, dtype=np.float32)
    age_s = np.asarray(age_s, dtype=np.float32)
    has = np.isfinite(last_values)
    b = np.zeros(len(codes), dtype=int)
    z = np.zeros(len(codes), dtype=np.float32)
    for j,(c,v) in enumerate(zip(codes,last_values)):
        if has[j]:
            event = VOCAB["codes"][int(c)]["name"]
            tab = numeric_tables[(event,"value")]
            b[j] = 1 + np.sum(np.asarray(tab["fine_bin_edges"]) < v)
            z[j] = np.clip((v-tab["median"])/tab["iqr"], -10, 10)
    hf = h * torch.exp(-dt_s[:,None,None] * torch.exp(W["core.flow.a"])[None,None,:])
    ctx = out + layer_norm(hf.mean(1),"core.out_norm") + mlp(
        torch.log1p(dt_s)[:,None]/10, "gap")
    prior = mlp(ctx,"z_prior")
    if samples:
        gen = torch.Generator().manual_seed(seed)
        eps = torch.randn(samples,len(codes),32,generator=gen)
        latent = prior[None,:,:32] + eps * torch.exp(prior[None,:,32:].clamp(-10,5))
    else:
        latent = prior[None,:,:32]
    view = ctx[None,:,:] + linear(latent,"z_head")
    q = layer_norm(view + W["type_emb.weight"][1],"query_norm")
    numbers = torch.zeros(len(codes),8)
    numbers[:,0] = torch.tensor(has.astype(np.float32))
    numbers[:,1] = torch.tensor(np.where(has,np.log1p(np.nan_to_num(age_s))/10,0))
    numbers[:,6] = torch.tensor(z)
    numbers[:,7] = torch.tensor(has.astype(np.float32))
    register = (W["embed.code.weight"][codes] + linear(numbers,"last_numeric") +
                W["embed.bins.value.weight"][b] +
                W["embed.bins.rate.weight"][0] + W["last_status.weight"][0])
    logits = mlp(q+register[None,:,:], "attr_heads.value_bin") + \
             W["embed_transition.value_bin.weight"][codes*17+b][None,:,:]
    return torch.softmax(logits,-1)

p98 = reading_probs98(x97[:,192:960].reshape(-1,4,192), x97[:,:192], dt97,
                     codes97, lasts97, ages97, samples=128)
sample_means98 = (p98[:,:,1:] * torch.tensor(centres97)[None,:,:]).sum(-1).numpy()
mean98 = sample_means98.mean(0)
print("READOUT MC RMSE", np.sqrt(np.mean((mean98-observed_mean97)**2)))
for ev98 in read_codes90:
    m98 = np.array([r.event==ev98 for r in rows97])
    print(ev98, "RMSE", np.sqrt(np.mean((mean98[m98]-observed_mean97[m98])**2)),
          "bias", np.mean(mean98[m98]-observed_mean97[m98]),
          "expected 4-sample SD", np.sqrt(np.mean(sample_means98[:,m98].var(0)))/2)
print("PRIOR SD RANGE", torch.exp(mlp(ctx97,"z_prior")[:,32:]).quantile(
    torch.tensor([0.,.5,1.])).tolist())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 98; forwards 7. Implemented candidate local reading head "
            "(type=1; measurement register numeric columns 6/7) and tested prior-sampled "
            "means. Auditing retained feature centering/scaling before any injection edit.\n")
```
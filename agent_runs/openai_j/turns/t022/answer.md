Latent averaging did not improve agreement, so it cannot yet explain the mismatch. I’ll check bin-boundary conventions and whether the last-value register uses raw values or bin centres. I’ll also measure the error expected from four latent draws. Agreement after these checks would make the reconstructed features usable for a selective edit.

```python
# Quantify latent sampling noise and test last-register conventions.
act = "gelu"
ctx = context_base + mlp((torch.log1p(dt)/10)[:,None], "gap", act)
zp = mlp(ctx, "z_prior", act)
zz = zp[None,:,:32] + zp[None,:,32:].clamp(-5,2).exp()*eps_mc
queries_mc = ln(ctx[None,:,:] + TF.linear(zz, wt("z_head.weight"), wt("z_head.bias"))
                + wt("type_emb.weight")[1], "query_norm")

last_raw = np.array([
    hm.zlast.iloc[i]*numeric_specs[(e,"value")]["iqr"]
    + numeric_specs[(e,"value")]["median"] for i,e in enumerate(hm.event)
])
boundary = np.array([
    bool(hm.has_last.iloc[i]) and np.any(np.isclose(
        last_raw[i], numeric_specs[(e,"value")]["fine_bin_edges"], atol=1e-5))
    for i,e in enumerate(hm.event)
])

register_results = []
baseline_probs_mc = None
for side, value_mode, age_unit in product(["left","right"], ["raw","centre"], ["seconds","hours"]):
    bb = np.array([
        1+np.searchsorted(numeric_specs[(e,"value")]["fine_bin_edges"], last_raw[i], side=side)
        if hm.has_last.iloc[i] else 0 for i,e in enumerate(hm.event)
    ])
    bv = torch.tensor(bb, dtype=torch.long)
    zvalues = hm.zlast.to_numpy().copy()
    if value_mode == "centre":
        zvalues = np.array([
            (VOCAB["value_bin_centres"][e][bb[i]-1]-numeric_specs[(e,"value")]["median"])
            / numeric_specs[(e,"value")]["iqr"] if bb[i] else 0.
            for i,e in enumerate(hm.event)
        ])
    aa = hm.age_s.to_numpy() / (3600. if age_unit == "hours" else 1.)
    num = torch.zeros((len(hm),8))
    num[:,0] = last_has
    num[:,1] = torch.tensor(np.log1p(aa)/10, dtype=torch.float32)
    num[:,2] = torch.tensor(zvalues, dtype=torch.float32)
    num[:,3] = last_has
    reg = (wt("embed.code.weight")[codes_t] + wt("embed.bins.value.weight")[bv]
           + TF.linear(num, wt("last_numeric.weight"), wt("last_numeric.bias"))
           + wt("last_status.weight")[0])
    xin = queries_mc + reg[None,:,:]
    hidden = activation(TF.linear(xin, wt("attr_heads.value_bin.0.weight"),
                                  wt("attr_heads.value_bin.0.bias")), act)
    logits = TF.linear(hidden, wt("attr_heads.value_bin.2.weight"),
                       wt("attr_heads.value_bin.2.bias"))
    logits += wt("embed_transition.value_bin.weight")[codes_t*17+bv][None,:,:]
    pp = logits.softmax(-1)
    pm = pp.mean(0)
    mu = (pm*centres_t).sum(-1).numpy()
    nl = -pm[torch.arange(len(hm)),target_t].clamp_min(1e-12).log().numpy()
    register_results.append({
        "bin_side":side, "last_value":value_mode, "age_unit":age_unit,
        "scaled_mean_rmse":np.sqrt(np.mean(((mu-target_mu)/scale_mu)**2)),
        "nll_rmse":np.sqrt(np.mean((nl-hm.original_nll.to_numpy())**2)),
        "nll":nl.mean()
    })
    if side == "left" and value_mode == "raw" and age_unit == "seconds":
        baseline_probs_mc = pp.clone()
        baseline_mu = mu.copy()

print(pd.DataFrame(register_results).sort_values("scaled_mean_rmse").to_string(index=False))
p4 = baseline_probs_mc.reshape(16,4,len(hm),17).mean(1)
mu4 = (p4*centres_t[None,:,:]).sum(-1).numpy()
nl4 = -p4[:,:, :].gather(2, target_t[None,:,None].expand(16,-1,1)).squeeze(-1).clamp_min(1e-12).log().numpy()
print("\nExpected four-draw sampling RMSE:",
      np.sqrt(np.mean(np.var(mu4/scale_mu[None,:],axis=0))),
      "NLL:", np.sqrt(np.mean(np.var(nl4,axis=0))))
for is_boundary in [False,True]:
    mm = boundary == is_boundary
    print("last value on boundary:", is_boundary, "n:", mm.sum(),
          "scaled mean RMSE:", np.sqrt(np.mean(((baseline_mu[mm]-target_mu[mm])/scale_mu[mm])**2)))
print("\nBy sensor:")
for event in hm.event.unique():
    mm = hm.event.to_numpy() == event
    print(event, "n:", mm.sum(), "scaled RMSE:",
          np.sqrt(np.mean(((baseline_mu[mm]-target_mu[mm])/scale_mu[mm])**2)),
          "mean error:", float(np.mean(baseline_mu[mm]-target_mu[mm])))

print("\nResponse kernel parameters:")
for drug, result in response_kernels.items():
    if isinstance(result, dict):
        print(drug, {k: v for k,v in result.items() if np.isscalar(v)})
    else:
        print(drug, repr(result)[:500])
```

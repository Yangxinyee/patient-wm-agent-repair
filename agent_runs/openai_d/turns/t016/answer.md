The first reconstruction is only approximate: its mean predictions differ by about 3.8 units. It identifies the measurement query type, but not enough details to justify an edit yet. I’ll check bin-edge conventions, register flags, and latent sampling. A sharp reduction in error would confirm the missing convention; persistent error would mean we need another approach.

```python
print("Zero-bin and status embedding norms:",
      {n: torch.linalg.vector_norm(W0[n], dim=1).numpy()[:2].round(4).tolist()
       for n in ("embed.bins.value.weight", "embed.bins.rate.weight", "last_status.weight")})

lb_options = {}
for side in ("left", "right"):
    vals = []
    for r in hm.itertuples(index=False):
        sp = numeric_specs[(r.event, "value")]
        value = r.last_z * sp["iqr"] + sp["median"]
        vals.append(0 if not r.has_last else
                    np.searchsorted(sp["fine_bin_edges"], value, side=side) + 1)
    lb_options[side] = torch.tensor(vals, dtype=torch.long)

register_trials = []
best_reconstruction = None
for activation in ("gelu", "silu"):
    gapv = wmlp(torch.log1p(dt0) / 10., "gap", activation)
    ctx = out0 + hnorm + gapv
    zp = wmlp(ctx, "z_prior", activation)
    view = ctx + wlin(zp[:, :32], "z_head")
    q = wnorm(view + W0["type_emb.weight"][1], "query_norm")
    for flag_col in (3, 4, 5, 6, 7, None):
        for age_unit in ("hours", "seconds"):
            nf = np.zeros((len(hm), 8), dtype=np.float32)
            nf[:, 0] = hm.has_last
            nf[:, 1] = np.log1p(hm.age.to_numpy() /
                               (3600. if age_unit == "hours" else 1.)) / 10.
            nf[:, 2] = hm.last_z
            if flag_col is not None:
                nf[:, flag_col] = hm.has_last
            for side, lbs in lb_options.items():
                for status in (0, 1):
                    reg = (wlin(torch.tensor(nf), "last_numeric")
                           + W0["embed.bins.value.weight"][lbs]
                           + W0["embed.bins.rate.weight"][0]
                           + W0["last_status.weight"][status])
                    wc = q + W0["embed.code.weight"][cid] + reg
                    hidden = actfn(wlin(wc, "attr_heads.value_bin.0"), activation)
                    logits = (wlin(hidden, "attr_heads.value_bin.2")
                              + W0["embed_transition.value_bin.weight"][cid * 17 + lbs])
                    probs = logits.softmax(-1)
                    means = (probs * centres).sum(-1).numpy()
                    err = means - target_means
                    config = (activation, flag_col, age_unit, side, status)
                    rmse = float(np.sqrt(np.mean(err**2)))
                    register_trials.append((rmse, config, float(err.mean())))
                    if best_reconstruction is None or rmse < best_reconstruction["rmse"]:
                        best_reconstruction = dict(rmse=rmse, config=config,
                            means=means, q=q.clone(), wc=wc.clone(),
                            hidden=hidden.clone(), zp=zp.clone(), probs=probs.clone(),
                            nf=nf.copy(), lbs=lbs.clone(), ctx=ctx.clone())
register_trials.sort(key=lambda x: x[0])
print("Best register conventions:")
for r in register_trials[:10]:
    print(r)
print("Best error by event:")
err = best_reconstruction["means"] - target_means
for ev in hm.event.unique():
    m = hm.event.to_numpy() == ev
    print(ev, "RMSE", round(float(np.sqrt(np.mean(err[m]**2))), 4),
          "bias", round(float(err[m].mean()), 4))
print("No previous reading RMSE:",
      float(np.sqrt(np.mean(err[hm.has_last.to_numpy() == 0]**2))))
print("Prior second-half quantiles:",
      np.quantile(best_reconstruction["zp"][:, 32:].numpy(), [0, .1, .5, .9, 1]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 11\nMatched 4510 factual reading rows to their preceding saved states. Initial local head reconstruction identifies measurement type index 1 but mean RMSE is about 3.76, too large to assume exact reconstruction. Checking bin boundaries, register layout, status and latent details before editing.\n")
```

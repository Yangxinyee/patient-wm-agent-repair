Changing register flags and bin boundaries did not explain the discrepancy. The latent prior has nonzero sampling variance, and the forward tool averages only four draws, so some mismatch may be sampling noise. I’ll separate that from implementation error by averaging more draws and testing mixed activation conventions. Error consistent with four-draw noise would validate the reconstruction.

```python
mixed_trials = []
for ga in ("gelu", "silu"):
    gv = wmlp(torch.log1p(dt0) / 10., "gap", ga)
    ctx = out0 + hnorm + gv
    for za in ("gelu", "silu"):
        zp = wmlp(ctx, "z_prior", za)
        vmean = ctx + wlin(zp[:, :32], "z_head")
        q = wnorm(vmean + W0["type_emb.weight"][1], "query_norm")
        for ha in ("gelu", "silu"):
            for au in ("hours", "seconds"):
                nf = np.zeros((len(hm), 8), dtype=np.float32)
                nf[:, 0], nf[:, 2], nf[:, 3] = hm.has_last, hm.last_z, hm.has_last
                nf[:, 1] = np.log1p(hm.age.to_numpy() /
                                   (3600. if au == "hours" else 1.)) / 10.
                reg = wlin(torch.tensor(nf), "last_numeric") + W0["embed.bins.value.weight"][lb]
                wc = q + W0["embed.code.weight"][cid] + reg
                hh = actfn(wlin(wc, "attr_heads.value_bin.0"), ha)
                logits = wlin(hh, "attr_heads.value_bin.2") + W0["embed_transition.value_bin.weight"][cid * 17 + lb]
                mu = (logits.softmax(-1) * centres).sum(-1).numpy()
                mixed_trials.append(dict(rmse=float(np.sqrt(np.mean((mu-target_means)**2))),
                                         config=(ga, za, ha, au), ctx=ctx, zp=zp, reg=reg))
mixed_trials.sort(key=lambda r: r["rmse"])
print("Mixed activation candidates:", [(r["config"], round(r["rmse"], 4)) for r in mixed_trials[:6]])

torch.manual_seed(1764)
mc_head_checks = []
for candidate in mixed_trials[:3]:
    ga, za, ha, au = candidate["config"]
    ctx, zp, reg = candidate["ctx"], candidate["zp"], candidate["reg"]
    sums = torch.zeros(len(hm))
    sums2 = torch.zeros(len(hm))
    psum = torch.zeros(len(hm), 17)
    for chunk in range(8):
        zz = zp[None, :, :32] + torch.exp(zp[None, :, 32:].clamp(-5, 2)) * torch.randn(16, len(hm), 32)
        vv = ctx[None] + wlin(zz, "z_head")
        qq = wnorm(vv + W0["type_emb.weight"][1], "query_norm")
        ww = qq + W0["embed.code.weight"][cid][None] + reg[None]
        hh = actfn(wlin(ww, "attr_heads.value_bin.0"), ha)
        ll = wlin(hh, "attr_heads.value_bin.2") + W0["embed_transition.value_bin.weight"][cid * 17 + lb][None]
        pp = ll.softmax(-1)
        mm = (pp * centres[None]).sum(-1)
        sums += mm.sum(0)
        sums2 += (mm**2).sum(0)
        psum += pp.sum(0)
    mc_mu = (sums / 128).numpy()
    single_var = (sums2 / 128 - (sums / 128)**2).clamp_min(0).numpy()
    err = mc_mu - target_means
    check = dict(config=candidate["config"], mean=mc_mu, probs=(psum/128).numpy(),
                 rmse=float(np.sqrt(np.mean(err**2))),
                 expected_four_draw_rmse=float(np.sqrt(np.mean(single_var)/4)),
                 bias=float(err.mean()), candidate=candidate)
    mc_head_checks.append(check)
    print("MC check:", {k: check[k] for k in ("config", "rmse", "expected_four_draw_rmse", "bias")})
    for ev in hm.event.unique():
        m = hm.event.to_numpy() == ev
        print(" ", ev, "error RMS", round(float(np.sqrt(np.mean(err[m]**2))), 4),
              "4-draw noise RMS", round(float(np.sqrt(np.mean(single_var[m])/4)), 4))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 12\nRegister search supports [has_last, age, z_value, has_value, ...], measurement type 1 and zero status. Bin-edge convention has negligible effect here. Residual reconstruction error remains ~3.76; checking mixed MLP activations and latent Monte Carlo noise next. No edits.\n")
```

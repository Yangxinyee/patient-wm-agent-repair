The best reconstruction has mean RMSE 3.58 and NLL RMSE 0.406—too large to trust yet. It favors GELU and the expected last-value layout. I’ll check bin-boundary conventions and latent averaging next; these should either explain the discrepancies or reveal that another assumption is wrong.

```python
import itertools, time
import torch
import torch.nn.functional as TF

t0 = time.time()

def local_value_head(data, cfg, bin_side="left", zmode="prior_mean", nsamples=32,
                     return_features=False, seed=917):
    nrow = len(data)
    sx0 = torch.tensor(S["x"][data.state_index.to_numpy()], dtype=torch.float32)
    dt0 = torch.tensor(data.dt_s.to_numpy(), dtype=torch.float32)
    hf = sx0[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt0[:, None, None] * torch.exp(hw("core.flow.a"))[None, None, :]
    )
    base = sx0[:, :192] + head_norm(hf.mean(1), "core.out_norm")
    ga = cfg.get("gap_activation", cfg.get("activation", "gelu"))
    pa = cfg.get("prior_activation", cfg.get("activation", "gelu"))
    va = cfg.get("value_activation", cfg.get("activation", "gelu"))
    ctx = base + head_mlp(torch.log1p(dt0)[:, None] / 10, "gap", ga)
    pars = head_mlp(ctx, "z_prior", pa)
    if zmode == "prior_mc":
        generator = torch.Generator().manual_seed(seed)
        eps = torch.randn(nsamples, nrow, 32, generator=generator)
        zz = pars[None, :, :32] + eps * torch.exp(pars[None, :, 32:].clamp(-7, 3))
    elif zmode == "posterior_mean":
        zz = sx0[None, :, 960:992]
    elif zmode == "none":
        zz = torch.zeros(1, nrow, 32)
    else:
        zz = pars[None, :, :32]
    view = ctx[None] + TF.linear(zz, hw("z_head.weight"), hw("z_head.bias"))
    qv = head_norm(view + hw("type_emb.weight")[int(cfg.get("type_id", 1))], "query_norm")

    cid0 = torch.tensor(data.code_id.to_numpy(), dtype=torch.long)
    lb0 = np.zeros(nrow, np.int64)
    ob0 = np.zeros(nrow, np.int64)
    for event in data.event.unique():
        m = data.event.eq(event).to_numpy()
        spec = numeric_specs[event, "value"]
        vals = data.last_value.to_numpy()[m]
        good = np.isfinite(vals)
        bins = np.zeros(len(vals), np.int64)
        bins[good] = 1 + np.searchsorted(spec["fine_bin_edges"], vals[good], side=bin_side)
        lb0[m] = bins
        ob0[m] = 1 + np.searchsorted(
            spec["fine_bin_edges"], data.observed.to_numpy()[m], side=bin_side
        )
    lb0 = torch.tensor(lb0)
    hl0 = torch.tensor(data.has_last.to_numpy(), dtype=torch.float32)
    zl0 = torch.tensor(data.z_last.to_numpy(), dtype=torch.float32)
    ag0 = torch.tensor(data.last_age_s.to_numpy(), dtype=torch.float32)
    if cfg.get("age_mode", "seconds_div10") == "hours_div10":
        ag0 = ag0 / 3600
    ag0 = torch.log1p(ag0) / 10
    lastvec = torch.zeros(nrow, 8)
    lastvec[:, int(cfg.get("zcol", 2))] = zl0
    lastvec[:, int(cfg.get("acol", 1))] = ag0
    for j in cfg.get("flags", (0, 3)):
        lastvec[:, j] = hl0
    lastenc = TF.linear(lastvec, hw("last_numeric.weight"), hw("last_numeric.bias"))
    inp = (
        qv + hw("embed.code.weight")[cid0][None] + lastenc[None]
        + hw("embed.bins.value.weight")[lb0][None]
        + hw("embed.bins.rate.weight")[0]
        + hw("last_status.weight")[int(cfg.get("status_id", 0))]
    )
    pre = TF.linear(inp, hw("attr_heads.value_bin.0.weight"), hw("attr_heads.value_bin.0.bias"))
    hidden = act_fn(pre, va)
    logits = TF.linear(hidden, hw("attr_heads.value_bin.2.weight"), hw("attr_heads.value_bin.2.bias"))
    logits = logits + hw("embed_transition.value_bin.weight")[cid0 * 17 + lb0][None]
    prob_samples = torch.softmax(logits, -1)
    prob = prob_samples.mean(0)
    cc = torch.tensor(np.array([
        [0.0] + VOCAB["value_bin_centres"][e] for e in data.event
    ]), dtype=torch.float32)
    pm_samples = (prob_samples * cc[None]).sum(-1)
    means0 = (prob * cc).sum(-1).numpy()
    nll0 = -torch.log(prob[torch.arange(nrow), torch.tensor(ob0)].clamp_min(1e-12)).numpy()
    out0 = dict(mean=means0, nll=nll0, prob=prob.numpy(),
                prior_logsd=pars[:, 32:].numpy(),
                sampling_mean_sd=pm_samples.std(0, unbiased=False).numpy())
    if return_features:
        out0.update(hidden=hidden.mean(0).numpy(), input=inp.mean(0).numpy(),
                    pre=pre.mean(0).numpy(), q=qv.mean(0).numpy())
    return out0

refine_rows = []
for ga, pa, va, side, zm in itertools.product(
    ("gelu", "silu"), ("gelu", "silu"), ("gelu", "silu"),
    ("left", "right"), ("prior_mean", "prior_mc")
):
    cfg0 = dict(head_cfg, gap_activation=ga, prior_activation=pa, value_activation=va)
    ans0 = local_value_head(hd, cfg0, bin_side=side, zmode=zm, nsamples=32)
    mr = np.sqrt(np.mean((ans0["mean"] - hd.predicted_mean.to_numpy()) ** 2))
    nr = np.sqrt(np.mean((ans0["nll"] - hd.nll.to_numpy()) ** 2))
    refine_rows.append(dict(gap_activation=ga, prior_activation=pa, value_activation=va,
                            bin_side=side, zmode=zm, mean_rmse=mr, nll_rmse=nr,
                            mean_bias=np.mean(ans0["mean"] - hd.predicted_mean.to_numpy()),
                            score=mr + 3 * nr))
head_refinement = pd.DataFrame(refine_rows).sort_values("score")
best_ref = head_refinement.iloc[0].to_dict()
head_cfg.update({k: best_ref[k] for k in ("gap_activation", "prior_activation", "value_activation")})
head_bin_side = best_ref["bin_side"]
print("=== REFINED RECONSTRUCTIONS ===")
print(head_refinement.head(12).to_string(index=False))

head_check = local_value_head(hd, head_cfg, bin_side=head_bin_side,
                              zmode="prior_mc", nsamples=256, return_features=True)
check_rows = []
for event in hd.event.unique():
    m = hd.event.eq(event).to_numpy()
    dm = head_check["mean"][m] - hd.predicted_mean.to_numpy()[m]
    dn = head_check["nll"][m] - hd.nll.to_numpy()[m]
    check_rows.append(dict(
        event=event, n=int(m.sum()), mean_rmse=float(np.sqrt(np.mean(dm**2))),
        mean_bias=float(dm.mean()), nll_rmse=float(np.sqrt(np.mean(dn**2))),
        nll_bias=float(dn.mean()),
        predicted_four_draw_mean_rmse=float(np.sqrt(
            np.mean(head_check["sampling_mean_sd"][m]**2) / 4
        ))
    ))
print("\n=== LATENT-SAMPLING CHECK ===")
print(pd.DataFrame(check_rows).to_string(index=False))
print("prior log-sd quantiles:", np.quantile(head_check["prior_logsd"], [0, .1, .5, .9, 1]))
saved_out = torch.tensor(S["x"][:, :192])
norm_out = head_norm(torch.tensor(S["x"][:, 192:960]).reshape(-1, 4, 192).mean(1), "core.out_norm")
print("saved-out vs normalized-h RMSE:", float(torch.sqrt(torch.mean((saved_out - norm_out)**2))))
print("seconds:", round(time.time() - t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 17: head reconstruction audit\n"
            + head_refinement.head(5).to_json(orient="records") + "\n"
            + pd.DataFrame(check_rows).to_json(orient="records") + "\n")
```
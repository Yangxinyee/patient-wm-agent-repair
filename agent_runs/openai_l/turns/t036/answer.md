Testing mixed activations and register layouts reduced the reconstruction error to about 1.76. The existing bin-boundary convention was correct, so it was not the main problem. I’ll check the leading layouts on all 1,000 readings with latent averaging. Sampling-sized errors without systematic event-specific bias would make a state-based correction more credible.

```python
r32_pre = torch.tensor(r29_export["x"][pre22])
r32_dt = torch.tensor(dt22, dtype=torch.float32)
r32_base = r32_pre[:, :192] + r30_norm(
    (r32_pre[:, 192:960].reshape(-1, 4, 192) *
     torch.exp(-r32_dt[:, None, None]*torch.exp(r30_w("core.flow.a"))[None, None, :])).mean(1),
    "core.out_norm")
r32_specs = []
for r32_result in r31_best:
    r32_spec = r32_result[1:]
    if r32_spec not in r32_specs:
        r32_specs.append(r32_spec)
    if len(r32_specs) == 8:
        break
r32_specs += [
    (("gelu", "seconds", .1, .1, "gelu", "gelu", 1),
     "left", "seconds", 2, 1, (0, 3)),
    (("gelu", "seconds", .1, .1, "gelu", "gelu", 1),
     "left", "seconds", 6, 3, (0,))
]

def r32_features(spec):
    cfg, side, age_unit, zp, ap, flags = spec
    ga, unit, ps, vs, pa, ha, ti = cfg
    ld = torch.log1p(r32_dt/(3600 if unit == "hours" else 1))
    raw = r30_mlp(r32_base+r30_mlp(
        (ld*ps)[:, None], "gap", ga), "z_prior", pa)
    viewbase = r32_base+r30_mlp((ld*vs)[:, None], "gap", ga)
    nf = torch.zeros(len(ds22), 8)
    nf[:, zp] = torch.tensor(last_z22)
    nf[:, ap] = torch.tensor(
        np.log1p(age_hours22*(3600 if age_unit == "seconds" else 1))/10,
        dtype=torch.float32)
    for flag in flags:
        nf[:, flag] = torch.tensor(has22)
    lb = r31_bins[side]
    const = (r30_w("embed.code.weight")[code22]
             +r30_lin(nf, "last_numeric")
             +r30_w("embed.bins.value.weight")[lb]
             +r30_w("embed.bins.rate.weight")[0]
             +r30_w("last_status.weight")[0])
    trans = r30_w("embed_transition.value_bin.weight")[code22*17+lb]
    return raw, viewbase, const, trans

r32_eps = torch.randn(64, len(ds22), 32,
                     generator=torch.Generator().manual_seed(3232))
r32_results = []
r32_details = {}
for r32_spec in r32_specs:
    r32_raw, r32_vb, r32_const, r32_trans = r32_features(r32_spec)
    r32_cfg = r32_spec[0]
    r32_ha, r32_ti = r32_cfg[-2:]
    r32_z = r32_raw[None, :, :32]+r32_eps*torch.exp(
        r32_raw[None, :, 32:].clamp(-7, 3))
    r32_q = r30_norm(
        r32_vb[None, :, :]+r30_lin(r32_z, "z_head")
        +r30_w("type_emb.weight")[r32_ti], "query_norm")
    r32_hidden = r30_act(r30_lin(
        r32_q+r32_const[None, :, :], "attr_heads.value_bin.0"), r32_ha)
    r32_p = (r30_lin(r32_hidden, "attr_heads.value_bin.2")
             +r32_trans[None, :, :]).softmax(-1)
    r32_m = (r32_p[:, :, 1:]*centres22[None, :, :]).sum(-1)
    r32_pbar = r32_p.mean(0)
    r32_error = r32_m.mean(0).numpy()-reference_mean22
    r32_variance = r32_m.var(0, unbiased=False).numpy()/4
    r32_nlls = {}
    for r32_side in ("left", "right"):
        r32_nll = -torch.log(r32_pbar.gather(
            1, r31_obsbins[r32_side][:, None]).squeeze(1).clamp_min(1e-12))
        r32_nlls[r32_side] = float(
            np.abs(r32_nll.numpy()-reference_nll22).mean())
    r32_result = (
        float(np.sqrt(np.mean(r32_error**2))),
        float(np.sqrt(np.mean(r32_variance))),
        r32_nlls, r32_spec)
    r32_results.append(r32_result)
    r32_details[r32_spec] = {
        "error": r32_error, "sampling_variance": r32_variance,
        "hidden_mean": r32_hidden.mean(0), "prob_mean": r32_pbar
    }

print("=== FULL-SAMPLE MC CHECKS ===")
for r32_result in sorted(r32_results, key=lambda x: x[0]):
    print(r32_result)

r32_best = min(r32_results, key=lambda x: x[0])[-1]
print("\n=== BEST-SPEC ERRORS BY EVENT / REGISTER PRESENCE ===")
for r32_event in ds22.event.unique():
    for r32_presence in (0, 1):
        r32_mask = ((ds22.event.to_numpy() == r32_event) &
                    (has22 == r32_presence))
        if r32_mask.any():
            r32_err = r32_details[r32_best]["error"][r32_mask]
            r32_var = r32_details[r32_best]["sampling_variance"][r32_mask]
            print(r32_event, "has_last", r32_presence, "n", int(r32_mask.sum()),
                  "bias", round(float(r32_err.mean()), 3),
                  "RMSE", round(float(np.sqrt(np.mean(r32_err**2))), 3),
                  "sampling_noise", round(float(np.sqrt(np.mean(r32_var))), 3))

print("\n=== REGISTER AND BIN WEIGHT MAGNITUDES ===")
print("last_numeric column norms:",
      r30_w("last_numeric.weight").norm(dim=0).tolist())
for r32_name in ("embed.bins.value.weight", "embed.bins.rate.weight",
                 "last_status.weight", "embed_transition.value_bin.weight"):
    r32_t = r30_w(r32_name)
    print(r32_name, "norm", float(r32_t.norm()),
          "first-row norm", float(r32_t[0].norm()),
          "maximum", float(r32_t.abs().max()))

with open("/work/notes.md", "a") as r32_note:
    r32_note.write(
        "\n## Run 32\n"
        "Register search reduced deterministic RMSE to 1.765; retained binning is "
        "left-boundary, not the source of mismatch. Full-sample latent MC checks "
        "and event/presence error diagnostics retained in r32_results/r32_details. "
        "Best spec retained as r32_best. No edits.\n"
    )
```
Separating the two gap terms did not explain the error. The remaining timing question is whether the register’s age is measured at the saved state or at the next reading. I’ll test both, plus the type index and numeric-field order. Error close to the Monte Carlo floor would validate the reconstruction; otherwise I’ll stop treating it as exact.

```python
rc_age_inputs = {
    "forecast_seconds": torch.log1p(rc_age) / 10,
    "forecast_hours": torch.log1p(rc_age / 3600) / 10,
    "state_seconds": torch.log1p((rc_age - rc_dt[:, 0]).clamp_min(0)) / 10,
    "state_hours": torch.log1p((rc_age - rc_dt[:, 0]).clamp_min(0) / 3600) / 10,
    "zero": torch.zeros_like(rc_age),
}
rc_layout_search = []
for prior_act, head_act in itertools.product(("silu", "gelu"), repeat=2):
    ctx1 = rc_base + rc_gaps[("seconds_div10", "silu")]
    zp1 = rc_mlp(ctx1, "z_prior", prior_act)
    vv1 = ctx1 + rc_lin(zp1[:, :32], "z_head")
    for type_idx in range(15):
        qq1 = rc_norm(vv1 + W0["type_emb.weight"][type_idx], "query_norm")
        for age_cfg, av1 in rc_age_inputs.items():
            ff1 = torch.zeros(rc_n, 8)
            ff1[:, 0], ff1[:, 1], ff1[:, 2], ff1[:, 3] = rc_has, av1, rc_z, rc_has
            reg1 = (rc_lin(ff1, "last_numeric") + W0["embed.bins.value.weight"][rc_bins]
                    + W0["embed.bins.rate.weight"][0])
            wc1 = qq1 + W0["embed.code.weight"][rc_codes] + reg1
            pp1 = (rc_mlp(wc1, "attr_heads.value_bin", head_act)
                   + W0["embed_transition.value_bin.weight"][rc_codes * 17 + rc_bins]).softmax(-1)
            mm1 = (pp1 * rc_centres).sum(-1).numpy()
            rc_layout_search.append((
                float(np.sqrt(np.mean((mm1 - target_means)**2))),
                (prior_act, head_act, type_idx, age_cfg, 2, 3)
            ))

rc_layout_search.sort(key=lambda x: x[0])
print("Age and type search:", [(round(x, 4), cfg) for x, cfg in rc_layout_search[:8]])

rc_permutation_search = []
for _, cfg in rc_layout_search[:4]:
    prior_act, head_act, type_idx, age_cfg, _, _ = cfg
    zp1 = rc_mlp(rc_base + rc_gaps[("seconds_div10", "silu")], "z_prior", prior_act)
    vv1 = rc_base + rc_gaps[("seconds_div10", "silu")] + rc_lin(zp1[:, :32], "z_head")
    qq1 = rc_norm(vv1 + W0["type_emb.weight"][type_idx], "query_norm")
    for z_col, flag_col in itertools.permutations(range(2, 8), 2):
        ff1 = torch.zeros(rc_n, 8)
        ff1[:, 0], ff1[:, 1] = rc_has, rc_age_inputs[age_cfg]
        ff1[:, z_col], ff1[:, flag_col] = rc_z, rc_has
        reg1 = (rc_lin(ff1, "last_numeric") + W0["embed.bins.value.weight"][rc_bins]
                + W0["embed.bins.rate.weight"][0])
        wc1 = qq1 + W0["embed.code.weight"][rc_codes] + reg1
        pp1 = (rc_mlp(wc1, "attr_heads.value_bin", head_act)
               + W0["embed_transition.value_bin.weight"][rc_codes * 17 + rc_bins]).softmax(-1)
        mm1 = (pp1 * rc_centres).sum(-1).numpy()
        rc_permutation_search.append((
            float(np.sqrt(np.mean((mm1 - target_means)**2))),
            (prior_act, head_act, type_idx, age_cfg, z_col, flag_col)
        ))

rc_permutation_search.sort(key=lambda x: x[0])
print("Numeric layout search:",
      [(round(x, 4), cfg) for x, cfg in rc_permutation_search[:8]])

best_layouts = sorted(rc_layout_search[:4] + rc_permutation_search[:4], key=lambda x: x[0])
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 16–19\nState alignment and out=norm(mean h) verified. "
            "Status embedding for readings is zero. Separate gap conventions did not "
            "resolve local head error (~3.36 versus ~0.93 four-draw noise RMS). "
            "Testing register age at saved-state time versus forecast time, type index, "
            "and numeric layout. No edits.\n")
```
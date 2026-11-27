State packing is correct, but the legacy transition cache disagrees with freshly indexed weights. The current head reconstruction is still unreliable. I’ll test prior timing, forecast timing, and register layouts independently; a sharp improvement across all reading types would identify a usable configuration.

```python
print("=== HELPER / COMPONENT CONSISTENCY ===")
probe67 = torch.randn(7, 192, generator=torch.Generator().manual_seed(67))
print("rc_norm versus fresh:",
      float((rc_norm(probe67, "query_norm") -
             rep_norm(probe67, "query_norm")).abs().max()))
print("rc_act GELU:", rc_act(torch.tensor([-2., -1., 0., 1., 2.]), "gelu").tolist())
for key67, val67 in rh_bases.items():
    canonical67 = {
        "flow": rep_flowbase66[idx66],
        "no_flow": 2 * rep_out66[idx66],
        "out_only": rep_out66[idx66],
    }.get(key67)
    if canonical67 is not None:
        print("base", key67, "max difference",
              float((val67 - canonical67).abs().max()))
print("rh_lb versus metadata differences:",
      int((rh_lb.numpy() != rh_meta.last_bin.to_numpy()).sum()))
print("rh_trans versus fresh same-index max difference:",
      float((rh_trans - rep_W["embed_transition.value_bin.weight"][
          rh_codes * 17 + rh_lb]).abs().max()))
print("Padding/status embedding norms:",
      float(rep_W["embed.bins.rate.weight"][0].norm()),
      float(rep_W["last_status.weight"][0].norm()))

rep_freshtrans67 = rep_W["embed_transition.value_bin.weight"][rh_codes * 17 + rh_lb]
rep_registers67 = {
    (zp, fp, age): rh_register(zp, fp, age).clone()
    for zp, fp, age in [
        (6, 3, "forecast_hours"), (2, 3, "forecast_hours"),
        (2, 5, "forecast_seconds"), (6, 3, "forecast_seconds")
    ]
}
rep_regs67 = {
    key: rep_fix66 + rep_linear(nf, "last_numeric")
    for key, nf in rep_registers67.items()
}
rep_eps67 = torch.randn(8, len(rh_meta), 32,
                       generator=torch.Generator().manual_seed(6701))
rep_results67 = []
for flow67, fb67 in [
    ("flow", rep_flowbase66[idx66]),
    ("no_flow", 2 * rep_out66[idx66])
]:
    for prior_gap67 in ["hours_div10", "seconds_div10", "hours"]:
        prior67 = rep_mlp(fb67 + rep_gap66[prior_gap67][idx66], "z_prior")
        mu67, ls67 = prior67[:, :32], prior67[:, 32:].clamp(-8, 3)
        for view_gap67 in ["hours_div10", "seconds_div10", "hours"]:
            vb67 = fb67 + rep_gap66[view_gap67][idx66]
            sums67 = {key: torch.zeros(len(rh_meta), 17)
                      for key in rep_regs67}
            for ee67 in rep_eps67:
                q67 = rep_norm(
                    vb67 + rep_linear(mu67 + ls67.exp() * ee67, "z_head")
                    + rep_W["type_emb.weight"][1], "query_norm")
                for reg67, fixed67 in rep_regs67.items():
                    hh67 = torch.nn.functional.gelu(rep_linear(
                        q67 + fixed67, "attr_heads.value_bin.0"))
                    sums67[reg67] += torch.softmax(
                        rep_linear(hh67, "attr_heads.value_bin.2")
                        + rep_freshtrans67, -1) / len(rep_eps67)
            for reg67, probs67 in sums67.items():
                means67 = (probs67 * rh_centres).sum(-1)
                err67 = means67 - rh_tool
                score67 = float((err67 / rh_iqr).square().mean().sqrt())
                rep_results67.append((
                    score67, float(err67.square().mean().sqrt()),
                    (flow67, prior_gap67, view_gap67, reg67)))
print("\n=== BEST FRESH CONFIGURATIONS ===")
for r67 in sorted(rep_results67)[:8]:
    print(r67)
print("Current edits:", edited())
```
The saved summary matches the normalized slots almost exactly, so state layout is not the problem. I’ll separate the prior-context, time-gap and register conventions next. A large reduction in error from one of those checks would identify the mismatch; otherwise, I’ll revisit how readings were matched to states.

```python
print("=== BEST RECONSTRUCTIONS ===")
print(reconstruction_table.sort_values("scaled_mean_rmse").head(12).to_string(index=False))

print("\n=== REGISTER TESTS ===")
print(pd.DataFrame(register_results).to_string(index=False))

print("\n=== LATENT INTEGRATION TESTS ===")
print(integration_table.sort_values("scaled_mean_rmse").head(8).to_string(index=False))

def tensor_rmse(x, y):
    return float(torch.sqrt(torch.mean((x - y)**2)))

flow_check = hh * torch.exp(-dt[:, None, None] * torch.exp(wt("core.flow.a"))[None, None, :])
summary_check = ln(flow_check.mean(1), "core.out_norm")
print("\n=== DEFAULT FORMULA CHECKS ===")
print("hflow vs seconds-based flow:", tensor_rmse(hflow, flow_check))
print("flowed_summary vs seconds-based summary:", tensor_rmse(flowed_summary, summary_check))
for description, feature in [
    ("log1p_seconds/10", torch.log1p(dt)/10),
    ("log_seconds/10", torch.log(dt.clamp_min(1))/10),
    ("log1p_seconds", torch.log1p(dt)),
    ("log1p_hours/10", torch.log1p(dt/3600)/10),
]:
    gap_check = mlp(feature[:, None], "gap", "gelu")
    print("gap_default vs", description, tensor_rmse(gap_default, gap_check))

print("ctx_default vs out + flowed_summary + gap:",
      tensor_rmse(ctx_default, out + summary_check + gap_default))
print("ctx_default vs 2*out + gap:",
      tensor_rmse(ctx_default, 2*out + gap_default))
print("view_default vs ctx_default + z_head(z_default):",
      tensor_rmse(view_default, ctx_default +
                  TF.linear(z_default, wt("z_head.weight"), wt("z_head.bias"))))
print("z_default vs saved posterior:", tensor_rmse(z_default, sx[:, 960:992]))

register_check = (
    wt("embed.code.weight")[codes_t]
    + TF.linear(nums, wt("last_numeric.weight"), wt("last_numeric.bias"))
    + wt("embed.bins.value.weight")[lastbins_t]
    + wt("embed.bins.rate.weight")[torch.zeros_like(lastbins_t)]
    + wt("last_status.weight")[0]
)
print("reg_default vs interleaved register:", tensor_rmse(reg_default, register_check))
print("vocab event kinds:", VOCAB["event_kinds"])
print("vocab atom kinds:", VOCAB["atom_kinds"])

print("\n=== CURRENT RECONSTRUCTION ERROR BY EVENT ===")
best_label = variant_table.sort_values("scaled_mean_rmse").iloc[0]["variant"]
best_inputs = variant_features[best_label]
print("best variant:", best_label, "feature shape:", tuple(best_inputs.shape))
for event in MEAS:
    sel = hm.event.to_numpy() == event
    print(event, "n:", int(sel.sum()),
          "mean observed NLL:", float(hm.loc[sel, "original_nll"].mean()))
```
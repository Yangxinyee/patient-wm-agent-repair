Freshly loaded weights reproduce the same error, ruling out a stale weight cache. I’ll now compare the two state/register caches and rebuild the view directly from saved states. Matching inputs and reconstruction error consistent with four-draw noise are prerequisites for trusting a head edit.

```python
print("=== PREVIOUS BEST CONFIGURATION ===")
print("rh_chosen:", rh_chosen)
print("best_reconstruction:", str(best_reconstruction)[:1200])

idx66 = pd.MultiIndex.from_frame(wr_meta[["stay_id", "hour", "event"]]).get_indexer(
    pd.MultiIndex.from_frame(rh_meta[["stay_id", "hour", "event"]]))
print("\n=== CACHE ALIGNMENT ===")
print("Unmatched rows:", int((idx66 < 0).sum()))
if (idx66 >= 0).all():
    print("State max difference:",
          float((rh_x - wr_x[idx66]).abs().max()))
    print("Centres max difference:",
          float((rh_centres - wr_centres[idx66]).abs().max()))
    print("Transition max difference:",
          float((rh_trans - rep_trans65[idx66]).abs().max()))
    print("Last z max difference:",
          float(np.max(np.abs(rh_meta.last_z.to_numpy()
                              - wr_meta.last_z.to_numpy()[idx66]))))
print("rh_last versus metadata:",
      float(np.max(np.abs(rh_last.numpy() - rh_meta.last_z.to_numpy()))))
print("wr_dt range:", wr_dt.min().item(), wr_dt.max().item())
print("rh_dt range:", rh_dt.min().item(), rh_dt.max().item())

rep_out66 = wr_x[:, :192]
rep_h66 = wr_x[:, 192:960].reshape(-1, 4, 192)
rep_dt66 = torch.tensor(wr_meta.dt.to_numpy(), dtype=torch.float32).reshape(-1, 1)
rep_flow66 = rep_h66 * torch.exp(
    -rep_dt66 * rep_W["core.flow.a"].exp()).unsqueeze(1)
rep_flowbase66 = rep_out66 + rep_norm(rep_flow66.mean(1), "core.out_norm")
print("out packing identity error:",
      float((rep_norm(rep_h66.mean(1), "core.out_norm") - rep_out66).abs().max()))
print("\n=== CACHED BASE IDENTIFICATION ===")
rep_gap66 = {}
for key66, x66 in {
    "hours_div10": torch.log1p(rep_dt66 / 3600) / 10,
    "seconds_div10": torch.log1p(rep_dt66) / 10,
    "hours": torch.log1p(rep_dt66 / 3600),
    "seconds": torch.log1p(rep_dt66),
}.items():
    rep_gap66[key66] = rep_mlp(x66, "gap")
    print(key66, "cached-base RMS",
          float((wr_base - rep_flowbase66 - rep_gap66[key66])
                .square().mean().sqrt()))

print("\n=== FRESH HEAD ON PREVIOUS 1000-ROW CACHE ===")
rep_nf66 = rh_register(6, 3, "forecast_hours").clone()
rep_fix66 = (rep_W["embed.code.weight"][rh_codes]
             + rep_W["embed.bins.value.weight"][rh_lb]
             + rep_W["embed.bins.rate.weight"][0]
             + rep_W["last_status.weight"][0])
print("Fixed-input max difference:", float((rep_fix66 - rh_fixed).abs().max()))
rep_base66 = rep_flowbase66[idx66] + rep_gap66["hours_div10"][idx66]
rep_prior66 = rep_mlp(rep_base66, "z_prior")
rep_mu66, rep_ls66 = rep_prior66[:, :32], rep_prior66[:, 32:].clamp(-8, 3)
rep_p66 = torch.zeros(len(rh_meta), 17)
rep_rng66 = torch.Generator().manual_seed(6601)
for draw66 in range(32):
    z66 = rep_mu66 + rep_ls66.exp() * torch.randn(
        len(rh_meta), 32, generator=rep_rng66)
    q66 = rep_norm(rep_base66 + rep_linear(z66, "z_head")
                  + rep_W["type_emb.weight"][1], "query_norm")
    h66 = torch.nn.functional.gelu(rep_linear(
        q66 + rep_fix66 + rep_linear(rep_nf66, "last_numeric"),
        "attr_heads.value_bin.0"))
    p66 = torch.softmax(
        rep_linear(h66, "attr_heads.value_bin.2") + rh_trans, dim=-1)
    rep_p66 += p66 / 32
rep_mean66 = (rep_p66 * rh_centres).sum(-1).numpy()
for ev66 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    m66 = rh_meta.event.to_numpy() == ev66
    print(ev66, "RMSE",
          round(float(np.sqrt(np.mean(
              (rep_mean66[m66] - rh_meta.tool_mean.to_numpy()[m66])**2))), 4))
print("Current edits:", edited())
```
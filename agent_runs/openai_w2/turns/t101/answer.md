No unit or categorical variant removes the mismatch, so those searches do not justify changing the mapping. I’ll test numeric-value and bin conventions next, alongside prior-versus-posterior alignment. A successful convention should reduce errors by orders of magnitude, not merely improve them slightly.

```python
print("=== VALUE CENTRE FORMAT ===")
centres87 = VOCAB["value_bin_centres"]
print(type(centres87).__name__)
if isinstance(centres87, dict):
    print("keys:", list(centres87)[:8])
else:
    print("length:", len(centres87), "first:", repr(centres87[0])[:250])

zraw87, bins_left87, bins_right87, rawvals87 = [], [], [], []
for row87 in flat_rows86:
    spec87 = numeric_specs[(row87["event"], "value")]
    val87 = float(row87["value"])
    rawvals87.append(val87)
    zraw87.append(np.clip((val87 - spec87["median"]) / spec87["iqr"], -10, 10))
    bins_left87.append(1 + np.searchsorted(spec87["fine_bin_edges"], val87, side="left"))
    bins_right87.append(1 + np.searchsorted(spec87["fine_bin_edges"], val87, side="right"))
zraw87 = torch.tensor(zraw87, dtype=torch.float32)
rawvals87 = torch.tensor(rawvals87, dtype=torch.float32)
bins_left87 = torch.tensor(bins_left87)
bins_right87 = torch.tensor(bins_right87)
binweight87 = Wrepair["embed.bins.value.weight"]
numweight87 = Wrepair["embed.numeric.weight"]

def numeric_atoms87(z, bins, flag=1):
    return (atoms86 + (z-zraw87)[:, None] * numweight87[:, 0]
            + (flag-1) * numweight87[:, 1]
            + binweight87[bins] - binweight87[bins_left87])

variants87 = [
    ("documented", zraw87, bins_left87, 1),
    ("right_edge_bins", zraw87, bins_right87, 1),
    ("numeric_zero", torch.zeros_like(zraw87), bins_left87, 1),
    ("numeric_absent", torch.zeros_like(zraw87), bins_left87, 0),
    ("bins_absent", zraw87, torch.zeros_like(bins_left87), 1),
    ("all_value_absent", torch.zeros_like(zraw87), torch.zeros_like(bins_left87), 0),
    ("raw_div_100", rawvals87/100, bins_left87, 1),
]
for scale87 in [0.25, 0.5, 2.0, 4.0]:
    variants87.append((f"numeric_scale_{scale87}", zraw87*scale87, bins_left87, 1))

variant_results87 = []
for name87, zz87, bb87, flag87 in variants87:
    mean87, max87 = posterior_error86(numeric_atoms87(zz87, bb87, flag87))
    variant_results87.append({"variant": name87, "mean_rmse": mean87, "max_rmse": max87})
print("\n=== NUMERIC VARIANTS ===")
print(pd.DataFrame(variant_results87).sort_values("mean_rmse").to_string(index=False))

print("\n=== LATENT ALIGNMENT CHECKS ===")
for name87, pred87 in [
    ("prior(ctx)", rmlp(ctx86, "z_prior")[:, :32]),
    ("posterior(empty observation)",
     rmlp(torch.cat([ctx86, empty_e_repair.expand(len(idx86), -1), u86], dim=1),
           "z_post")[:, :32])
]:
    err87 = ((pred87-truth86).square().mean(dim=1)).sqrt()
    print(name87, "mean RMSE:", float(err87.mean()), "max:", float(err87.max()))

print("\n=== WORST MATCHED OBSERVATION BUNDLES ===")
for pos87 in np.argsort(posterior_error_v2[idx86])[-5:]:
    print(meta85.iloc[idx86[pos87]].to_dict(),
          "RMSE:", float(posterior_error_v2[idx86[pos87]]),
          "readings:", [(r["event"], r["value"]) for r in sets86[pos87]])
print("Current edits:", edited())
```
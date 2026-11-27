The documented numeric convention still fits best, but it does not explain the errors. I’ll check encoder activation and pooling variants, and verify that our reference readings still match the CSV. This will test both the reconstruction assumptions and the possibility of an accidentally altered working copy.

```python
print("=== REFERENCE RECORD INTEGRITY ===")
check_ids88 = set(meta85.iloc[idx86]["stay_id"].astype(int))
pieces88 = []
for chunk88 in pd.read_csv("/data/train.csv", chunksize=200000):
    selected88 = chunk88[chunk88.stay_id.isin(check_ids88)]
    if len(selected88):
        pieces88.append(selected88)
raw_check88 = pd.concat(pieces88, ignore_index=True)
compare88 = reference_records_repair.merge(
    raw_check88, on=["stay_id", "hour", "event"], suffixes=("_ref", "_csv"))
for col88 in ["value", "rate", "dose"]:
    a88 = compare88[col88 + "_ref"].to_numpy()
    b88 = compare88[col88 + "_csv"].to_numpy()
    print(col88, "matched rows:", len(a88),
          "differences:", int(np.count_nonzero(
              ~np.isclose(a88, b88, equal_nan=True))))

print("\n=== ACTIVATION AND POOLING SEARCH ===")
activation88 = {"gelu": tF.gelu, "relu": tF.relu, "silu": tF.silu}
encoder_results88 = []
for act1_name88, act1_88 in activation88.items():
    atom_hidden88 = rlinear(
        act1_88(rlinear(atoms86, "encode.mlp1.0")), "encode.mlp1.2")
    atom_sum88 = torch.zeros(len(idx86), 192)
    atom_sum88.index_add_(0, flat_owner86, atom_hidden88)
    for pool88 in ["sqrt", "mean", "sum"]:
        denominator88 = (counts86.sqrt() if pool88 == "sqrt"
                         else counts86 if pool88 == "mean"
                         else torch.ones_like(counts86))
        pooled88 = atom_sum88 / denominator88[:, None]
        pooled88 += rlinear(torch.log1p(counts86)[:, None], "encode.size")
        for act2_name88, act2_88 in activation88.items():
            e88 = rnorm(rlinear(
                act2_88(rlinear(pooled88, "encode.mlp2.0")),
                "encode.mlp2.2"), "encode.norm")
            z88 = rmlp(torch.cat([ctx86, e88, u86], dim=1), "z_post")[:, :32]
            err88 = ((z88-truth86).square().mean(dim=1)).sqrt()
            encoder_results88.append({
                "mlp1": act1_name88, "pool": pool88, "mlp2": act2_name88,
                "mean_rmse": float(err88.mean()), "max_rmse": float(err88.max())})
print(pd.DataFrame(encoder_results88).sort_values("mean_rmse").head(10).to_string(index=False))

print("\n=== NUMERIC INPUT FROM BIN CENTRES ===")
zc88 = []
for row88, bin88 in zip(flat_rows86, bins_left87.tolist()):
    cs88 = centres87[row88["event"]]
    centre88 = cs88[bin88] if isinstance(cs88, list) else cs88[str(bin88)]
    spec88 = numeric_specs[(row88["event"], "value")]
    zc88.append(np.clip((centre88-spec88["median"])/spec88["iqr"], -10, 10))
zc88 = torch.tensor(zc88, dtype=torch.float32)
print("bin-centre numeric:", posterior_error86(
    numeric_atoms87(zc88, bins_left87)))
print("Current edits:", edited())
```
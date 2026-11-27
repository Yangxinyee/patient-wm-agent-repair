Absent actions were already zero, so that hypothesis is refuted. I’ll now test the observation-only unit mappings and categorical offsets against saved posteriors, using bundles whose atoms can be matched exactly. A near-zero error would identify the missing convention; otherwise we’ll move beyond categorical encoding.

```python
from itertools import product

lookup86 = {}
for key86, frame86 in reference_records_repair.groupby(["stay_id", "hour"], sort=False):
    lookup86[(int(key86[0]), int(round(float(key86[1]) * 3600)))] = frame86.to_dict("records")

eligible86 = []
sets86 = []
for j86, row86 in meta85.iterrows():
    if has_action85[j86] or abs(row86.hour - round(row86.hour)) > 1e-5:
        continue
    rows86 = lookup86.get((int(row86.stay_id), int(round(row86.hour * 3600))), [])
    if rows86 and all(r["event"] in measurement_events for r in rows86):
        if len(rows86) == int(row86.n_atoms):
            eligible86.append(j86)
            sets86.append(rows86)
    if len(eligible86) >= 240:
        break

idx86 = np.asarray(eligible86)
flat_rows86 = [r for rows86 in sets86 for r in rows86]
flat_owner86 = torch.tensor([i for i, rows86 in enumerate(sets86) for r in rows86])
counts86 = torch.tensor([len(rows86) for rows86 in sets86], dtype=torch.float32)
atoms86 = torch.stack([embed_record_atom_v2(r) for r in flat_rows86])
ctx86 = posterior_ctx_repair[idx86]
truth86 = truth85[idx86]
u86 = torch.zeros_like(ctx86)
glucose86 = torch.tensor([r["event"].startswith("glucose") for r in flat_rows86])
old_unit86 = torch.tensor([unit_ids_repair[r["event"]] for r in flat_rows86])

def encode_flat86(atoms):
    pooled = torch.zeros(len(idx86), 192)
    pooled.index_add_(0, flat_owner86, rmlp(atoms, "encode.mlp1"))
    pooled = pooled / counts86.sqrt()[:, None]
    pooled += rlinear(torch.log1p(counts86)[:, None], "encode.size")
    return rnorm(rmlp(pooled, "encode.mlp2"), "encode.norm")

def posterior_error86(atoms):
    e = encode_flat86(atoms)
    z = rmlp(torch.cat([ctx86, e, u86], dim=1), "z_post")[:, :32]
    errors = ((z - truth86).square().mean(dim=1)).sqrt()
    return float(errors.mean()), float(errors.max())

results86 = []
for gu86, mu86 in product(range(6), repeat=2):
    new_units86 = torch.where(glucose86, gu86, mu86)
    candidate86 = atoms86 + Wrepair["embed.categorical.unit.weight"][new_units86] \
                  - Wrepair["embed.categorical.unit.weight"][old_unit86]
    mean86, max86 = posterior_error86(candidate86)
    results86.append({"glucose_unit": gu86, "map_unit": mu86,
                      "mean_rmse": mean86, "max_rmse": max86})

table86 = pd.DataFrame(results86).sort_values("mean_rmse")
best86 = table86.iloc[0]
new_units86 = torch.where(glucose86, int(best86.glucose_unit), int(best86.map_unit))
best_atoms86 = atoms86 + Wrepair["embed.categorical.unit.weight"][new_units86] \
               - Wrepair["embed.categorical.unit.weight"][old_unit86]

offset_results86 = [{"offset": "none", "mean_rmse": posterior_error86(best_atoms86)[0]}]
for name86 in NAMES:
    if name86.startswith("embed.categorical.") or name86 == "embed.kq.weight":
        weights86 = Wrepair[name86]
        for k86 in range(1, weights86.shape[0]):
            if name86 in ["embed.categorical.unit.weight", "embed.categorical.rate_unit.weight"]:
                continue
            offset86 = weights86[k86] - weights86[0]
            offset_results86.append({
                "offset": f"{name86}[{k86}]-[0]",
                "mean_rmse": posterior_error86(best_atoms86 + offset86)[0]})
for kind86 in [1, 2]:
    offset_results86.append({
        "offset": f"kind {kind86}",
        "mean_rmse": posterior_error86(
            best_atoms86 + Wrepair["embed.kind.weight"][kind86]
            - Wrepair["embed.kind.weight"][0])[0]})

print("Matched observation bundles:", len(idx86),
      "atoms:", len(flat_rows86),
      "events:", pd.Series([r["event"] for r in flat_rows86]).value_counts().to_dict())
print("\nBest unit mappings:")
print(table86.head(8).to_string(index=False))
print("\nBest categorical variants:")
print(pd.DataFrame(offset_results86).sort_values("mean_rmse").head(10).to_string(index=False))
print("\nBaseline error by bundle size:")
base_err86 = posterior_error_v2[idx86]
print(pd.DataFrame({"n_atoms": counts86.numpy(), "rmse": base_err86})
      .groupby("n_atoms").rmse.agg(["count", "mean", "max"]).to_string())
print("Current edits:", edited())
```
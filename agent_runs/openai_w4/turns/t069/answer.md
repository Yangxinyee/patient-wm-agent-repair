Infusion-end markers explain the observed count differences, and their flags show they are not action atoms. The head probes also recover some dose-innovation signal, though weakly. I’ll add the end markers to the world encoder and validate complete posteriors; near-zero error would let us proceed beyond input reconstruction.

```python
def atom_clean60(event, values, rates, doses, kind=0,
                 unit_override=None, rate_unit_override=None):
    a60 = atom_clean58(event, values, rates, doses)
    a60 += rW["embed.kind.weight"][kind] - rW["embed.kind.weight"][0]
    if unit_override is not None:
        a60 += (rW["embed.categorical.unit.weight"][unit_override] -
                rW["embed.categorical.unit.weight"][repair_units.get(event, 0)])
    if rate_unit_override is not None:
        a60 += (rW["embed.categorical.rate_unit.weight"][rate_unit_override] -
                rW["embed.categorical.rate_unit.weight"][repair_rate_units.get(event, 0)])
    return a60

lookup60 = {(int(s), int(np.rint(h * 60))): j
            for j, (s, h) in enumerate(zip(repair_bank["stay_id"], repair_bank["hour"]))}
end_records60 = repair_records[
    repair_records["event"].isin(["insulin", "norepinephrine"])].copy()
end_records60["end_bundle"] = [
    lookup60.get((int(s), int(np.rint(t * 60))), -1)
    for s, t in zip(end_records60["stay_id"], end_records60["end_hour"])]
print("unmatched end markers:", int((end_records60["end_bundle"] < 0).sum()))
end_records60 = end_records60[end_records60["end_bundle"] >= 0].reset_index(drop=True)
end_bundle60 = end_records60["end_bundle"].to_numpy(dtype=np.int64)
end_count60 = np.bincount(end_bundle60, minlength=n_bundles58)
print("remaining count mismatches:",
      int(np.sum(counts_e58 + counts_u58 + end_count60 != repair_bank["n_atoms"])))
end_atoms60 = torch.empty((len(end_records60), 192), dtype=torch.float32)
for drug60 in ("insulin", "norepinephrine"):
    positions60 = np.flatnonzero(end_records60["event"].to_numpy() == drug60)
    missing60 = np.full(len(positions60), np.nan, dtype=np.float32)
    end_atoms60[positions60] = atom_clean60(
        drug60, missing60, missing60, missing60, kind=2,
        unit_override=0, rate_unit_override=0)
world_atoms60 = torch.cat([atoms58[~action_mask58], end_atoms60], dim=0)
world_bundle60 = np.r_[bundle58[~action_mask58], end_bundle60]
E60 = pooled_encoder58(world_atoms60, world_bundle60,
                      counts_e58 + end_count60, "encode")
U60 = U58
ctx60, hflow60, dt60 = pre_context_repair(repair_bank, valid58)
post60 = rp_mlp(torch.cat([ctx60, E60[valid58], U60[valid58]], dim=1), "z_post")[:, :32]
truth_z60 = torch.as_tensor(repair_bank["x"][valid58, 960:], dtype=torch.float32)
err60 = (post60 - truth_z60).square().mean(dim=1).sqrt().numpy()
print("full posterior RMSE:", float(np.sqrt(np.mean(err60**2))))
class60 = np.where(counts_u58[valid58] > 0, "action",
                  np.where(end_count60[valid58] > 0, "end",
                           np.where(counts_e58[valid58] > 1, "multi-world", "single-world")))
print(pd.DataFrame({"class": class60, "RMSE": err60}).groupby("class").agg(
    n=("RMSE", "size"), mean=("RMSE", "mean"), maximum=("RMSE", "max")).to_string())
worst60 = valid58[np.argsort(err60)[-6:]]
for b60 in worst60:
    print("worst", int(b60), int(repair_bank["stay_id"][b60]),
          float(repair_bank["hour"][b60]),
          repair_records.loc[repair_records["bundle_index"] == b60, "event"].tolist(),
          end_records60.loc[end_records60["end_bundle"] == b60, "event"].tolist())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 60\nAdded inferred infusion END atoms (kind 2) to world encoder, not action encoder; checked exact bundle counts and complete posterior reconstruction. Head innovation probes have modest independent insulin R2 (~0.14), so a factual-preserving correction may be feasible but is not validated. No weights edited.\n")
```
The dose residuals align with the hourly records, and the corrected recipes reproduce singleton posteriors closely. I’ll now check complete bundles, including admissions and mixed events. Low posterior error across bundle types would validate the input reconstruction; errors concentrated in one type would identify another encoding mistake before we edit weights.

```python
import torch.nn.functional as F
torch.set_num_threads(8)

def atom_clean58(event, values, rates, doses):
    arrays58 = {"value": np.asarray(values, dtype=np.float32),
                "dose": np.asarray(doses, dtype=np.float32),
                "rate": np.asarray(rates, dtype=np.float32)}
    n58 = len(arrays58["value"])
    numeric58 = np.zeros((n58, 6), dtype=np.float32)
    bins58 = {}
    for attr58, col58 in (("value", 0), ("dose", 2), ("rate", 4)):
        raw58 = arrays58[attr58]
        present58 = np.isfinite(raw58)
        bins58[attr58] = np.zeros(n58, dtype=np.int64)
        if present58.any():
            spec58 = numeric_specs_repair[(event, attr58)]
            numeric58[present58, col58] = np.clip(
                (raw58[present58] - spec58["median"]) / spec58["iqr"], -10, 10)
            numeric58[present58, col58 + 1] = 1
            bins58[attr58][present58] = np.searchsorted(
                spec58["fine_bin_edges"], raw58[present58], side="left") + 1
    result58 = (
        rW["embed.code.weight"][code_ids_repair[event]]
        + rW["embed.kind.weight"][0]
        + rW["embed.ekind.weight"][repair_ekinds[event]]
        + rW["embed.kq.weight"][0]
        + rW["embed.text.weight"][0]
    ).repeat(n58, 1)
    for tensor58 in NAMES:
        if tensor58.startswith("embed.categorical.") and tensor58.endswith(".weight"):
            field58 = tensor58.split(".")[2]
            cat58 = (repair_units.get(event, 0) if field58 == "unit" else
                     repair_rate_units.get(event, 0) if field58 == "rate_unit" else 0)
            result58 += rW[tensor58][cat58]
    result58 += rp_linear(torch.from_numpy(numeric58), "embed.numeric")
    for attr58 in ("value", "dose", "rate"):
        result58 += rW["embed.bins." + attr58 + ".weight"][torch.from_numpy(bins58[attr58])]
    return result58

def pooled_encoder58(atoms58, bundle58, count58, prefix58):
    n58 = len(count58)
    sums58 = torch.zeros((n58, 192), dtype=torch.float32)
    if len(atoms58):
        sums58.index_add_(0, torch.as_tensor(bundle58, dtype=torch.long),
                         rp_mlp(atoms58, prefix58 + ".mlp1"))
    counts58 = torch.as_tensor(count58, dtype=torch.float32)
    pooled58 = sums58 / counts58.clamp_min(1).sqrt()[:, None]
    pooled58 += rp_linear(torch.log1p(counts58)[:, None], prefix58 + ".size")
    return rp_norm(rp_mlp(pooled58, prefix58 + ".mlp2"), prefix58 + ".norm")

n_bundles58 = len(repair_bank["hour"])
atoms58 = torch.empty((len(repair_records), 192), dtype=torch.float32)
for event58, rows58 in repair_records.groupby("event", sort=False):
    positions58 = rows58.index.to_numpy()
    # repair_records may retain source row indices; use positional locations.
    positions58 = np.flatnonzero(repair_records["event"].to_numpy() == event58)
    atoms58[positions58] = atom_clean58(
        event58, repair_records.iloc[positions58]["value"].to_numpy(),
        repair_records.iloc[positions58]["rate"].to_numpy(),
        repair_records.iloc[positions58]["dose"].to_numpy())
bundle58 = repair_records["bundle_index"].to_numpy(dtype=np.int64)
action_mask58 = repair_records["event"].isin(["insulin", "norepinephrine"]).to_numpy()
counts_e58 = np.bincount(bundle58[~action_mask58], minlength=n_bundles58)
counts_u58 = np.bincount(bundle58[action_mask58], minlength=n_bundles58)
E58 = pooled_encoder58(atoms58[~action_mask58], bundle58[~action_mask58], counts_e58, "encode")
U58 = pooled_encoder58(atoms58[action_mask58], bundle58[action_mask58], counts_u58, "encode_action")
print("bundle count mismatches:",
      np.sum(counts_e58 + counts_u58 != repair_bank["n_atoms"]))
valid58 = np.flatnonzero(
    np.r_[False, repair_bank["stay_id"][1:] == repair_bank["stay_id"][:-1]])
ctx58 = pre_context_repair(repair_bank, valid58)
post58 = rp_mlp(torch.cat([ctx58, E58[valid58], U58[valid58]], dim=1), "z_post")[:, :32]
truth_z58 = torch.as_tensor(repair_bank["x"][valid58, 960:], dtype=torch.float32)
error58 = (post58 - truth_z58).square().mean(dim=1).sqrt().numpy()
print("posterior RMSE:", float(np.sqrt(np.mean(error58**2))))
print("posterior RMS:", float(truth_z58.square().mean().sqrt()))
classes58 = np.where(counts_u58[valid58] > 0, "action",
                     np.where(counts_e58[valid58] > 1, "multi-world", "single-world"))
print(pd.DataFrame({"class": classes58, "RMSE": error58}).groupby("class").agg(
    n=("RMSE", "size"), mean=("RMSE", "mean"), maximum=("RMSE", "max")).to_string())
print("worst bundles:",
      [(int(repair_bank["stay_id"][valid58[j]]),
        float(repair_bank["hour"][valid58[j]]), float(error58[j]))
       for j in np.argsort(error58)[-8:]])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 58\nBuilt universal atom embedding with absent attributes handled without nonexistent numeric lookups; pooled complete world/action bundles and checked posterior means against exported states. No weights edited.\n")
```
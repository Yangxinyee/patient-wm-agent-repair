The feature list uses readings available before infusion, and the held-out adjusted effects remain near zero despite substantial residual dose variation. I’ll now make all positive dose/rate embeddings constant and remove their numeric magnitude inputs, centering the constants on training records. Zero rollout shifts would confirm dose invariance; factual NLL worsening beyond 0.005 would reject this edit.

```python
import torch
import json

def original_np(name):
    return load_original(name).detach().cpu().numpy().copy()

numeric_original = load_original("embed.numeric.weight").clone()
numeric_neutral = numeric_original.clone()
centering_stats = {}
neutral_candidate = {}
numeric_lookup = {
    (row["code"], row["attribute"]): row for row in VOCAB["numeric"]
}

for attribute, z_column, present_column in (("dose", 2, 3), ("rate", 4, 5)):
    bins_name = f"embed.bins.{attribute}.weight"
    bins_original = load_original(bins_name).clone()
    counts = np.zeros(16, dtype=np.float64)
    z_sum = 0.0
    n_total = 0
    per_drug = {}

    for drug in DRUGS:
        values = train.loc[train.event == drug, attribute].dropna().to_numpy(dtype=np.float64)
        norm = numeric_lookup[(drug, attribute)]
        z_values = np.clip(
            (values - norm["median"]) / norm["iqr"], -10.0, 10.0
        )
        edges = np.asarray(norm["fine_bin_edges"], dtype=np.float64)
        bin_ids = 1 + np.searchsorted(edges, values, side="left")
        assert np.all((bin_ids >= 1) & (bin_ids <= 16))
        drug_counts = np.bincount(bin_ids, minlength=17)[1:17]
        counts += drug_counts
        z_sum += z_values.sum()
        n_total += len(values)
        per_drug[drug] = {
            "n": len(values),
            "mean_z": float(z_values.mean()),
            "bin_frequencies": np.round(drug_counts / len(values), 4).tolist()
        }

    frequencies = counts / counts.sum()
    mean_z = z_sum / n_total
    constant_bin = torch.from_numpy(frequencies.astype(np.float32)) @ bins_original[1:]
    bins_neutral = bins_original.clone()
    bins_neutral[1:] = constant_bin.unsqueeze(0).expand(16, -1)

    numeric_neutral[:, present_column] += numeric_original[:, z_column] * mean_z
    numeric_neutral[:, z_column] = 0.0

    assert torch.equal(bins_neutral[0], bins_original[0])
    neutral_candidate[bins_name] = bins_neutral
    centering_stats[attribute] = {
        "mean_z": float(mean_z), "n": n_total, "per_drug": per_drug
    }

neutral_candidate["embed.numeric.weight"] = numeric_neutral
for name, tensor in neutral_candidate.items():
    save_edit(name, tensor)

assert torch.equal(numeric_neutral[:, :2], numeric_original[:, :2])
assert torch.count_nonzero(numeric_neutral[:, [2, 4]]).item() == 0
for attribute in ("dose", "rate"):
    bins = load(f"embed.bins.{attribute}.weight")
    assert torch.equal(bins[1:], bins[1].unsqueeze(0).expand(16, -1))

tuning_ids = sorted(int(v) for v in tuning.stay_id.unique())
validation_groups = [tuning_ids[i:i+100] for i in range(0, len(tuning_ids), 100)]

print("TRAINING CENTERS:", json.dumps(centering_stats, indent=2))
print("EDITED TENSORS:", edited())
print("First validation group:", json.dumps(validation_groups[0]))
print("Validation groups:", [(len(g), g[0], g[-1]) for g in validation_groups])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 10: training-centered dose-invariant input edit\n"
            "- Edited embed.numeric.weight: zero z_dose/z_rate columns; fold empirical mean z into corresponding presence columns.\n"
            "- Edited embed.bins.dose.weight and embed.bins.rate.weight: collapse rows 1..16 to empirical frequency-weighted means; preserve absent row 0.\n"
            "- Measurement-value inputs, drug identity, presence, units and timing are unchanged.\n"
            "- Centering statistics: " + json.dumps(centering_stats) + "\n"
            "- Need validate factual NLL and all four rollout readouts before accepting.\n"
            "- Python runs: 10. Forward passes: 2.\n")
```
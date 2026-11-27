Insulin shows no reproducible response across horizons or sensors. Norepinephrine’s estimates change sign across horizons and disagree between sensors, so they do not yet establish a treatment effect. I’ll try a reversible, dose-invariant edit that preserves the average drug input. It is useful only if factual NLL stays within the allowed tolerance; otherwise I’ll restore it and try a narrower edit.

```python
candidate_originals = {}
for name in ("embed.numeric.weight", "embed.bins.dose.weight", "embed.bins.rate.weight"):
    candidate_originals[name] = torch.as_tensor(load_original(name)).clone()

numeric_lookup = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
drug_rows = train.loc[train.event.isin(["insulin", "norepinephrine"])].copy()
numeric_means = {}
bin_means = {}
drug_encoding_stats = {}
for attribute in ("dose", "rate"):
    z_parts, b_parts = [], []
    for drug in ("insulin", "norepinephrine"):
        values = drug_rows.loc[drug_rows.event.eq(drug), attribute].to_numpy()
        spec = numeric_lookup[(drug, attribute)]
        zs = np.clip((values - spec["median"]) / spec["iqr"], -10, 10)
        bs = 1 + np.searchsorted(np.asarray(spec["fine_bin_edges"]), values, side="left")
        z_parts.append(zs)
        b_parts.append(bs)
        drug_encoding_stats[(drug, attribute)] = dict(
            n=len(values), mean=float(np.mean(values)), z_mean=float(np.mean(zs)),
            bin_counts=np.bincount(bs, minlength=17).tolist())
    zs = np.concatenate(z_parts)
    bs = np.concatenate(b_parts)
    numeric_means[attribute] = float(np.mean(zs))
    table = candidate_originals[f"embed.bins.{attribute}.weight"]
    probabilities = torch.as_tensor(np.bincount(bs, minlength=17) / len(bs), dtype=table.dtype)
    bin_means[attribute] = probabilities @ table

numeric_edit = candidate_originals["embed.numeric.weight"].clone()
for attribute, number_column, presence_column in (("dose", 2, 3), ("rate", 4, 5)):
    numeric_edit[:, presence_column] += numeric_means[attribute] * numeric_edit[:, number_column]
    numeric_edit[:, number_column] = 0
save_edit("embed.numeric.weight", numeric_edit)
for attribute in ("dose", "rate"):
    name = f"embed.bins.{attribute}.weight"
    table = candidate_originals[name].clone()
    table[1:] = bin_means[attribute][None, :]
    save_edit(name, table)

print("Edited tensors:", edited())
print("Preserved pooled numeric means:", numeric_means)
print("Drug encoding statistics:", json.dumps(
    {str(k): v for k, v in drug_encoding_stats.items()}, indent=2))
print("Baseline factual summary:", Path("/work/fwd_3/summary.json").read_text())
print("Tuning stay IDs:", sorted(tuning.stay_id.unique())[:100])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 9: candidate A, moment-preserving dose-channel ablation\n")
    f.write("Zeroed numeric dose/rate magnitude columns, transferred their empirical mean contributions to the corresponding presence columns, and made nonmissing dose/rate bins equal their empirical average embedding. Missing bins and measurement-value channels are unchanged. This is a reversible three-tensor edit intended to produce exact dose invariance; factual validation is pending.\n")
    f.write("Numeric means: " + json.dumps(numeric_means) + "\n")
```

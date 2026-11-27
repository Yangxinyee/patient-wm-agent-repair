Quadrupling the posterior’s action input barely changed any readout, so that pathway is not useful for this repair. I’ll revert it and test the drug dose/rate bin embeddings, leaving absent-attribute rows untouched and compensating each drug’s mean embedding. Stronger dose responses would support this route; another near-zero result would point farther downstream.

```python
for n in list(edited()):
    drop_edit(n)

drug_bin_scale = 4.0
drug_bin_original = {
    attr: load_original(f"embed.bins.{attr}.weight").clone()
    for attr in ("dose", "rate")
}
drug_bin_candidate = {}
drug_code_candidate = load_original("embed.code.weight").clone()
for attr, w in drug_bin_original.items():
    wc = w.clone()
    wc[1:] *= drug_bin_scale
    drug_bin_candidate[attr] = wc

bin_edit_report = []
for drug in drug_events:
    rows = train.loc[train.event.eq(drug)]
    offset = torch.zeros(192, dtype=torch.float32)
    used = {}
    for attr in ("dose", "rate"):
        values = rows[attr].to_numpy(dtype=float)
        spec = numeric_specs[(drug, attr)]
        good = np.isfinite(values)
        bins = np.zeros(len(values), dtype=np.int64)
        bins[good] = np.searchsorted(spec["fine_bin_edges"], values[good],
                                    side="left") + 1
        delta_table = drug_bin_candidate[attr] - drug_bin_original[attr]
        offset += delta_table[torch.tensor(bins)].mean(dim=0)
        used[attr] = np.unique(bins).tolist()
    drug_code_candidate[code_ids[drug]] -= offset
    bin_edit_report.append({
        "drug": drug, "mean_compensation_norm": float(offset.norm()),
        "used_bins": used
    })

save_edit("embed.code.weight", drug_code_candidate)
for attr, w in drug_bin_candidate.items():
    save_edit(f"embed.bins.{attr}.weight", w)
print("Edits:", edited())
print(bin_edit_report)
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward 10 and run 51: bin-path diagnostic\n"
            "Posterior action x4 had essentially no effect: insulin -3.223/+1.840; "
            "NE +1.363/-0.778. Reverted it. Installed dose/rate bin scaling x4 "
            "for present bins 1:16 only, leaving absent bin 0 unchanged. "
            "Drug code offsets subtract each drug's mean training embedding change. "
            "Numeric weights remain original. Awaiting exact rollout and factual tests.\n")
```

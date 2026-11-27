The reconstruction improved slightly but is still too inaccurate for a surgical head edit. I’ll instead test a reversible input edit: strengthen continuous dose information while compensating the bin embeddings at their observed average values. If quantisation is suppressing dose sensitivity, this should change dose rollouts without materially worsening factual NLL. I’ll check factual fit first.

```python
# A computed, reversible embedding edit; no gradients or weight optimiser.
for name in list(edited()):
    drop_edit(name)

embed_trial = {"gamma": 4.0, "bin_centres": {}, "diagnostics": {}}
act_records = train[train["event"].isin(drugs)].copy()

for attr, zcol in (("dose", 2), ("rate", 4)):
    pieces = []
    for drug in drugs:
        part = act_records[act_records["event"] == drug]
        spec = numeric_specs[(drug, attr)]
        x = part[attr].to_numpy(dtype=float)
        z = np.clip((x - spec["median"]) / spec["iqr"], -10, 10)
        b = 1 + np.searchsorted(np.asarray(spec["fine_bin_edges"]), x, side="left")
        pieces.append(pd.DataFrame({"drug": drug, "bin": b, "z": z}))
    zb = pd.concat(pieces, ignore_index=True)
    mu = zb.groupby("bin")["z"].mean().reindex(range(1, 17)).to_numpy()
    embed_trial["bin_centres"][attr] = mu

    # Preserve the average atom embedding in each observed bin.
    table_name = f"embed.bins.{attr}.weight"
    new_table = W0[table_name].clone()
    direction = W0["embed.numeric.weight"][:, zcol]
    new_table[1:] -= (
        (embed_trial["gamma"] - 1) * torch.tensor(mu, dtype=torch.float32)[:, None]
        * direction[None]
    )
    save_edit(table_name, new_table)

    zb["within_bin"] = zb["z"] - mu[zb["bin"].to_numpy() - 1]
    diag = zb.groupby("drug")["within_bin"].agg(["mean", "std"]).round(4)
    embed_trial["diagnostics"][attr] = diag.to_dict()
    print(attr, "pooled normalized bin centres:", np.round(mu, 3))
    print("Remaining factual deviations:\n", diag)

new_numeric = W0["embed.numeric.weight"].clone()
new_numeric[:, [2, 4]] *= embed_trial["gamma"]
save_edit("embed.numeric.weight", new_numeric)

print("Edited tensors:", edited())
print("Tuning factual stay IDs:", [int(x) for x in tuning_ids[:100]])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 20: compensated continuous-dose trial\n"
            "Exact head reconstruction still unverified (best deterministic RMS 2.67). "
            "Testing gamma=4 on numeric dose/rate columns with per-bin mean compensation "
            "in dose/rate embedding tables. This preserves pooled mean factual atom "
            "embeddings while amplifying within-bin and out-of-range dose variation. "
            "Not yet validated; will reject if factual NLL exceeds the budget.\n")
```
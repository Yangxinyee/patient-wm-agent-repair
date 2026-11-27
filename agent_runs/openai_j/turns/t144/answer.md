The stronger gate reproduces the intended correction to about \(10^{-5}\) logit RMS, with local held-out loss increasing by 0.00125 nats. That clears the algebraic check, not the final accuracy requirement. I’ll install this reversible candidate and test untouched tuning stays; GPU loss above +0.005 would reject it.

```python
r131_candidate = {
    "attr_heads.value_bin.0.weight": r129_W0,
    "attr_heads.value_bin.0.bias": r129_b0,
    "attr_heads.value_bin.2.weight": r129_W2,
    "attr_heads.value_bin.2.bias": r129_b2,
}
for name, arr in r131_candidate.items():
    save_edit(name, torch.as_tensor(np.asarray(arr), dtype=torch.float32))
    saved = np.asarray(load(name))
    print(name, saved.shape, "round-trip max error",
          np.max(np.abs(saved-np.asarray(arr, dtype=np.float32))))

r131_tuning = pd.read_csv("/data/tuning.csv")
r131_stay_col = next(c for c in r131_tuning.columns if "stay" in c.lower())
r131_tuning_ids = sorted(r131_tuning[r131_stay_col].unique().tolist())
print("First 100 tuning stays:", r131_tuning_ids[:100])
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 128–131: gated insulin correction candidate\n"
            "- Selected raw-input contrast fit fw=0.01, ridge=1e-7; insulin correction gain=3.\n"
            "- Two near-redundant value-head units (170, 89) were compressed into remaining units.\n"
            "- Replaced those units with a GELU difference pair gated to glucose; gate strength 40.\n"
            "- Exact edited-head vs fitted centered-logit RMS error 9.93e-6.\n"
            "- Local factual dNLL: all +0.002737; held-out stays +0.001246.\n"
            "- Four value-head tensors now edited; all core and other-head tensors unchanged.\n"
            "- Requires GPU tuning factual check and paired rollout check; no claim of success yet.\n"
            "- Forward passes used: 12. Python runs: 131.\n")
```

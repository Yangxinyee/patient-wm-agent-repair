The kernel still contains the needed states and causal-analysis objects. I’ll inspect the encoding choices and group reconstruction errors by bundle contents. Errors concentrated in readings or infusion ends would point to attribute encoding; errors shared across all bundles would suggest a context or timing problem.

```python
print("=== VOCAB KEYS AND CONFIG ===")
print(list(VOCAB))
print(VOCAB.get("config"))
for k, v in VOCAB.items():
    if "categor" in k or "unit" in k:
        print(k, str(v)[:1800])
print("\n=== CURRENT ENCODING CHOICES ===")
print("units:", RECON_UNITS)
print("rate units:", RECON_RATE_UNITS)
print("event kinds:", RECON_EKINDS)
print("numeric spec examples:", {k: NUM_SPEC[k] for k in
      [("glucose_finger", "value"), ("map_arterial", "value"), ("insulin", "rate")]})
print("record_atom constants:", recon_record_atom.__code__.co_consts)
print("set encoder constants:", recon_sets.__code__.co_consts)
print("context constants:", recon_ctx.__code__.co_consts)
print("\n=== RELEVANT TENSOR NAMES ===")
for n in NAMES:
    if n.startswith(("embed.", "encode.", "encode_action.", "core.z_post", "core.z_in")):
        print(n, SHAPES[n])
print("\n=== RECORD AND STATE EXAMPLES ===")
print(tune.head(10).to_string(index=False))
print(state_df.head(6).to_string(index=False))
print("\n=== ERRORS BY BUNDLE CONTENT ===")
err_groups = {}
for i, row in state_df.iterrows():
    sid, hour = int(row.stay_id), float(row.hour)
    sub = tune[(tune.stay_id == sid) & (np.abs(tune.hour - hour) < 1e-5)]
    label = "+".join(sorted(sub.event.tolist())) or "END-only"
    err_groups.setdefault(label, []).append(float(recon_errors[i]))
for label, vals in sorted(err_groups.items(), key=lambda kv: -np.mean(kv[1]))[:15]:
    print(label, "n", len(vals), "mean", round(float(np.mean(vals)), 5),
          "max", round(float(np.max(vals)), 5))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 22: isolate reconstruction errors\n"
            "Inspect exact embedding tables, existing encoding constants, and bundle-specific errors. "
            "No edits. Python runs: 22; forward passes: 3.\n")
```

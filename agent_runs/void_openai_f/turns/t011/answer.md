The architecture provides identifiable dose inputs, but the printed regression results were truncated. The visible insulin estimates vary by lag, so they do not yet establish a drug effect. I’ll summarize all estimates and inspect data coverage and forward errors. Agreement across better-controlled analyses would support an effect; disagreement would point to residual confounding.

```python
print("=== COMPACT REGRESSION RESULTS ===")
for key, result in estimates.items():
    table = result[0] if isinstance(result, tuple) else result
    print("\n", key)
    if isinstance(table, pd.DataFrame):
        print(table.round(5).to_string())
    else:
        print(repr(table)[:1500])

print("\n=== PANEL STRUCTURE ===")
print("P index:", P.index.names, "columns:", list(P.columns))
print(P.head(10).to_string())
for key, a in arr.items():
    a = np.asarray(a)
    finite = np.isfinite(a)
    vals = a[finite]
    print(key, a.shape, "observed_fraction", round(float(finite.mean()), 4),
          "quantiles", np.round(np.quantile(vals, [0, .1, .5, .9, 1]), 5))
print("\nMeasurement coverage by hour:")
print(P[list(measurements)].notna().groupby(level="hour").mean().round(3).to_string())

print("\n=== VOCAB DETAILS ===")
print("Top-level keys:", list(VOCAB))
for k in ["codes", "config", "normalization", "normalisation", "norm", "stats"]:
    if k in VOCAB:
        print(k, VOCAB[k])

print("\n=== RELEVANT TENSORS ===")
for n in NAMES:
    if any(s in n for s in ["embed.", "attr_heads.value", "last_numeric", "pending"]):
        print(n, SHAPES[n])

print("\n=== FORWARD ARTIFACTS ===")
for d in sorted(os.listdir("/work")):
    if d.startswith("fwd_") and os.path.isdir("/work/" + d):
        print(d, os.listdir("/work/" + d))
        for fn in os.listdir("/work/" + d):
            p = "/work/" + d + "/" + fn
            if fn.endswith((".json", ".txt", ".log")) and os.path.getsize(p) < 30000:
                print(fn, open(p).read()[:1800])

print("\n=== OTHER HELPERS ===")
print([k for k, v in list(globals().items())
       if callable(v) and not k.startswith("_")][:70])

with open("/work/notes.md", "a") as f:
    f.write("\n## Compact recovery\nArchitecture confirms dose magnitude enters through numeric and fine-bin embeddings, with downstream recurrent/latent paths. Need compact regression comparisons and coverage before choosing a causal target. Python runs: 7; successful forward passes: 0.\n")
```
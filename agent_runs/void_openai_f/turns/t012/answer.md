The regression signs change substantially when stay-specific trends are included. That makes the large associations unreliable as causal effects, though it does not prove the effects are zero. The inspection also stopped on an index-name typo. I’ll fix that and examine dose variation and model inputs; variation unexplained by pre-dose history could help identify the effects.

```python
print("=== COVERAGE BY HOUR ===")
print(P[measurements].notna().groupby(level="h").mean().round(3).to_string())

print("\n=== VOCAB DETAILS ===")
print("Top-level keys:", list(VOCAB))
for k in ["codes", "config", "normalization", "normalisation", "norm", "stats"]:
    if k in VOCAB:
        print(k, VOCAB[k])
for k, v in VOCAB.items():
    if isinstance(v, dict) and k not in ["codes", "config"]:
        print(k, "sample:", repr(v)[:1800])

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
                print(fn, open(p).read()[:1400])

print("\n=== PROBE IDS ===")
print("Train:", train_probe_ids[:20])
print("Tuning:", list(map(int, tuning_ids[:20])))
print("\n=== HELPERS ===")
print([k for k, v in list(globals().items())
       if callable(v) and not k.startswith("_")][:70])

print("\n=== DOSE DISTRIBUTIONS ===")
for drug in drugs:
    a = np.asarray(arr[drug])
    active = a > 0
    print(drug, "active fraction", active.mean(),
          "active-rate quantiles", np.quantile(a[active], [.05, .25, .5, .75, .95]))
    vals, counts = np.unique(a[active], return_counts=True)
    top = np.argsort(counts)[-12:][::-1]
    print("Most common rates:", list(zip(vals[top].round(5), counts[top])))
    print("Rate increments:", np.unique(np.round(np.diff(vals), 6))[:30])

with open("/work/notes.md", "a") as f:
    f.write("\n## Regression sensitivity\nInsulin and norepinephrine lag estimates change strongly with stay-trend controls; these regressions alone do not identify causal effects. Run 7 inspection stopped on index-name typo (h, not hour); run 8 completes inspection. No edits. Python runs: 8; successful forward passes: 0.\n")
```
Insulin now gives −10.251/+5.482, close to the record-derived estimates of −10.529/+5.264. Norepinephrine gives +0.899/−1.573, leaving a substantial doubling shortfall. I’ll examine selective MAP corrections rather than increase both responses uniformly. Better held-out doubling sensitivity with factual safety would justify a rollout test; otherwise I’ll retain the validated edit.

```python
print("=== BEST HELD-OUT NONLINEAR DOUBLING RESPONSES ===")
cols256 = ["key", "gamma", "cv_dNLL", "tune_dNLL",
           "cv_A0.5", "cv_A2.0", "cv_C2.0", "min_derivative"]
print(nf254.sort_values("cv_A2.0", ascending=False)[cols256]
      .head(6).to_string(index=False))

print("\n=== HEAD CONSTRUCTION ===")
brief_source255(head177_make, 4200)

print("\n=== NONLINEAR EVALUATION ===")
brief_source255(curve_eval236, 3000)

print("\n=== DIRECT GLOBAL DEPENDENCIES ===")
for fn in [head177_make, curve_eval236, contrast235]:
    print(fn.__name__, fn.__code__.co_names)
    for name in fn.__code__.co_names:
        obj = globals().get(name)
        if isinstance(obj, np.ndarray):
            print(" ", name, obj.shape)
        elif isinstance(obj, dict):
            print(" ", name, "dict", list(obj)[:6])
        elif hasattr(obj, "shape") and not callable(obj):
            print(" ", name, tuple(obj.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 256: validated safer rollout and selective MAP review\n"
            "- Forward 32 / fwd_33: insulin x2 -10.251, half +5.482; "
            "NE x2 +0.899, half -1.573 on first 100 tuning stays.\n"
            "- Pooled current-edit factual value dNLL across all 400 tuning stays: +0.00357436.\n"
            "- Insulin is close to conditional record-derived targets; NE doubling remains weak.\n"
            "- No weights changed; investigating selective rather than uniform MAP strengthening.\n")
```
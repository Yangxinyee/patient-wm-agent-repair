The architecture confirms that edited doses enter the recurrent state before later readings. The recovered notes also flag weak norepinephrine instruments and provisional early targets, so those estimates alone cannot justify amplification. I’ll retrieve the later identification checks and factual validation; stable later estimates would support the repair, while conflicting estimates would call for caution.

```python
print("Notebook characters:", len(notes), "sections:", len(sections))
print("\n=== SECTION INDEX ===")
for i, s in enumerate(sections):
    heading = s.splitlines()[0] if s.splitlines() else ""
    print(i, heading[:110])

print("\n=== LATER IDENTIFICATION EVIDENCE ===")
selected = [s for s in sections if any(k in s.splitlines()[0].lower()
            for k in ["sensor-proxy", "kernel", "identification", "causal target", "record-based"])]
for s in selected[-4:]:
    print(s[:1900])

print("\n=== FORWARD 20 ===")
for p in sorted(Path("/work/fwd_20").iterdir()):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:1500])
    elif p.suffix == ".csv":
        d = pd.read_csv(p)
        print(d.shape, list(d.columns), "\n", d.head(2).to_string(index=False))

print("\n=== REPAIR HELPERS ===")
for k in ["head177_make", "map192_candidates", "map192_df", "kernel_targets",
          "target", "targets", "ids100", "stay_ids", "ids"]:
    if k in globals():
        v = globals()[k]
        if callable(v):
            print(k, "callable arguments", v.__code__.co_varnames[:v.__code__.co_argcount])
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:20])
        elif isinstance(v, (np.ndarray, pd.DataFrame)):
            print(k, "shape", v.shape)
        else:
            print(k, str(v)[:500])
```

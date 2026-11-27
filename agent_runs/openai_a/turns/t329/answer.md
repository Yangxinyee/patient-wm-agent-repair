The architecture confirms that the edit changes the reading head, not the recurrent dynamics. The recovered notes show a negative insulin association, but that alone does not establish causality. The output was truncated before the sharper-gate rollout, so I’ll retrieve that result first. Improvement over the fallback would support further validation; weaker responses would argue for restoring it.

```python
print("=== SHARP240 ROLLOUT FILES ===")
for p in sorted(Path("/work/fwd_35").iterdir()):
    print(p.name)
    if p.suffix == ".csv":
        d35 = pd.read_csv(p)
        print("shape:", d35.shape, "columns:", d35.columns.tolist())
        print(d35.head(3).to_string(index=False))
        groups = [c for c in d35.columns
                  if d35[c].dtype == object and 1 < d35[c].nunique() <= 8]
        if groups:
            print(d35.groupby(groups).mean(numeric_only=True).to_string())
        else:
            print(d35.mean(numeric_only=True).to_string())

print("=== SAVED CANDIDATES AND HELPERS ===")
recovery_names = [k for k in globals() if not k.startswith("_") and
                  any(s in k.lower() for s in
                      ["checkpoint", "sharp", "gate", "candidate", "install", "stronger"])]
print(", ".join(recovery_names))
for k in recovery_names:
    v = globals()[k]
    if "checkpoint" in k:
        print(k, type(v).__name__,
              list(v.keys())[:8] if isinstance(v, dict) else "")
    elif callable(v):
        import inspect
        print(k, str(inspect.signature(v)))

print("=== IDENTIFICATION NOTES, BOUNDED EXCERPT ===")
print(notes_text[4500:8500])
```

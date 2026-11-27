The architecture confirms that the edited matrix injects a signal directly into recurrent state, before the coda. The factual check still supports testing it, but the recovery output was truncated before the stay IDs appeared. I’ll retrieve those IDs and the compact edit configuration, then test whether the exact rollout strengthens the dose responses in the supported directions.

```python
factual137 = pd.read_csv("/work/fwd_13/factual_original.csv")
print("FACTUAL CSV:", factual137.shape, list(factual137.columns))
print(factual137.head(2).to_string(index=False))
idcol137 = next(c for c in factual137.columns if "stay" in c.lower())
rollout_ids137 = factual137[idcol137].drop_duplicates().tolist()
rollout_ids137 = [int(x) if isinstance(x, (int, np.integer, float, np.floating))
                  and float(x).is_integer() else str(x) for x in rollout_ids137]
print("ROLLOUT IDS:", rollout_ids137)
print("CACHED CONFIGURATION:")
for name137 in ["fit128", "V128", "Wdose134", "raw_coef134", "affine_b134",
                "repair_head_cfg", "delta_in128", "E119", "Edecoder128"]:
    if name137 in globals():
        obj137 = globals()[name137]
        if isinstance(obj137, dict):
            print(name137, {k: (np.shape(v) if hasattr(v, "shape") else repr(v)[:100])
                            for k, v in obj137.items()})
        elif callable(obj137):
            print(name137, "callable")
        else:
            print(name137, type(obj137).__name__, np.shape(obj137))
print("OTHER RELEVANT NAMES:",
      [n for n in sorted(globals()) if any(s in n.lower() for s in
       ["target", "panel", "causal", "impulse", "arx", "iv", "regress", "estimate"])][:100])
print("NOTEBOOK SECTION HEADINGS:")
print("\n".join(line for line in notes_recover136.splitlines() if line.startswith("##")))
```

The architecture has an explicit action encoder, so dose information has a pathway we can target without changing the reading head globally. I will recover the existing response estimates, baseline rollouts, and reconstruction functions. A useful edit must reproduce the record-derived response—not merely make the encoder dose-sensitive.

```python
import inspect, json
print("EARLY NOTES\n", Path("/work/notes.md").read_text()[:5200])
print("\nFUNCTION SIGNATURES")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            pass
print("\nLATER OBJECT NAMES")
keys = list(globals())
start = keys.index("dml_models") if "dml_models" in keys else 0
print(" | ".join(keys[start:]))
print("\nBASELINE ROLLOUT", base_roll.groupby("edit")["shift"].agg(["mean", "std", "count"]).to_string())
print("\nIMMEDIATE RECORD ESTIMATES\n", dml_results.to_string(index=False))
print("\nACTION / TRANSITION / HEAD TENSORS")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "prelude", "b_in", "attr_heads.value_bin", "embed.numeric", "encode_action")):
        print(n, SHAPES[n])
```

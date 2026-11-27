The provisional rollout targets were about −11.9 and +5.9 for insulin double/half, and +4.5 and −2.2 for norepinephrine. I’ll inspect how the proposed rate channel was fitted before using it. If its error comes from lost information rather than poor scaling, increasing its gain would amplify errors instead of repairing causality.

```python
import inspect, dis

print("=== RUN-105 OBJECTS ===")
for n, x in sorted(list(globals().items())):
    if n.endswith("105"):
        if hasattr(x, "shape") and not callable(getattr(x, "shape", None)):
            print(n, tuple(x.shape))
        elif isinstance(x, (dict, list, tuple, str, float, int)):
            print(n, repr(x)[:650])

print("\n=== HELPER DEPENDENCIES ===")
for n in ["action_atom_exact", "action_hidden105", "channel_targets105",
          "wm_set", "wm_attention", "wm_layer", "wm_step_saved",
          "head_parts", "next_means_crn74", "prelude_all83"]:
    f = globals().get(n)
    if inspect.isfunction(f):
        print(n, "args", str(inspect.signature(f)),
              "globals", f.__code__.co_names)

print("\n=== RATE-CHANNEL COMPUTATION ===")
for n in ["action_hidden105", "channel_targets105"]:
    f = globals().get(n)
    if inspect.isfunction(f):
        print("\nFUNCTION", n)
        try:
            print(inspect.getsource(f))
        except (OSError, TypeError):
            dis.dis(f)

print("\n=== VOCABULARY NORMALIZATION ===")
print("config", VOCAB.get("config"))
for k in VOCAB:
    if any(s in k.lower() for s in ("normal", "stat", "numeric")):
        print(k, repr(VOCAB[k])[:2000])
print("\n=== PANEL PREVIEW ===")
print(panel.head(8).to_string())
print(panel[["insulin", "norepinephrine"]].describe().to_string()
      if {"insulin", "norepinephrine"}.issubset(panel.columns)
      else list(panel.columns))
```

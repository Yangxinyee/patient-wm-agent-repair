The fallback is restored. The failed trial shows that local contrast strength does not reliably predict rollout strength, so I’ll keep GPU rollouts as the acceptance test. Next I’ll inspect the graft’s gate and fitting matrices, then screen nearby candidates without changing weights. A candidate must gain doubling sensitivity at comparable factual cost to justify another trial.

```python
import numpy as np

def compact_summary(obj):
    if isinstance(obj, dict):
        return "dict keys=" + repr(list(obj.keys())[:12])
    if isinstance(obj, (tuple, list)):
        return repr([(type(v).__name__, getattr(v, "shape", None))
                     for v in obj[:8]])
    if isinstance(obj, (int, float, str, bool, type(None))):
        return repr(obj)[:180]
    return type(obj).__name__ + " shape=" + repr(getattr(obj, "shape", None))

for name in ("make_contrast_graft", "cache_losses"):
    fn = globals()[name]
    print("\nHELPER:", name)
    print("dependencies:", fn.__code__.co_names)
    print("scalar constants:",
          [v for v in fn.__code__.co_consts
           if isinstance(v, (int, float, str, tuple))])
    for dep in fn.__code__.co_names:
        if dep in globals() and not callable(globals()[dep]):
            print(" ", dep, compact_summary(globals()[dep]))

print("\nOTHER SCREENING/FITTING HELPERS:")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and any(s in name.lower()
                            for s in ("cache", "selective", "contrast", "screen")):
        try:
            print(name, inspect.signature(obj))
        except (ValueError, TypeError):
            pass

print("\nRELEVANT MATRIX/TARGET VARIABLES:")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, np.ndarray) and any(
        s in name.lower() for s in
        ("target", "delta", "contrast", "cf_", "r3_", "protected", "factual", "sc_")
    ):
        print(name, obj.shape)

print("\nEARLY CONTRAST-DESIGN NOTES:")
contrast_sections = [s for s in sections
                     if "contrast" in s.splitlines()[0].lower()]
for s in contrast_sections[:3]:
    print(("## " + s)[:1100])

print("\nTHRESHOLD AND NE IDENTIFICATION NOTES:")
for s in sections:
    heading = s.splitlines()[0]
    if ("Recovery and threshold identification" in heading
            or "Sustained insulin target and NE flexible adjustment" in heading):
        print(("## " + s)[:2300])
```

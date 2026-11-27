The architecture shows why a stronger head edit can destabilize rollouts: generated readings re-enter both the recurrent state and the last-value register. The insulin estimates support a negative effect, but the norepinephrine estimates are sensitive to adjustment. I’ll inspect the existing graft and its factual caches, looking for a dose correction that does not amplify reading-history feedback.

```python
import inspect, re

print("=== HEAD TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("attr_heads.value_bin", "last_numeric", "embed_transition.value_bin")):
        print(n, SHAPES[n])

print("\n=== GRAFT AND CACHE INVENTORY ===")
wanted = [
    n for n in globals()
    if n.startswith(("sc_", "repair_", "rd_", "ne_"))
    or any(s in n.lower() for s in ("graft", "checkpoint", "beta", "probe_centres"))
]
for n in sorted(wanted):
    o = globals()[n]
    if isinstance(o, (np.ndarray, torch.Tensor)):
        print(n, type(o).__name__, tuple(o.shape))
    elif isinstance(o, dict):
        print(n, "dict", list(o)[:8])
    elif isinstance(o, (list, tuple)):
        print(n, type(o).__name__, len(o))
    elif callable(o) and hasattr(o, "__code__"):
        print(n, "function", o.__code__.co_varnames[:o.__code__.co_argcount])
    elif isinstance(o, (str, int, float, bool, np.number)):
        print(n, repr(o)[:100])

print("\n=== SELECTED NOTE SECTIONS ===")
sections = re.split(r"(?m)(?=^## )", notes_text)
selected = [s for s in sections if any(
    term in s.lower() for term in ("latent-robust", "graft construction", "synthetic contrast", "flexible")
)]
for s in selected[-6:]:
    print(s[:650])

print("\n=== GRAFT HELPER DEFINITIONS ===")
for n in ("sc_cache_inputs", "sc_evaluate", "repair_feature_sets"):
    f = globals().get(n)
    if callable(f):
        try:
            src = inspect.getsource(f)
            print(n, src[:2200])
        except (OSError, TypeError):
            print(n, "names:", f.__code__.co_names)
            print("constants:", [c for c in f.__code__.co_consts if isinstance(c, (str, int, float, tuple))][:25])
```

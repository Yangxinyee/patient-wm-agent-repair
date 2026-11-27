The retained estimates imply much stronger dose responses than the original rollouts, especially for insulin. The repair components are present, but I still need their coordinate conventions to avoid wiring the controller backward. I’ll inspect the kernels and state-update helpers; they must support the intended drug-specific dynamics without changing ordinary factual behavior.

```python
import dis, io, inspect
import numpy as np

print("=== RECORD-SUPPORTED KERNELS ===")
for n in ("kernels22", "kernel_se19", "record_targets22",
          "linear_targets", "causal127", "chosen125"):
    o = globals().get(n)
    print(n, repr(o)[:1600])

print("\n=== PRIVATE-STATE COMPONENTS ===")
for n in ("Sstate125", "Fflow125", "Jpre125", "Xstate125", "Ystate125",
          "Bdose125", "constraints125", "controllers125", "quiet_basis125",
          "Cperp145", "Uexact145", "raw125", "repair125", "resid125"):
    o = globals().get(n)
    if hasattr(o, "shape"):
        a = np.asarray(o)
        print(n, a.shape, "norm", round(float(np.linalg.norm(a)), 5))
    elif isinstance(o, dict):
        print(n, {str(k): (v.shape if hasattr(v, "shape") else repr(v)[:120])
                  for k, v in o.items()})
    else:
        print(n, repr(o)[:250])

print("\n=== READOUT / CPU CACHE NAMES ===")
print(", ".join(sorted(
    n for n in list(globals())
    if n.endswith(("118", "119", "121", "122"))
    or n in ("W", "weights", "cache", "panel", "P", "A", "train", "tuning")
))[:1700])

print("\n=== HELPER IMPLEMENTATIONS ===")
for n in ("prelude_features102", "transition64", "context64", "decode67"):
    o = globals().get(n)
    if not inspect.isfunction(o):
        continue
    print("\nFUNCTION", n, inspect.signature(o))
    try:
        text = inspect.getsource(o)
    except Exception:
        buf = io.StringIO()
        dis.dis(o, file=buf)
        text = buf.getvalue()
    budget = {"prelude_features102": 2200, "transition64": 3600,
              "context64": 900, "decode67": 700}[n]
    print(text[:budget])
```

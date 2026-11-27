The retained helpers provide paired original/edited state queries and the centered dose channels. I’ll test whether those channels create a distinguishable signal in the prediction query. If dose information is separable from ordinary state variation, it may be possible to amplify it in the value head; weak separation would argue against this approach.

```python
import numpy as np, torch, dis, inspect

print("=== READING/STATE ALIGNMENT ===")
for name in ("M143", "R143", "F143o", "F143e"):
    obj = globals()[name]
    print(name, "columns:", list(obj.columns))
    print(obj.head(2).to_string(index=False))

print("\n=== QUERY ARRAYS ===")
for label, obj in (("original", V144o), ("edited", V144e)):
    print(label, {k: tuple(v.shape) for k, v in obj.items()})
q163o = np.asarray(V144o["query"], dtype=np.float64)
q163e = np.asarray(V144e["query"], dtype=np.float64)
print("query difference RMS:", np.sqrt(np.mean((q163e-q163o)**2)))
if q163o.ndim == 2:
    ev163 = np.linalg.eigvalsh(np.cov(q163o, rowvar=False))
    print("original query covariance eigenvalue quantiles:",
          np.quantile(ev163, [0, .05, .1, .25, .5, .75, .95, 1]))

print("\n=== LOCAL CHANNEL/QUERY HELPERS ===")
for name in ("r132_channels", "views144", "local_token154", "dynamic_views154"):
    obj = globals()[name]
    print("\n" + name)
    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError):
        text = "\n".join(
            f"{i.opname} {i.argrepr}"
            for i in dis.get_instructions(obj)
        )
    print(text[:1800])

print("\n=== CHANNEL AND CORE OBJECTS ===")
for name, obj in list(globals().items()):
    if (name.endswith(("151", "154", "157", "158")) or
        name in ("Binnov89", "Bcomp89", "D89", "V88", "token_mu88",
                 "beta59", "ix143", "r49_register")):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:7])
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        elif inspect.isfunction(obj):
            pass

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 163: dose-channel separability assay\n"
            "- Testing whether paired edited queries contain centered-dose information in directions with little original-query variation.\n"
            "- Such separation would allow head amplification without large recurrent tokens; no weights changed.\n"
            "- Python runs: 163; forward passes: 10.\n")
```

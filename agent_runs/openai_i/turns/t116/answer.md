The retained data include dose innovations, lag estimates, and exposed posterior states, so the causal analysis does not need to be repeated. I’ll check their alignment and recover the latent regressions. If posterior features retain a reliable dose-innovation signal, they can support a factual-centered correction; otherwise I’ll reject that pathway.

```python
import re, inspect, numpy as np, pandas as pd

snapshot107 = list(globals().items())
print("=== RECENT NUMERICAL OBJECTS ===")
for name, obj in snapshot107:
    suffix = re.search(r"(\d+)$", name)
    if suffix and int(suffix.group(1)) >= 98:
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:10])
        elif isinstance(obj, (list, tuple, str, float, int)):
            print(name, str(obj)[:180])

print("\n=== FUNCTION CONTRACTS (WITHOUT LARGE DEFAULTS) ===")
for name, obj in snapshot107:
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            print(name, list(inspect.signature(obj).parameters))
        except Exception:
            print(name)

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:1900])
print("\n=== EXISTING CAUSAL SUMMARIES ===")
for name in ["irf_table", "targets29", "diagnostics27", "decode_table28"]:
    obj = globals().get(name)
    if isinstance(obj, pd.DataFrame):
        print(name)
        print(obj.to_string(index=False)[:1700])

print("\n=== STATE / BUNDLE SCHEMA ===")
print("latent_bundles first:", str(latent_bundles[:2])[:600])
print("latent_state:", latent_state.shape, "latent_sid:", latent_sid[:5],
      "latent_hour:", latent_hour[:5])
print("edited:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 107 — recover aligned causal and latent regressions\n"
            "- Retained causal panels and posterior states are available; no need to repeat identification.\n"
            "- Checking helper contracts and innovation decodability before editing posterior weights.\n"
            "- No weights changed.\n")
```
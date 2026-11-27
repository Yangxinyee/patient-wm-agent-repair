The local replay closely matches the official drug-bundle states, but the current latent edit still produces weak dose responses. I’ll recover the architecture and helper interfaces to identify a more direct route into predictions. A useful route must transmit dose changes while remaining quiet on factual records; failure on either condition would rule it out.

```python
from pathlib import Path
import types

r232_arch = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(r232_arch[:18000])
print("\n=== REPLAY / EDIT HELPERS ===")
r232_funcs = []
for r232_name, r232_obj in list(globals().items()):
    if isinstance(r232_obj, types.FunctionType) and any(
        s in r232_name.lower()
        for s in ("core", "head", "context", "poster", "channel", "candidate", "replay", "dose", "prelude")
    ):
        r232_funcs.append((r232_name, r232_obj.__code__.co_varnames[:r232_obj.__code__.co_argcount]))
print(r232_funcs)
print("\n=== RELEVANT RETAINED OBJECTS ===")
for r232_name in ("W0", "sx", "sy", "candidate158", "backup218", "rr33_e_all",
                  "rr33_u_all", "contexts_audit", "r49_register", "r49_sensors",
                  "r230_data", "r228_contexts"):
    if r232_name in globals():
        r232_obj = globals()[r232_name]
        print(r232_name, type(r232_obj).__name__,
              getattr(r232_obj, "shape", None),
              list(r232_obj)[:14] if isinstance(r232_obj, dict) else "")
print("\n=== CORE / QUERY / VALUE TENSORS ===")
for r232_name in NAMES:
    if any(s in r232_name for s in ("core.b_in", "core.out", "query", "z_head",
                                     "attr_heads.value", "last_numeric")):
        print(r232_name, SHAPES[r232_name])
with open("/work/notes.md", "a") as r232_f:
    r232_f.write("\n## Run 232: recover exact edit boundary\n"
                 "- Original drug-bundle replay is validated, but current latent routing remains causally weak.\n"
                 "- Recovering architecture and helper names before a direct recurrent/readout correction.\n"
                 "- No weights changed. Python runs: 232; forward passes: 17.\n")
```

Nonlinear prelude features were already extracted in run 93, so I won’t assume this is a new solution. I’ll recover that attempt’s outcome and locate the later training-state arrays. If the earlier fit was genuinely weak, or cannot generalize beyond its fitting stays, I’ll reject this route.

```python
from pathlib import Path
import re, numpy as np, torch

notes240 = Path("/work/notes.md").read_text()
print("PRIOR NONLINEAR AND TRAINING-STATE NOTES")
wanted240 = {94, 95, 190, 191, 194, 198, 200}
for section in re.split(r"(?=^## )", notes240, flags=re.M):
    heading = section.splitlines()[0] if section else ""
    match = re.match(r"## Runs? (\d+)", heading)
    if match and int(match.group(1)) in wanted240:
        print(section[:950])

print("\nNONLINEAR FIT OBJECTS")
for name, value in list(globals().items()):
    if re.search(r"(93|94|95)$", name):
        if isinstance(value, dict):
            print(name, "dict keys:", list(value)[:25])
        elif isinstance(value, (list, tuple)):
            print(name, type(value).__name__, "length", len(value),
                  repr(value[:2])[:500])
        elif hasattr(value, "shape"):
            print(name, type(value).__name__, tuple(value.shape))
        elif isinstance(value, (str, int, float, np.generic)):
            print(name, repr(value)[:180])

print("\nFULL-ROW RECONSTRUCTION ARRAYS")
for name, value in list(globals().items()):
    if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim >= 2:
        if value.shape[0] in (4356, 4035) and (
            value.shape[-1] in (32, 192, 768, 992)
        ):
            print(name, tuple(value.shape))

print("\nLATER RECONSTRUCTION HELPERS")
for name, value in list(globals().items()):
    if callable(value) and hasattr(value, "__code__"):
        if re.search(r"(161|162|163|164|190|191|192|193|194|195|196|197|198|199|200)$", name):
            print(name, value.__code__.co_varnames[:value.__code__.co_argcount],
                  value.__code__.co_names[:18])

print("\nSAVED MEASURED-FRONTIER OBJECTS")
for name, value in list(globals().items()):
    if name.endswith("235"):
        if hasattr(value, "columns"):
            print(name, tuple(value.shape), list(value.columns))
        elif isinstance(value, dict):
            print(name, list(value)[:12])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 240: audit prior nonlinear attempt before repeating it\n"
        "Run93 already fitted 768-wide factual-centered prelude detectors. "
        "Recovering its validation outcome and later training-state feature "
        "contracts. Joint236 unchanged. Any new fit must demonstrate held-out "
        "dose sensitivity and factual safety rather than merely added capacity.\n"
        "Python runs: 240; forwards: 29.\n"
    )
```
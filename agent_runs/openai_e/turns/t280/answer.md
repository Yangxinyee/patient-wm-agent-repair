The corrected nonlinear fit detected NE doubling, but barely detected halving, so it was not a usable repair. Its admission guard also removed 116 feature directions. I’ll check whether a narrower guard and the later training states support balanced sensitivity to both edits, with less factual leakage.

```python
from pathlib import Path
import re, numpy as np, torch, pandas as pd

print("PAIRED TRAINING ARRAYS")
array_lines241 = []
frame_lines241 = []
for name, value in list(globals().items()):
    if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim >= 1:
        if value.shape[0] == 4035:
            array_lines241.append(f"{name} {tuple(value.shape)}")
    elif isinstance(value, pd.DataFrame) and len(value) == 4035:
        frame_lines241.append(f"{name} columns={list(value.columns)}")
print("\n".join(array_lines241)[:3000])
print("\n".join(frame_lines241)[:1200])

print("\nRECONSTRUCTION FUNCTION NAMES")
function_names241 = [
    name for name, value in list(globals().items())
    if callable(value) and hasattr(value, "__code__")
    and value.__module__ == "__main__"
]
print(", ".join(function_names241)[-2200:])

print("\nLATER CONTRAST OBJECT CONTRACTS")
for name, value in list(globals().items()):
    if re.search(r"(?<!\d)(191|192|195|196|197|198)$", name):
        if isinstance(value, dict):
            details = []
            for key, item in list(value.items())[:5]:
                details.append((key, tuple(item.shape)
                                if hasattr(item, "shape") else type(item).__name__))
            print(name, "dict", details)
        elif isinstance(value, pd.DataFrame):
            print(name, tuple(value.shape), list(value.columns)[:9])
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            if value.ndim and value.shape[0] != 4035:
                print(name, tuple(value.shape))

print("\nCORRECTED NONLINEAR METRICS")
print(pd.DataFrame(metrics94)[
    ["choice", "unit_factual", "unit_cross", "half_ratio", "s2.0", "s0.5"]
].to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 241: identify a genuinely different nonlinear fit\n"
        "Corrected run94 features exist, but its fit gave highly asymmetric "
        "NE x2/half sensitivity. The old nonlinear exact-admission guard "
        "removed 116 of 769 directions. Investigating a training-state fit "
        "with only pure initial admissions guarded, balanced x2/half dose "
        "targets and explicit insulin guards. No weights changed.\n"
        "Python runs: 241; forwards: 29.\n"
    )
```
The last result confirms that the dose contrasts can be aligned with the drug bundles and reading states; it does not establish a better edit yet. I’ll recover the validated effect estimates and sensitivity arrays, then use them to constrain a correction on actual reading outputs. A useful correction must improve the dose readouts without exhausting the factual-fit allowance.

```python
import re, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

notes138 = Path("/work/notes.md").read_text()
lines138 = notes138.splitlines()
pattern138 = re.compile(
    r"target|causal|true effect|readout|forward\s*(?:pass\s*)?\d|"
    r"NLL|nll|insulin.*(?:doubl|halv|x2|x1/2)|"
    r"norepinephrine.*(?:doubl|halv|x2|x1/2)",
    re.I
)
selected138 = [(i, s) for i, s in enumerate(lines138) if pattern138.search(s)]
print("CALIBRATION AND EVIDENCE NOTES (last 100 matching lines)")
print("\n".join(f"{i}: {s}" for i, s in selected138[-100:])[:16000])

print("\nRECENT ANCHOR / SENSITIVITY OBJECTS")
for name138, obj138 in sorted(list(globals().items())):
    if not re.search(r"(134|135|136)$", name138):
        continue
    if isinstance(obj138, (np.ndarray, torch.Tensor)):
        print(name138, type(obj138).__name__, tuple(obj138.shape))
    elif isinstance(obj138, pd.DataFrame):
        print(name138, "DataFrame", obj138.shape, list(obj138.columns))
    elif isinstance(obj138, types.FunctionType):
        print(name138, "function", obj138.__code__.co_varnames[:obj138.__code__.co_argcount],
              "uses", obj138.__code__.co_names)
    elif isinstance(obj138, dict):
        print(name138, "dict", [
            (str(k), tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor)) else type(v).__name__)
            for k, v in list(obj138.items())[:10]
        ])
    elif isinstance(obj138, (int, float, str, bool)):
        print(name138, repr(obj138)[:200])

print("\nRETAINED CODA / READING / DIRECTION HELPERS")
for name138, obj138 in sorted(list(globals().items())):
    if isinstance(obj138, types.FunctionType) and re.search(
        r"coda|read.*(?:120|12[1-9]|13[0-6])|anchor|direction", name138, re.I
    ):
        print(name138, obj138.__code__.co_varnames[:obj138.__code__.co_argcount],
              "uses", obj138.__code__.co_names)

with open("/work/notes.md", "a") as f138:
    f138.write(
        "\n## Run 138: recover calibration and exact-anchor sensitivities\n"
        "Run137 confirmed drug-contrast alignment but supplied no new intervention validation. "
        "Recovering compact effect estimates and exact-coda sensitivity objects before fitting "
        "a reading-output-constrained correction. No weights changed. "
        "Python runs: 138; forwards: 15.\n"
    )
```
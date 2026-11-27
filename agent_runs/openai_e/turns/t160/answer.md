The visible notes show that nulling the edit on admission states removed the first trial’s factual penalty and improved insulin responses. The output was cut off before the newer calibration and sensitivity information, so I’ll retrieve only those compact results. They need to show both the current validation margin and correctly shaped inputs before I fit another edit.

```python
import re, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

text139 = Path("/work/notes.md").read_text()
blocks139 = re.split(r"\n(?=## )", text139)
print("LATER VALIDATION SUMMARY")
for block139 in blocks139:
    header139 = block139.splitlines()[0] if block139 else ""
    if re.search(r"Forward (?:8|9|1[0-5])|Run (?:6[3-9]|7[0-9]|8[0-9]|10[3-9])", header139):
        relevant139 = [
            s for s in block139.splitlines()[1:]
            if re.search(r"NLL|nll|x2|half|target|scale|accepted|rollout", s, re.I)
        ]
        if relevant139:
            print(header139[:100])
            for s139 in relevant139[:1]:
                print(s139[:360])

print("\nANCHOR AND SENSITIVITY OBJECTS")
for name139, obj139 in sorted(list(globals().items())):
    if not re.search(r"(134|136)$", name139) or name139.startswith("_"):
        continue
    if isinstance(obj139, (np.ndarray, torch.Tensor)):
        print(name139, tuple(obj139.shape))
    elif isinstance(obj139, types.FunctionType):
        print(name139, "function", obj139.__code__.co_varnames[:obj139.__code__.co_argcount])
    elif isinstance(obj139, dict):
        print(name139, "dict", [
            (str(k), tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor)) else type(v).__name__)
            for k, v in list(obj139.items())[:4]
        ])
    elif isinstance(obj139, (int, float, bool)):
        print(name139, obj139)

print("\nEDIT ARCHIVES")
for path139 in ["/work/baseline77.npz", "/work/ne_reduced107.npz"]:
    with np.load(path139) as a139:
        print(Path(path139).name, [
            (k, tuple(a139[k].shape), float(a139[k]) if a139[k].ndim == 0 else round(float(np.linalg.norm(a139[k])), 4))
            for k in a139.files
        ])

with open("/work/notes.md", "a") as f139:
    f139.write(
        "\n## Run 139: bounded calibration recovery\n"
        "Run138 output was truncated before newer validation and sensitivity metadata. "
        "Retrieving only later validation summaries and exact-anchor object shapes. "
        "No weights changed. Python runs: 139; forwards: 15.\n"
    )
```
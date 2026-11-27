The records support a negative insulin effect and a positive norepinephrine effect; neither appears absent. The main difficulty is making the norepinephrine correction selective enough to preserve factual fit. I’ll inspect the dose contrasts and saved response direction before fitting another detector. Lower factual activation with retained dose sensitivity would justify a GPU test.

```python
import inspect
import re
import numpy as np
import torch
from pathlib import Path

def brief195(x):
    if isinstance(x, dict):
        return {str(k): brief195(v) for k, v in list(x.items())[:10]}
    if torch.is_tensor(x):
        return ("Tensor", tuple(x.shape))
    if isinstance(x, np.ndarray):
        return ("array", x.shape) if x.size > 6 else x.tolist()
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, len(x))
    return repr(x)[:120]

print("COUNTERFACTUAL TRAINING OBJECT")
print(brief195(cf_train192))
print("\nACTION START INFORMATION")
print("example", next(iter(start_info192.items())))
print("pure-action mask count", int(np.sum(pure192)))

print("\nSAVED NE DIRECTION AND MATRIX")
with np.load("/work/ne_candidate148.npz", allow_pickle=False) as a195:
    feature195 = a195["feature"].copy()
    direction195 = a195["direction"].copy()
    jacobian195 = a195["jacobian"].copy()
    print([(k, a195[k].shape) for k in a195.files])
    print("feature norm", float(np.linalg.norm(feature195)),
          "direction norm", float(np.linalg.norm(direction195)))
    for k195 in ["x2_slope", "choice"]:
        print(k195, a195[k195])
with np.load("/work/ne_candidate151.npz", allow_pickle=False) as a195:
    delta195 = a195["ne_delta"].copy()
    print("gamma", a195["gamma"])
    print("top singular values", np.linalg.svd(delta195, compute_uv=False)[:5])
    print("outer-product coefficient",
          float(np.sum(delta195 * np.outer(direction195, feature195)) /
                np.sum(np.outer(direction195, feature195) ** 2)))

print("\nRETAINED FEATURE / RESPONSE HELPERS")
helper195 = []
for n195, v195 in list(globals().items()):
    if inspect.isfunction(v195) and re.search(
        r"prelude|posterior|detector|ridge|response|core|contrast|atom", n195, re.I
    ):
        try:
            signature195 = str(inspect.signature(v195))
            signature195 = signature195.split("tensor(")[0][:180]
        except Exception:
            signature195 = ""
        helper195.append((n195, signature195))
for item195 in helper195[-30:]:
    print(item195)

print("\nTARGETED NE CONSTRUCTION NOTES")
note_lines195 = Path("/work/notes.md").read_text().splitlines()
for run195 in [115, 147, 148, 149, 150, 151, 154, 155, 180, 191, 192]:
    indices195 = [
        j for j, line in enumerate(note_lines195)
        if re.match(rf"## Run {run195}(?:\D|$)", line)
    ]
    if indices195:
        j195 = indices195[-1]
        print("\n".join(note_lines195[j195:j195 + 3])[:550])

with Path("/work/notes.md").open("a") as f195:
    f195.write(
        "\n## Run 195: inspect balanced dose-contrast fit inputs\n"
        "Recovered record-based signs and saved NE response construction. "
        "Inspecting paired dose contrasts for a stay-validated factual-penalized "
        "linear detector; no weights changed. Python runs: 195; forwards: 22.\n"
    )
```
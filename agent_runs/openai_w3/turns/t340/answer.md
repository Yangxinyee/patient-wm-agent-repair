The doubled-insulin response is now close to the record-derived estimate: −12.39 versus −12.96. Half-dose insulin and both norepinephrine responses remain too weak. One possibility is that generated readings change the dose-policy correction during rollout. I’ll inspect that context dependence; reducing it would support this hypothesis only if both dose directions improve without breaking factual fit.

```python
import torch, numpy as np, inspect, re, dis
from pathlib import Path

print("=== CHANGED POSTERIOR INPUT ROWS ===")
p0 = load("z_post.0.weight")
p00 = load_original("z_post.0.weight")
rowdiff293 = torch.linalg.vector_norm(p0 - p00, dim=1)
changed_rows293 = torch.where(rowdiff293 > 1e-6)[0].tolist()
print("changed rows:", changed_rows293)
print("largest:", [(int(i), round(float(rowdiff293[i]), 4))
                   for i in torch.argsort(rowdiff293, descending=True)[:12]])
for i in torch.argsort(rowdiff293, descending=True)[:6].tolist():
    print("row", i, "ctx/e/u norms",
          [round(float(torch.linalg.vector_norm(p0[i, j:j+192])), 4)
           for j in (0, 192, 384)],
          "bias", round(float(load("z_post.0.bias")[i]), 4))

print("\n=== CONTEXT AND RESPONSE ARRAYS ===")
rows293 = []
for k, v in sorted(list(globals().items())):
    if hasattr(v, "shape") and (
        re.search(r"ctx|context|policy|direction|deriv|jac|target|state", k, re.I)
        or re.search(r"(224|234|254|261|265|267|278)$", k)
    ):
        rows293.append(f"{k}: {type(v).__name__} {tuple(v.shape)}")
    elif isinstance(v, dict) and re.search(r"(224|234|254|261|265|267|278)$", k):
        rows293.append(f"{k}: dict keys {list(v)[:12]}")
print("\n".join(rows293)[:3500])

print("\n=== CARRIER POLICY NOTES ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if re.search(r"Python run(?:s)?(?: |: )(14[5-9]|15[0-4]|20[3-6]|265|267)\b", line):
        print(line[:420])

print("\n=== RESPONSE HELPER INTERFACES ===")
for name in ["context_with_flow", "response186", "response_original228",
             "response_raw234", "logic224", "original_tokens278",
             "posterior261", "parts254"]:
    f = globals().get(name)
    if inspect.isfunction(f):
        print(name, inspect.signature(f), "dependencies", f.__code__.co_names)

print("\n=== POSTERIOR CARRIER DICTIONARIES ===")
for name in ["post154", "zin154", "zb154"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                     for k, v in obj.items()])
    elif obj is not None:
        print(name, type(obj).__name__, getattr(obj, "shape", None))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 293; forwards 32. Alpha=.01 factual delta +.00185; rollouts (-12.385,+1.361,+.415,-1.193). Main insulin result is close to record-derived reference, but other directions remain weak. Investigating whether engineered posterior policy/context coupling reinterprets generated outcome changes as dose-selection information; this is a hypothesis, not established. No weights changed; checkpoint_alpha01 retains current validated fallback.\n")
```

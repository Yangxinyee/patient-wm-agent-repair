The original compensation fit predates discovery of the nonzero empty-action encoding, so its inputs need checking. I’ll verify the token construction and held-out split before refitting. Better reconstruction with corrected inputs would support this explanation; GPU factual validation will still be required.

```python
import torch
print("=== EMPTY-ACTION VECTORS ===")
for k, v in list(globals().items()):
    if "empty" in k.lower() and hasattr(v, "shape") and int(np.prod(v.shape)) <= 768:
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        print(k, tuple(a.shape), "norm", float(np.linalg.norm(a)))
print("\n=== CORRECTION VARIABLES ===")
for k, v in list(globals().items()):
    if k.endswith(("268", "269")) and not callable(v) and k not in ("text269",):
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, (list, tuple, int, float)):
            print(k, repr(v)[:180])
print("\n=== TOKEN HELPER RETURN ===")
r274 = route_tokens229(c271, Ubase215)
def describe274(x, prefix="return"):
    if isinstance(x, (tuple, list)):
        for j, y in enumerate(x):
            describe274(y, prefix + f"[{j}]")
    elif hasattr(x, "shape"):
        print(prefix, type(x).__name__, tuple(x.shape))
    else:
        print(prefix, type(x).__name__)
describe274(r274)
print("\n=== SAMPLE/SPLIT CONTRACT ===")
print("tr222", tr222.dtype, np.unique(tr222, return_counts=True))
print("active222", active222.dtype, np.unique(active222, return_counts=True))
print("tokens shapes", tuple(orig_tokens222.shape), tuple(tokens222.shape))
print("Ubase215", type(Ubase215).__name__, tuple(Ubase215.shape))
print("\n=== TOKEN HELPER CONSTANTS ===")
print(route_tokens229.__code__.co_consts)
print("\n=== SAMPLE INDEX VARIABLES ===")
for k, v in list(globals().items()):
    if k.endswith(("220", "222", "228", "229")) and hasattr(v, "shape"):
        if len(v.shape) == 1 and v.shape[0] == orig_tokens222.shape[0]:
            print(k, tuple(v.shape), str(getattr(v, "dtype", "")))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 274 — compensation input-contract audit\n"
            "- Checking actual empty-action vectors, route token returns, and stay-held sample masks.\n"
            "- Prior compensation predates the empty-action discovery; no revised fit is yet accepted.\n"
            "- No installed weights changed.\n")
```
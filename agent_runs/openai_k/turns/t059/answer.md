The architecture confirms that the value head has 17 classes, including “no value.” The inspection output was truncated before showing the failing helper, so the precise mismatch is still unresolved. I’ll inspect only that helper and its bin tables; this should distinguish a missing class-zero centre from an error in the logits.

```python
import numpy as np
import torch
import inspect, dis

print("=== RECENT OBJECT INVENTORY ===")
for k in sorted(list(globals())):
    if k.startswith(("r48", "r49", "r50", "r51", "rr48", "rr49", "rr50", "rr51")):
        v = globals()[k]
        detail = str(inspect.signature(v)) if callable(v) else str(getattr(v, "shape", ""))
        if isinstance(v, dict):
            detail = "keys=" + str(list(v)[:20])
        print(k, type(v).__name__, detail)

print("\n=== r49_readouts ===")
if "r49_readouts" in globals():
    try:
        print(inspect.getsource(r49_readouts))
    except (OSError, TypeError):
        print("names:", r49_readouts.__code__.co_names)
        print("locals:", r49_readouts.__code__.co_varnames)
        print("constants:", r49_readouts.__code__.co_consts)
        dis.dis(r49_readouts)

print("\n=== HEAD TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("attr_heads.value", "last_numeric", "last_status", "z_head", "type_emb", "query_norm")):
        print(n, SHAPES[n])

print("\n=== VALUE VOCAB ===")
print("codes:", VOCAB["codes"])
for k in ("config", "value_bin_centres", "normalisation"):
    if k in VOCAB:
        print(k, str(VOCAB[k])[:1800])
print("\n=== NOTEBOOK EARLY FINDINGS ===")
lines = Path("/work/notes.md").read_text().splitlines()
selected = [s for s in lines if any(t in s.lower() for t in
            ("impulse", "nll", "causal", "effect", "rollout", "one-hour", "target shift"))]
print("\n".join(selected[:22])[:2400])
```

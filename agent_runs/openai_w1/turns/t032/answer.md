The architecture provides a low-risk route: editing only the value head would leave the other teacher-forced outputs unchanged. I’ll inspect the retained states and head helpers to see whether they expose dose-innovation features. Those features must decode treatment variation on held-out stays, not merely reproduce the dosing policy.

```python
import re
print("=== GLOBAL NAMES ===")
print(" ".join(sorted(k for k in globals() if not k.startswith("_"))))
print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "attr_heads.value", "last_", "transition.value", "embed.numeric",
        "encode_action", "z_head", "z_prior", "gap", "query_norm",
        "type_emb", "filter", "out_norm"
    )):
        print(name, SHAPES[name])
print("\n=== RETAINED DATA / STATE OBJECTS ===")
for name, obj in list(globals().items()):
    if re.search(r"train|tun|panel|state|repair|HM|head|hidden|pred|num_best|best_|meta|vocab|sid", name, re.I):
        if isinstance(obj, (np.ndarray, pd.DataFrame, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape),
                  list(obj.columns)[:16] if isinstance(obj, pd.DataFrame) else "")
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:16])
        elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
            print(name, "function")
print("\n=== VOCAB CONFIG / CODES / NORMALIZATION ===")
for key in VOCAB:
    if key in {"config", "codes"} or "norm" in key:
        print(key, VOCAB[key])
print("\n=== FORWARD FILES ===")
for path in sorted(Path("/work").glob("fwd_*/*")):
    print(str(path))
```
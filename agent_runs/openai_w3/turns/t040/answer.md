The architecture confirms separate action and observation encoders, but the output was truncated before the helper definitions. I’ll request only the relevant helpers and vocabulary next. This should distinguish an encoding mistake from a stale cached variable before any weights change.

```python
import inspect, json, types

live_functions = {
    n: v for n, v in list(globals().items())
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"
}
print("=== HELPER SIGNATURES ===")
for n, v in live_functions.items():
    try:
        print(n, inspect.signature(v))
    except Exception:
        print(n)

print("\n=== VOCABULARY METADATA ===")
print("top-level keys:", list(VOCAB))
for k in ("config", "codes", "units", "categorical", "normalisation", "normalization", "norm"):
    if k in VOCAB:
        text = json.dumps(VOCAB[k], default=str)
        print(k, text[:2200])

print("\n=== ENCODING TENSORS ===")
for n in NAMES:
    if n.startswith(("embed.numeric", "encode.", "encode_action.", "z_post.", "z_prior.")):
        print(n, tuple(SHAPES[n]))

print("\n=== LIVE OBJECT INDEX ===")
items = []
for n, v in list(globals().items()):
    if n.startswith("_") or isinstance(v, types.ModuleType) or n in live_functions:
        continue
    if hasattr(v, "shape") and not callable(getattr(v, "shape", None)):
        items.append(f"{n}:{type(v).__name__}{tuple(v.shape)}")
    elif isinstance(v, dict):
        items.append(f"{n}:dict[{len(v)}] keys={list(v)[:5]}")
    elif isinstance(v, (list, tuple)):
        items.append(f"{n}:{type(v).__name__}[{len(v)}]")
    elif isinstance(v, str) and len(v) > 300:
        items.append(f"{n}:str[{len(v)}]")
print("\n".join(items)[:3500])

print("\n=== HELPER SOURCE AVAILABILITY ===")
for n, v in live_functions.items():
    if any(s in n.lower() for s in ("atom", "embed", "encode", "posterior", "post", "bundle")):
        try:
            text = inspect.getsource(v)
            print(n, text[:1800])
        except Exception:
            print(n, "<inspect source unavailable>")

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 35; forwards 3. Architecture confirms separate action and observation DeepSets encoders. Run 34 output was truncated before helpers; requesting bounded metadata and helper index. No edits.\n")
```
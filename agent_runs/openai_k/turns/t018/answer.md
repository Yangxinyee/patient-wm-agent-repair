The architecture confirms that actions enter a shared recurrent state, so a broad edit could disturb more than glucose and MAP. Several implementation details are still unspecified. I’ll inspect the vocabulary and retained helpers with shorter output; resolving those details should let us test a localized correction rather than guess.

```python
import json

print("=== REMAINING TENSORS ===")
for name in NAMES[-55:]:
    print(name, SHAPES[name])

print("\n=== VOCABULARY STRUCTURE ===")
for key, value in VOCAB.items():
    if "bin" in key:
        print(key, "keys:", list(value)[:15] if isinstance(value, dict) else str(type(value)))
    else:
        text = json.dumps(value, default=str)
        print(key, text[:1500])

print("\n=== GLOBAL INVENTORY ===")
items = []
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"key", "value", "items", "text"}:
        continue
    if inspect.isfunction(value):
        try:
            items.append(f"{key}{inspect.signature(value)}")
        except Exception:
            items.append(f"{key}(?)")
    elif hasattr(value, "shape"):
        items.append(f"{key}:{type(value).__name__}{tuple(value.shape)}")
    elif isinstance(value, dict):
        items.append(f"{key}:dict({list(value)[:8]})")
    elif isinstance(value, (list, tuple)):
        items.append(f"{key}:{type(value).__name__}[{len(value)}]")
print("\n".join(items))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 13: bounded vocabulary/helper inspection\n"
            "- Architecture confirms action inputs affect shared state, making factual and other-head checks essential.\n"
            "- Inspecting remaining tensor names, vocabulary conventions, and retained helper signatures.\n"
            "- Python runs: 13; forward passes: 4; no weights edited.\n")
```
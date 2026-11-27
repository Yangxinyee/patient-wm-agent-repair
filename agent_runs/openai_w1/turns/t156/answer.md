The probe changed only the dose and rate numeric columns, and the original weights are now restored. The architecture shows that these columns affect latent and recurrent state, so scaling them alone could disturb more than glucose and MAP predictions. I’ll locate the state and head caches for a targeted comparison; matching bundle keys and accurate head reconstruction are necessary before using that comparison to design an edit.

```python
import inspect, pathlib, numpy as np, torch

live_items = list(globals().items())
fn_items = [(n, o) for n, o in live_items
            if not n.startswith("_") and inspect.isfunction(o)
            and getattr(o, "__module__", "") == "__main__"]
arr_items = [(n, o) for n, o in live_items
             if not n.startswith("_") and isinstance(o, (np.ndarray, torch.Tensor))]
dict_items = [(n, o) for n, o in live_items
              if not n.startswith("_") and isinstance(o, dict)
              and n not in {"SHAPES", "VOCAB", "__builtins__"}]

print("FUNCTION NAMES:", " ".join(n for n, _ in fn_items))
print("\nARRAY NAMES:", " ".join(n for n, _ in arr_items))
print("\nDICT NAMES:", " ".join(n for n, _ in dict_items))
print("\nRELEVANT CACHE DETAILS:")
tokens = ("head", "state", "cache", "post", "atom", "replay", "enc",
          "obs", "act", "fit", "kernel", "dose", "innov", "bundle")
rows = []
for n, o in arr_items + dict_items:
    if any(t in n.lower() for t in tokens):
        desc = tuple(o.shape) if hasattr(o, "shape") else (len(o), list(o.keys())[:5])
        rows.append((n, desc))
print(*rows[-45:], sep="\n")

print("\nFORWARD FILES:")
for k in range(1, 9):
    p = pathlib.Path(f"/work/fwd_{k}")
    print(k, [f.name for f in p.iterdir()] if p.exists() else [])

print("\nSELECTED FUNCTION SIGNATURES:")
for n, o in fn_items:
    if any(t in n.lower() for t in ("head", "replay", "atom", "bundle", "view")):
        try:
            print(n, inspect.signature(o))
        except Exception:
            print(n, "?")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 142\nNumeric probe restoration confirmed. Architecture shows dose embeddings influence posterior and recurrent state, not only values. Recovering bounded cache inventory after run 141 output truncation. Counts: 142 Python runs, 8 forwards.\n")
```
The records show why dose associations alone are misleading: adjusted insulin effects are near −2 per rate unit, while norepinephrine effects are near +3 after startup. The architecture offers dose-specific paths through the posterior and recurrent update. I’ll recover the retained helpers and fitted quantities next; matching the original transition and its dose sensitivity is the prerequisite for a targeted edit.

```python
import types
snapshot = list(globals().items())

print("=== FUNCTIONS ===")
print([n for n, v in snapshot
       if isinstance(v, types.FunctionType) and not n.startswith("_")])

print("\n=== DATAFRAMES ===")
for n, v in snapshot:
    if isinstance(v, pd.DataFrame) and not n.startswith("_"):
        print(n, v.shape, list(v.columns)[:12])

print("\n=== ARRAYS/TENSORS ===")
for n, v in snapshot:
    if isinstance(v, (np.ndarray, torch.Tensor)) and not n.startswith("_") and v.ndim >= 2:
        print(n, tuple(v.shape))

print("\n=== RELEVANT DICTIONARIES ===")
for n, v in snapshot:
    if isinstance(v, dict) and not n.startswith("_") and n not in {"SHAPES", "VOCAB"}:
        if any(s in n.lower() for s in
               ("repair", "latent", "causal", "effect", "impulse", "state", "lag", "probe", "best")):
            print(n, len(v), list(v)[:12])

print("\n=== IDENTIFICATION FOLLOW-UP ===")
nt = Path("/work/notes.md").read_text()
start = nt.find("## Identification checks")
end = nt.find("## Python 29")
segment = nt[start:end]
print(segment[:3500])
```

The saved analysis includes independent state samples and record-based effect estimates, but the inventory was too long to show the recurrence helpers. I’ll narrow the inspection to those helpers and the action tensors. A useful repair should act through dose information and work on independent states, rather than merely shift average glucose or MAP.

```python
import builtins, types

print("--- Functions ---")
for name, obj in builtins.list(globals().items()):
    if isinstance(obj, types.FunctionType):
        print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount])

print("\n--- Latest persistent objects ---")
items = [(n, o) for n, o in builtins.list(globals().items())
         if not n.startswith("_") and not isinstance(o, types.ModuleType)]
for name, obj in items[-75:]:
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(name, "dict", builtins.list(obj.keys())[:12])
    elif isinstance(obj, (builtins.list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj)[:100])
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:120])

print("\n--- Action/core tensors ---")
for name in NAMES:
    if name.startswith(("core.", "encode_action.", "z_post.", "z_in.", "embed.numeric")):
        print(name, SHAPES[name])
print("\n--- Config ---", VOCAB["config"])
print("\n--- Causal estimates ---", dyn_estimates)
print("\n--- Available forward files ---")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
```

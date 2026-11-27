The recovered notes support nonzero drug effects, but also show that earlier reconstruction depended on specific register conventions and latent averaging. The output was truncated again, so I still cannot identify the current helper code. I’ll print only a compact inventory next; finding a reusable, validated reconstruction would let us calibrate without guessing.

```python
snapshot220 = list(globals().items())
print("=== ARRAYS AND TABLES ===")
print("\n".join(
    f"{nm}: {type(ob).__name__} {tuple(ob.shape)}"
    for nm, ob in sorted(snapshot220)
    if not nm.startswith("_") and isinstance(ob, (np.ndarray, torch.Tensor, pd.DataFrame))
))
print("\n=== DICTIONARIES / SEQUENCES ===")
for nm, ob in sorted(snapshot220):
    if nm.startswith("_") or nm in {"W", "WO", "BASE217", "SHAPES", "__builtins__"}:
        continue
    if isinstance(ob, dict):
        print(nm, "dict", len(ob), str(list(ob)[:8])[:140])
    elif isinstance(ob, (list, tuple)) and nm not in {"snapshot220"}:
        print(nm, type(ob).__name__, len(ob))
print("\n=== FUNCTIONS (NO DEFAULT VALUES) ===")
for nm, ob in sorted(snapshot220):
    if not nm.startswith("_") and inspect.isfunction(ob):
        print(nm, list(inspect.signature(ob).parameters))
print("\n=== NUMERIC CONFIG ===")
print(str(V["config"])[:700])
print(str(V["numeric_note"])[:1000])
print(str(V["numeric"].get("glucose_finger", V["numeric"]))[:1200])
print("\n=== RELEVANT FILES ===")
for p in sorted(Path("/work").glob("fwd_*")):
    print(p.name, [q.name for q in p.iterdir()])
```

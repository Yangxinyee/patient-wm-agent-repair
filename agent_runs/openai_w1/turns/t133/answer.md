The value head has a small editable output layer, and the exported states include the information needed to test its dose sensitivity. No candidate is validated yet. I’ll inspect the head reconstruction and recurrent-layer layout; matching the original predictions locally would let me fit and test an algebraic edit without backpropagation.

```python
import dis

def rx_show_function(name, limit=4200):
    f = globals()[name]
    print("\nFUNCTION", name)
    try:
        src = inspect.getsource(f)
        print(src[:limit])
        if len(src) > limit:
            print("[source truncated; total characters:", len(src), "]")
    except Exception:
        print("names:", f.__code__.co_names)
        print("constants:", [
            x for x in f.__code__.co_consts
            if isinstance(x, (str, int, float, tuple)) and len(str(x)) < 250
        ])
        dis.dis(f)

rx_show_function("ce_make_heads", 5000)
rx_show_function("rr_encode_atoms112", 2600)

print("\n=== RECURRENT WEIGHTS ===")
for n in NAMES:
    if n.startswith("core."):
        print(n, SHAPES[n])

print("\n=== EXPORT FILES ===")
for p in sorted(Path("/work").glob("fwd_*/*")):
    print(str(p))

print("\n=== DATA AND CONFIG ===")
print(rr_train.head(8).to_string(index=False))
print(rr_vocab["config"])
print("state4 stays:", len(np.unique(rr_state4["stay_id"])),
      "state2 stays:", len(np.unique(rr_state2["stay_id"])))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 122\nValue head has a 192-unit hidden layer and a 17-class linear output. Exact posterior inputs and exported full recurrent slots are available. Inspecting existing head reconstruction and recurrent tensor layout to build a factual-constrained, gradient-free dose-sensitivity edit. No edits. Counts: 122 Python runs, 6 forward passes.\n")
```
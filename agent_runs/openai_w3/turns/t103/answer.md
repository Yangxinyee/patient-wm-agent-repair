The architecture confirms that drug information reaches the recurrent state through the action token, while the previous posterior-only edit used a nearly ignored path. I’ll recover the exact replay and causal estimates next. The correction needs both a measurable effect on predicted readings and a near-zero effect under recorded dosing.

```python
import inspect, re

print("TENSOR NAMES")
for n in NAMES:
    if n.startswith(("core.", "attr_heads.value", "query_norm", "type_emb", "last_numeric", "z_head")):
        print(n, SHAPES[n])

print("\nKERNEL ARRAYS AND FUNCTIONS")
for n in sorted(list(globals())):
    if n.startswith("_"):
        continue
    o = globals()[n]
    if hasattr(o, "shape") and not callable(o):
        print(n, type(o).__name__, tuple(o.shape))
    elif inspect.isfunction(o):
        print(n, "function", inspect.signature(o), "uses", o.__code__.co_names)
    elif isinstance(o, str) and len(o) > 500 and n != "arch":
        print(n, "STRING", len(o), o[:120].replace("\n", " "))

print("\nEARLY NOTEBOOK")
notes92 = Path("/work/notes.md").read_text()
print(notes92[:18000])

print("\nFUNCTION SOURCES IF AVAILABLE")
for n in ("prelude_cache88", "prelude81", "fn77", "fn76"):
    if n in globals():
        try:
            print(n, inspect.getsource(globals()[n]))
        except Exception:
            print(n, "source unavailable; constants", globals()[n].__code__.co_consts)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 92; forwards 7. Confirmed action-token insertion point. "
            "Recovering exact replay/readout helpers and identification evidence before editing.\n")
```
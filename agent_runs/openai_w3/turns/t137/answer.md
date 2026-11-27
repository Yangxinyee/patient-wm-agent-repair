The attribute register actually has eight inputs, so the simplified readout formula is missing details. That could explain the mismatch. I’ll inspect the readout and state-update helpers directly; finding the omitted fields would support that explanation, while a complete implementation would point elsewhere.

```python
import dis, inspect, re
g124 = dict(globals())
print("=== NUMBERED GLOBALS ===")
desc124 = []
for key124, obj124 in sorted(g124.items()):
    if re.search(r"\d", key124) and not inspect.isfunction(obj124):
        s124 = type(obj124).__name__
        if hasattr(obj124, "shape"):
            s124 += ":" + str(tuple(obj124.shape))
        elif isinstance(obj124, dict):
            s124 += ":" + str(list(obj124)[:5])
        elif isinstance(obj124, (str, list, tuple)):
            s124 += ":len=" + str(len(obj124))
        desc124.append(key124 + "=" + s124)
print("\n".join(desc124)[:4200])
print("=== STORED SOURCE STRINGS ===")
for key124, obj124 in sorted(g124.items()):
    if isinstance(obj124, str) and len(obj124) > 300 and any(
        term124 in obj124 for term124 in ("def means", "def next_view", "def parts", "def state_update")):
        print(key124, len(obj124), obj124[:6500])
print("=== READOUT HELPER IMPLEMENTATIONS ===")
for name124 in ("means105", "next_view105", "parts105", "isolated_record"):
    f124 = g124[name124]
    print("\nFUNCTION", name124, "GLOBALS", f124.__code__.co_names)
    try:
        print(inspect.getsource(f124))
    except Exception:
        text124 = dis.Bytecode(f124).dis()
        print(text124[:2800])
```
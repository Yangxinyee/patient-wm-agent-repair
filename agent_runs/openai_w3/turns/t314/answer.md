The repair uses very large posterior and compensation weights. That raises an off-record stability concern, though it does not prove the cause of the asymmetry. I’ll inspect the cached dose contrasts and replay interfaces so we can test a smaller correction around the original model instead of rescaling the existing transport.

```python
print("INTERFACES")
for name in ["posterior153", "posterior261", "prelude81", "prelude_parts145",
             "parts254", "response_original228", "response_raw234",
             "state_update82", "next_view105", "normalize267"]:
    f = globals()[name]
    print(name, str(inspect.signature(f)), "GLOBALS:", f.__code__.co_names)

print("\nTOKEN, ACTION, AND REPLAY CACHES")
for n, v in snapshot274:
    if re.search(r"tokens|^tok|^U\d|^u\d|^C\d|^E\d|^post|^zin|^zb|states|mem|^(h|out|flow|slot)\d|^W$|^T$|267|255|254|228", n):
        if hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict", list(v)[:5])
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, len(v))

print("\nCOMPACT POSTERIOR RECONSTRUCTION")
for name in ["posterior261", "posterior153", "prelude81"]:
    print(name)
    instructions277 = list(dis.get_instructions(globals()[name]))
    text277 = "\n".join(f"{i.offset} {i.opname} {i.argrepr[:95]}" for i in instructions277)
    print(text277[:1900])

print("\nLOCAL-CONTRAST FINDINGS")
for line in lines275:
    if re.search(r"Python run (21[3-9]|22[0-5])\b", line):
        print(line[:360])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 277; forwards 24. Existing posterior/FFN edits have very large norms, raising an off-record stability hypothesis (not established). Recovering original-token dose contrasts and replay interfaces to test a simpler correction around original weights. Current validated checkpoint273 unchanged.\n")
```
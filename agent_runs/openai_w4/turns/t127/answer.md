The architecture has separate action and reading pathways, so a localized repair remains possible. The latest leakage test does not yet establish that it preserves factual fit. I’ll inspect the retained candidate and validation arrays next; these should let us test a concrete edit rather than rely on isolated channel responses.

```python
import inspect, re
print("=== FUNCTIONS ===")
for nm, ob in sorted(list(globals().items())):
    if inspect.isfunction(ob) and getattr(ob, "__module__", "") == "__main__":
        try:
            ss = str(inspect.signature(ob))
        except Exception:
            ss = "?"
        print(nm, ss[:180])

print("=== VARIABLE NAMES ===")
vv = []
for nm, ob in sorted(list(globals().items())):
    if nm.startswith("_") or inspect.ismodule(ob) or inspect.isfunction(ob):
        continue
    if nm in {"notes_text", "NAMES", "SHAPES", "VOCAB", "In", "Out"}:
        continue
    if hasattr(ob, "shape"):
        vv.append(f"{nm}:{tuple(ob.shape)}")
    elif isinstance(ob, dict):
        vv.append(f"{nm}:dict[{len(ob)}]")
    elif isinstance(ob, (list, tuple)):
        vv.append(f"{nm}:{type(ob).__name__}[{len(ob)}]")
    elif isinstance(ob, (int, float, str, bool)):
        vv.append(f"{nm}:{str(ob)[:35]}")
print(" | ".join(vv)[:5800])
print("=== RECENT NOTEBOOK HEADINGS ===")
nt = open("/work/notes.md").read()
print("\n".join(re.findall(r"(?m)^## .*$", nt)[-14:]))
print("=== CAUSAL / KERNEL NOTES ===")
selected = [
    ln for ln in nt.splitlines()
    if any(x in ln.lower() for x in [
        "deconvol", "kernel", "distributed", "longitudinal", "instrument",
        "immediate", "placebo", "identified", "identification", "sustained"
    ])
]
print("\n".join(selected[:20])[:2400])
```

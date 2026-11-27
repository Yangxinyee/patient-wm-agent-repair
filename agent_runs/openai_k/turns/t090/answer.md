The earlier records support roughly −2 glucose units per insulin-rate unit and +3 arterial-MAP units per norepinephrine-rate unit after one hour, with longer persistence for glucose. The latest carrier tests did not recover those responses. I’ll inspect the retained implementation and candidate objects next; this should distinguish a missing dose signal from a weak downstream readout.

```python
import inspect
print("NOTES: PRIOR IMPLEMENTATION")
print(notes83[6000:10400])
print("\nRELEVANT OBJECTS")
keys84 = [k for k in list(globals()) if any(s in k.lower() for s in
          ("core", "prior", "carrier", "bridge", "81", "82", "66", "78", "80", "target", "dose"))]
for k in sorted(keys84):
    v = globals()[k]
    if callable(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = ""
        print(k, "FUNCTION", sig)
    elif hasattr(v, "shape"):
        print(k, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:10])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v))
print("\nTENSOR NAMES")
print("\n".join(n + " " + str(SHAPES[n]) for n in NAMES
                if any(s in n for s in ("z_post", "z_prior", "z_in", "z_head",
                                       "encode_action", "attr_heads.value", "core.c_out"))))
```

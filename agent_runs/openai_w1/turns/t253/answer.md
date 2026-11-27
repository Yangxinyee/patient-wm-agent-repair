The earlier 5× numeric edit passed one factual check but produced weak, asymmetric rollout effects, so it is only a fallback. The current candidate uses two dedicated marker channels. I’ll check their dimensions and finite-response calculations before installing them; a valid repair must affect predicted readings, not merely hidden states.

```python
import inspect, dis, io, numpy as np

print("RECENT INVENTORY, REMAINING:")
for _n225, _v225 in sorted(list(globals().items())):
    if (_n225.startswith("_") or
        not re.search(r"215|220|221|222", _n225) or _n225 < "logits"):
        continue
    print(_describe224(_n225, _v225))

print("\nRELAY OBJECT NAMES:")
print(", ".join(sorted(n for n,v in list(globals().items())
                     if not n.startswith("_") and re.search(r"relay|rp_|rr_|marker",n))))

print("\nFUNCTION DETAILS:")
for _fn225 in ["transition214", "finite_readout221", "norm215", "fixedpost215", "install_carrier200"]:
    _obj225 = globals().get(_fn225)
    if not callable(_obj225):
        continue
    print(_fn225, inspect.signature(_obj225))
    print("names:", _obj225.__code__.co_names)
    print("constants:", repr(_obj225.__code__.co_consts)[:700])
    if _fn225 == "finite_readout221":
        _buf225 = io.StringIO()
        dis.dis(_obj225, file=_buf225)
        print(_buf225.getvalue()[:3600])

print("\nKEY ARRAY SCALES:")
for _n225 in ["B221", "M221", "T221", "A215", "C215", "U215", "V215", "Ff215", "F222", "Q222", "Phi222"]:
    _a225 = np.asarray(globals()[_n225])
    print(_n225, _a225.shape,
          "rms", np.sqrt(np.mean(_a225**2)),
          "range", (np.nanmin(_a225), np.nanmax(_a225)))
print("\nCOUNTERFACTUAL CACHE KEYS:")
for _key225 in [("insulin",2.0), ("norepinephrine",2.0), ("norepinephrine",3.0)]:
    print(_key225, {k: (v.shape if hasattr(v,"shape") else type(v).__name__)
                   for k,v in cf215[_key225].items()})

with open("/work/notes.md","a") as _f225:
    _f225.write("\n## Run 225\nRecovered 5x numeric fallback: factual +.00208 on one batch but weak/asymmetric rollout response. Inspecting marker/relay dimensions and finite-response helper before attempting a dedicated output route. No weights changed.\n")
```
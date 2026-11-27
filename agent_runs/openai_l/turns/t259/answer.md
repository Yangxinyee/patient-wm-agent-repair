The accepted kernels imply delayed glucose lowering and a shorter MAP-raising response. The failed candidate replaced only two value-head units, so independent factual coverage may still make that approach workable. I’ll inspect how its dose features and targets were constructed; if the features are unstable even on factual states, I’ll revise them rather than merely refit the output layer.

```python
import dis
print("=== NONLINEAR FEATURE CONSTRUCTION ===")
print(dis.Bytecode(make_arch215).dis()[:6100])
print("\n=== FITTING DATA ===")
for k in ["headbase214", "Xraw204", "Hf214", "offset204",
          "family204", "directions207", "pruneorder214", "paired214"]:
    v = globals().get(k)
    if isinstance(v, dict):
        print(k, "keys", list(v.keys())[:12])
        for kk, vv in list(v.items())[:2]:
            if isinstance(vv, dict):
                print(" ", kk, {q: (getattr(w, "shape", None) or type(w).__name__)
                                for q, w in vv.items()})
            else:
                print(" ", kk, getattr(vv, "shape", None),
                      str(vv)[:150] if not hasattr(vv, "shape") else "")
    else:
        print(k, getattr(v, "shape", None),
              str(v)[:150] if not hasattr(v, "shape") else "")
print("\n=== BASE VALUE-HEAD TENSORS ===")
for k in NAMES:
    if k.startswith("attr_heads.value_bin"):
        print(k, SHAPES[k])
print("Family gate factual range",
      np.quantile(Xraw204 @ gatecanon200, [0, .01, .5, .99, 1]))
```

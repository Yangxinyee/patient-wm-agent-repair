After refreshing the cache, the private dose signals are approximately proportional to dose. However, halving some rates changes ordinary action features much more than doubling them. That could make the head’s exposure proxy cancel the halving effect. I’ll inspect the proxy and its dependencies; this hypothesis predicts a dose-dependent proxy change despite nearly linear private transport.

```python
print("=== HEAD SPECIFICATION OBJECTS ===")
print([n for n, specs in _r195_spec_objects])
_r196_specs = _r195_spec_objects[-1][1]
for _r196_sp in _r196_specs:
    print("\nSensor:", _r196_sp["event"], "drug:", _r196_sp["drug"])
    for _r196_k, _r196_v in _r196_sp.items():
        if hasattr(_r196_v, "shape"):
            print(_r196_k, "shape", tuple(_r196_v.shape),
                  "norm", float(torch.as_tensor(_r196_v).float().norm()))
        elif isinstance(_r196_v, dict):
            print(_r196_k, {
                k: tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100]
                for k, v in _r196_v.items()
            })
        elif isinstance(_r196_v, (list, tuple)):
            print(_r196_k, "length", len(_r196_v),
                  [tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100]
                   for v in _r196_v[:4]])
        else:
            print(_r196_k, repr(_r196_v)[:220])

print("\n=== EARLIER FUNCTION SIGNATURES ===")
for _r196_n, _r196_o in _r194_functions[:31]:
    print(_r196_n, str(_r194_inspect.signature(_r196_o)))

print("\n=== KEY REPLAY CACHE CONSISTENCY ===")
for _r196_n in (
    "core.flow.a", "encode.norm.weight", "encode.norm.bias",
    "encode_action.norm.weight", "encode_action.norm.bias",
    "attr_heads.value_bin.0.weight"
):
    print(_r196_n, "W0 versus installed max difference",
          float((W0[_r196_n] - load(_r196_n)).abs().max()))
print("Saved-state dimensions:", s4["x"].shape)
print("Replay return:", tuple(pred188.shape))
print("Captured replay locals:", list(loc188))
print("Head input and transition:", tuple(_r149_wc32.shape),
      tuple(_r149_transition.shape))

print("\n=== HEAD DATA / MASK OBJECT NAMES ===")
_r196_matches = []
for _r196_n, _r196_o in list(globals().items()):
    if (
        _r196_n.startswith(("_r147_", "_r148_", "_r149_"))
        or _r196_n in ("readings", "reading4", "readings4", "cache133", "head_cache")
    ):
        if hasattr(_r196_o, "shape"):
            _r196_matches.append((_r196_n, tuple(_r196_o.shape)))
        elif isinstance(_r196_o, dict):
            _r196_matches.append((_r196_n, list(_r196_o)[:12]))
        elif isinstance(_r196_o, (list, tuple)):
            _r196_matches.append((_r196_n, "length " + str(len(_r196_o))))
        elif isinstance(_r196_o, (int, float, str)):
            _r196_matches.append((_r196_n, repr(_r196_o)[:80]))
print(_r196_matches[:55])
```
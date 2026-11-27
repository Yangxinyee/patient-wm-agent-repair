Correcting memory boundaries reduced ordinary-state error, but did not produce a faithful reconstruction. The private-coordinate error also worsened. I’ll compare the old and new calculations stage by stage and check which weights their helpers actually read. Identical inputs and weights would rule out stale helper namespaces; differences would locate the inconsistency.

```python
import sys
print("=== HELPER NAMESPACE CONSISTENCY ===")
for name in ("wm_step_saved", "wm_step_boundary176", "wm_layer",
             "wm_attention", "wm_norm", "wm_mlp", "wm_lin",
             "wm_set", "action_atom_exact"):
    f = globals()[name]
    g = f.__globals__
    w = g.get("W0", {})
    delta = []
    for key in ("core.c_out.weight", "encode_action.norm.weight", "embed.code.weight"):
        if key in w:
            delta.append((key, float(torch.max(torch.abs(
                torch.as_tensor(w[key]) - torch.as_tensor(W0[key]))))))
    print(name, "root globals", g is globals(), "W0 same", w is W0,
          "e same", g.get("wm_e4") is wm_e4, "state same", g.get("s4") is s4,
          "weight differences", delta)
    if name in ("wm_layer", "wm_attention", "wm_step_saved"):
        print(" settings", g.get("wm_memory_mode"), g.get("wm_kv_layout"),
              "lin helper same", g.get("wm_lin") is wm_lin)

def capture177(f, *args, **kwargs):
    local = {}
    def trace(frame, event, arg):
        if frame.f_code is f.__code__:
            if event == "return":
                local.update(frame.f_locals)
            return trace
        return None
    prior = sys.gettrace()
    sys.settrace(trace)
    try:
        result = f(*args, **kwargs)
    finally:
        sys.settrace(prior)
    return result, local

old177, locals_old177 = capture177(wm_step_saved, 3, False, "input", "seconds")
new177, locals_new177 = capture177(wm_step_boundary176, 3, False, "input", "seconds")
print("\n=== STAGE COMPARISON AT j=3 ===")
for a, b in (("h4", "h"), ("u4", "u"), ("z4", "z"), ("tok4", "tokens"),
             ("E4", "E"), ("mem4", "memory"), ("ages4", "ages"),
             ("age4", "age"), ("b4", "b"), ("delta4", "delta"), ("abar4", "abar")):
    x, y = locals_old177.get(a), locals_new177.get(b)
    if torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape:
        print(a, b, "RMSE", float(torch.sqrt(torch.mean((x-y)**2))),
              "max", float(torch.max(torch.abs(x-y))) if x.numel() else 0.0)
    else:
        print(a, b, "shapes", getattr(x, "shape", None), getattr(y, "shape", None))
truth177 = torch.from_numpy(s4["x"][3, 192:960]).reshape(1, 4, 192)
for label, pred in (("old", old177), ("new", new177)):
    print(label, "ordinary RMSE",
          float(torch.sqrt(torch.mean((pred[:,:,ordinary176]-truth177[:,:,ordinary176])**2))),
          "private coordinates", pred[0,0,[61,21,15,66]].tolist())
print("truth private", truth177[0,0,[61,21,15,66]].tolist())
```

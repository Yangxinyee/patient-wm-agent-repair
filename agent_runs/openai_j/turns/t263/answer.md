The recovered estimates support a negative insulin effect and a positive norepinephrine effect, but the norepinephrine estimate relies on stronger assumptions about treatment selection. The current correction responds much more strongly to halving than doubling. I’ll inspect its gating and input contrasts to see whether that asymmetry is repairable; if it reflects missing dose information, another head-only correction may not help.

```python
import dis

print("EXACT MAP CONSTRUCTION")
fn233 = head177_make
print("arguments/locals:", fn233.__code__.co_varnames)
print("constants:", fn233.__code__.co_consts)
instructions233 = [
    f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn233)
]
print("\n".join(instructions233)[:4300])

print("\nCONSTRUCTION OBJECTS")
for name in ["head177_names", "head177_free", "trial158_free",
             "trial158_gate", "wm_v151", "v208", "center208", "scale208",
             "head177_compressed", "mapcoef225", "mapcoef224"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                      for k, v in obj.items()])
    elif hasattr(obj, "shape"):
        a = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
        print(name, a.shape, "range", float(a.min()), float(a.max()),
              "norm", float(np.linalg.norm(a)))
    else:
        print(name, repr(obj)[:200])

print("\nCURRENT LOCAL HEAD OUTPUT")
weights233 = {name: load(name) for name in head177_names}
pred233 = head177_eval(wm_train145, weights233)
print(type(pred233).__name__)
if isinstance(pred233, dict):
    print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
           for k, v in pred233.items()])

print("\nCOUNTERFACTUAL INPUT CONTRASTS")
for key in [("norepinephrine", 0.5), ("norepinephrine", 2.0)]:
    mc = wm_cf146[key]
    print(key, "active", np.asarray(mc["active"]).sum(),
          "delta_rate range", np.asarray(mc["delta_rate"]).min(),
          np.asarray(mc["delta_rate"]).max(),
          "inputs", tuple(mc["inputs"].shape))
    dx = (mc["inputs"] - wm_train145["inputs"]).mean(0).numpy()
    for event in ["map_arterial", "map_cuff"]:
        mask = (events206 == event) & np.asarray(mc["active"], dtype=bool)
        print(event, "n", mask.sum(),
              "mean contrast norm", np.linalg.norm(dx[mask], axis=1).mean())
```
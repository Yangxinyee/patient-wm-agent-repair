The inspection stopped because an event-label array was treated as numeric; no weights changed. I’ll rerun it with type-safe summaries. The key check remains whether a candidate strengthens the arterial response, rather than merely improving another local score.

```python
import dis, inspect

def _summary223(_v):
    if isinstance(_v, dict):
        return {str(_k): _summary223(_x)
                for _k, _x in list(_v.items())[:10]}
    if isinstance(_v, torch.Tensor):
        return ("Tensor", tuple(_v.shape))
    if isinstance(_v, np.ndarray):
        if _v.dtype.kind in "biufc":
            return ("array", _v.shape,
                    float(np.min(_v)), float(np.max(_v)))
        _u, _c = np.unique(_v, return_counts=True)
        return ("array", _v.shape, list(zip(_u.tolist(), _c.tolist()))[:12])
    if isinstance(_v, (list, tuple)):
        return (type(_v).__name__, len(_v),
                [_summary223(_x) for _x in _v[:5]])
    return repr(_v)[:150]

print("=== SCORE DEFINITIONS ===")
for _n in ["events206", "labels217", "target217", "asymmetric217",
           "head177_names", "trial158_gate", "wm_train145", "wm_tune145",
           "wm_cf146", "local216", "map192_nebase", "base217", "tbase217"]:
    print(_n, _summary223(globals()[_n]))

print("screen constants:", screen_map218.__code__.co_consts)
print("scalar evaluator signature:", inspect.signature(scalar_eval208))
print("scalar evaluator dependencies:", scalar_eval208.__code__.co_names)

print("\n=== RESPONSE-SCORING BYTECODE ===")
_r223_bc = list(dis.Bytecode(screen_map218))
_r223_start = next(
    (_i for _i, _ins in enumerate(_r223_bc)
     if _ins.opname == "LOAD_GLOBAL" and _ins.argval == "wm_cf146"), 0)
for _ins in _r223_bc[_r223_start:]:
    print(_ins.offset, _ins.opname, _ins.argrepr)

print("\n=== PRIOR ACTION-EDIT RESULTS ===")
_r223_hits = []
for _s in re.split(r"(?m)(?=^## )", open("/work/notes.md").read()):
    if re.search(r"(Forward [789]|posterior edit|action-input)", _s[:150], re.I):
        _r223_hits.append(_s[:800])
print("\n".join(_r223_hits[:4]))
```
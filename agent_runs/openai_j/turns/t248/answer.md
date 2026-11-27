The installer exactly reproduces the active weights, so we have a reliable baseline. The record-based norepinephrine estimates favor a positive effect, though its magnitude remains uncertain. I’ll check what the frontier’s “A” and “C” scores measure; only a genuine improvement in the arterial response would justify testing that candidate.

```python
print("=== SCORE DEFINITIONS ===")
for _n in ["events206", "labels217", "target217", "asymmetric217",
           "head177_names", "trial158_gate"]:
    _v = globals()[_n]
    if isinstance(_v, np.ndarray):
        print(_n, _v.shape, "range", float(_v.min()), float(_v.max()))
    else:
        print(_n, repr(_v)[:1200])
print("screen constants:", screen_map218.__code__.co_consts)
print("scalar evaluator signature:", inspect.signature(scalar_eval208))
print("scalar evaluator dependencies:", scalar_eval208.__code__.co_names)

def _summary222(_v, _depth=0):
    if isinstance(_v, dict):
        return {str(_k): _summary222(_x, _depth+1)
                for _k, _x in list(_v.items())[:12]}
    if isinstance(_v, (np.ndarray, torch.Tensor)):
        return (type(_v).__name__, tuple(_v.shape))
    if isinstance(_v, (list, tuple)):
        return (type(_v).__name__, len(_v),
                [_summary222(_x, _depth+1) for _x in _v[:2]])
    return repr(_v)[:120]

for _n in ["wm_train145", "wm_tune145", "wm_cf146", "local216",
           "map192_nebase", "base217", "tbase217"]:
    print(_n, _summary222(globals()[_n]))

print("\n=== SCREENING RESPONSE BYTECODE ===")
_r222_bc = list(dis.Bytecode(screen_map218))
_r222_start = next(
    (_i for _i, _ins in enumerate(_r222_bc)
     if _ins.opname == "LOAD_GLOBAL" and _ins.argval == "wm_cf146"), 0)
for _ins in _r222_bc[_r222_start:]:
    print(_ins.offset, _ins.opname, _ins.argrepr)

print("\n=== PRIOR ACTION-EDIT RESULTS ===")
for _s in re.split(r"(?m)(?=^## )", open("/work/notes.md").read()):
    if re.search(r"(Forward 7|Forward 8|Forward 9|posterior edit|action-input)", _s[:150], re.I):
        print(_s[:1100])
```
The records support persistent glucose lowering from insulin and shorter-lived MAP increases from norepinephrine; the original rollouts largely miss both. The saved edit uses large coefficients, so I’ll check whether its small factual variance holds for each slot, where the weights actually act. Large slot-level perturbations would rule out this candidate before a GPU test.

```python
print("CACHE SHAPES")
for k in ['cache136','hbase136','hh136','E136','XF134','FC134','F133',
          'mem136','mask136','age136','idx136','indices136','sensors136']:
    if k not in globals(): continue
    v = globals()[k]
    if isinstance(v, (tuple,list)):
        print(k, [(type(x).__name__, tuple(x.shape) if hasattr(x,'shape') else str(x)[:80]) for x in v])
    elif hasattr(v, 'shape'):
        print(k, tuple(v.shape))
    else: print(k, str(v)[:200])
print("\nHELPER IMPLEMENTATIONS")
import dis
for k in ['prelude_cache88','parts105','state_update82','next_view105','readout134']:
    f = globals()[k]
    print("\n", k)
    try:
        print(inspect.getsource(f))
    except Exception:
        print("argument names:", f.__code__.co_varnames[:f.__code__.co_argcount])
        print("referenced names:", f.__code__.co_names)
        if k in ['prelude_cache88','readout134']:
            dis.dis(f)
print("\nSMALL PILOT ARRAYS")
for k in ['scales136','T136','cols136','sensors134','sensors135','target135']:
    if k in globals(): print(k, globals()[k])
for i, x in enumerate(cache136):
    if isinstance(x, torch.Tensor):
        print("cache", i, "mean norm", float(x.norm(dim=-1).mean()), "shape", tuple(x.shape))
        if x.shape[-1] == 768:
            scores = (x.double() - torch.tensor(mu136)) @ torch.tensor(B136)
            print("dose scores by slot: mean", scores.mean(0).numpy(), "SD", scores.std(0).numpy(),
                  "max absolute", scores.abs().amax(0).numpy())
print("J136 norm", np.linalg.norm(J136), "CURRENT EDITS", edited())
```

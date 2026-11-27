Two things from this run.

First, the cost file's three rows agree on the cheapest h dims. Dims 100 and 155 are cheapest in every row (slightly negative cost), followed by 151, 128 and 144. These are my candidates for the memory, staging and reference dims.

Second, the reading codes are glucose_finger (6), glucose_lab (10), map_arterial (7) and map_cuff (5); the drugs are codes 8 and 9. So the readout gate must turn the insulin tilt on for codes 6 and 10 only, and the NE tilt on for codes 7 and 5 only.

`step_b(..., ret=True)` exposes the prelude internals I need (tokens, norm1 output, attention weights, E, hf, sp, abar, hn). The functions I expect to build on are `tf_eval`, `build_ds` and `rollout_stay`, which fell past the truncated list. Next I check that they exist, get their signatures, and disassemble `tf_states`. That shows how bundles, memory and dt feed `step_b`, so I can write a teacher-forced loop that collects the attention weights and value vectors at every bundle. I need those to fit the rate-staging decoder. If `tf_eval` or `build_ds` is missing, I will rebuild them from `tf_states` and `head_value`.

```python
import dis, io, contextlib
fnames = [k_ for k_,v_ in globals().items() if callable(v_) and hasattr(v_,'__code__') and not k_.startswith('_')]
print('ALL FUNCS', fnames)
for nm_ in ['tf_eval','build_ds','rollout_stay','prep_stay','tf_stay','stay_inputs','rates_by_bundle','true_shift']:
    if nm_ in globals():
        co = globals()[nm_].__code__
        print(nm_, co.co_varnames[:co.co_argcount], '| locals', co.co_varnames[co.co_argcount:][:40])
buf = io.StringIO()
with contextlib.redirect_stdout(buf): dis.dis(tf_states)
lines = [l_ for l_ in buf.getvalue().splitlines() if any(t_ in l_ for t_ in ['LOAD_','STORE_','CALL','BINARY_SUBSCR','BUILD','RETURN','FOR_ITER'])]
txt = '\n'.join(' '.join(l_.split()[-3:]) for l_ in lines)
print(txt[:3500])
```
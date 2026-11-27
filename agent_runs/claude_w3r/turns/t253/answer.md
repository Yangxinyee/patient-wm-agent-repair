The kernel has a full local simulator that reads weights from the dict `Wt` (`core_step_v`, `step_at`, `run_from`, `prelude_att`, `pred_mean_gf`, `wc_row`). I can test any weight edit locally by temporarily replacing entries of `Wt`, without spending forward passes.

Next I will list the 181 tensor names and shapes, and the keys of the small dicts left from earlier work (`Wpre`, `P`, `PA`, `cands`, `best`, `gg`). That identifies exactly which tensors carry prelude head 2's value and output maps, `core.b_in`, the output norm and the value head. It also shows whether earlier runs already stored candidate directions I can reuse.

```python
print(' | '.join(f"{n}{tuple(SHAPES[n])}" for n in NAMES))
for nm in ['Wpre','P','P_','PA','S1','T_comb','cands','best','gg','ns2','CAP','ARR','TOK','PA','att_tab','mean_by']:
    v=globals().get(nm)
    if isinstance(v,dict):
        print(nm, '->', ', '.join(f"{k}:{(tuple(x.shape) if hasattr(x,'shape') and hasattr(x,'ndim') else (x if isinstance(x,(int,float,str)) else type(x).__name__))}" for k,x in list(v.items())[:40]))
```
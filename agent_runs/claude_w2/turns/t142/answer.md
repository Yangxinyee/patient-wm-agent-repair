ARCHITECTURE.md settles two things. Memory M holds the bundle vectors e (measurements and visit events only), and its keys get `core.age(log(1+age))`. The readout is the predicted mean, Σ p(bin)·centre, edited minus factual, so it is bounded by the bin centres (glucose_finger 72..271).

The run-133 variants (unit, kind0, numbias, …) were probably tests of how the memory entries or the age term are built. That suggests the core step still had a mismatch of ~2–3 in some norm. I'll list the defined functions and the stored search results (`errs`, `best`, `res_`, `variant`) to see exactly what was being matched and how close it got.

```python
fn_names=[k for k,v in globals().items() if isinstance(v,types.FunctionType)]
print("FUNCS:", fn_names)
for nm_ in ['variant','gm','lay','au','gapmode','half','ty','top','kk','fn','sk','eps','gscale','perm','blocks','unit_row','side','act','assign','best_s','stat_opts']:
    if nm_ in globals(): 
        v_=globals()[nm_]; print(nm_,'=',repr(v_)[:200])
print('errs',[ (e if not hasattr(e,'shape') else e.shape) for e in errs][:15])
print('best[:6]',best[:6])
print('res_[:6]',res_[:6])
```
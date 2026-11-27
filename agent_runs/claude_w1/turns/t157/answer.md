run_stay, core_step and prelude_attn all read weights from a global dict Wo by tensor name. So I can test any weight edit locally and exactly: patch the Wo entries, re-run, restore. The head replica head_mean takes rows of head inputs. Next I need two things:
- the gen_samples pipeline (stay → bundles → states → head inputs at each reading), to measure an edit's effect on teacher-forced predictions at lags 1, 2 and 3;
- the prelude weight names and shapes, plus the layout of a bundle dict, to build the action-token injection.

This run inspects these, keeping the output short.

```python
c = gen_samples.__code__; print('gen_samples names', ' '.join(c.co_names)[:400]); print(' locals', ' '.join(c.co_varnames)[:300])
c = wc_vec.__code__; print('wc_vec args', c.co_varnames[:c.co_argcount], 'names', ' '.join(c.co_names)[:200])
print([ (k, tuple(np.shape(v))) for k,v in Wo.items() if 'prelude' in k][:30])
s0 = setA[0]; dfs = tu[tu.stay_id==s0]
B0 = bundles_of_stay(dfs, RU if 'RU' in globals() else None)
print('nb', len(B0), 'keys', list(B0[1].keys()), [ (k, np.shape(v)) for k,v in B0[1].items() if not isinstance(v,(int,float,str))][:8])
```
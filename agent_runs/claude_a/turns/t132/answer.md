The cost file is a (3, 192) array, not a vector, so my indexing failed. It probably holds three per-dim cost measures from an earlier run, for example different ways of repurposing a dim or different reading types. I'll print each row's summary and cheapest dims and run the rest of the inspection again. Dims that are cheap in all three rows are the candidates for the memory, staging and reference dims.

```python
dc = np.load('/work/dimcost.npy')
for r_ in range(3):
    o_ = np.argsort(dc[r_]); print('row',r_, 'pct', np.round(np.percentile(dc[r_],[0,25,50,75,100]),5), 'cheapest', o_[:10].tolist(), np.round(dc[r_][o_[:10]],5))
tot_ = dc.sum(0); o_ = np.argsort(tot_); print('sum cheapest', o_[:15].tolist(), np.round(tot_[o_[:15]],5))
print('corr rows', np.round(np.corrcoef(dc),2))
print('VOCAB keys', list(VOCAB.keys()))
cd_ = VOCAB['codes'] if 'codes' in VOCAB else None
print('codes', cd_ if not isinstance(cd_, dict) else list(cd_.items())[:20])
print('f2 cols', list(f2.columns)); print(f2.head(3).to_string()[:500])
print(f2.iloc[:,1].value_counts().head(10) if f2.shape[1]>1 else '')
for fn_ in [step_b, tf_states, MH, rollout_gen, head_value]:
    co = fn_.__code__
    print(fn_.__name__, 'args', co.co_varnames[:co.co_argcount], '| locals', co.co_varnames[co.co_argcount:][:50], '| names', co.co_names[:50])
```
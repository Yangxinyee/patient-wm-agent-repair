The inventory lists many arrays from runs 100–125 that the notebook doesn't describe. Several of them suggest I had already started the "residual on lagged dose innovation" step in the notebook's PLAN:
- `mu_all`/`res_all`/`rs_all`/`rt_all` (153,015 readings: model means and residuals, probably from my exact local replica over training stays);
- `NUm`/`NUNm`/`Dm` (30,000 × 26 per-stay-hour dose/innovation matrices);
- `hI_full`/`hN_full` (impulse responses);
- `results_core`, `results`.

Next I will print short summaries of these, plus the head of the notebook and whether function sources can be retrieved. Then I can tell which estimates exist and how far that step got. If `res_all` regressed on lagged innovations already gives the missing coefficients c_k, the next step is building the edit. If not, I compute them.

```python
import numpy as np
print(s[:max(0, len(s)-8000)][:2500])
print('-----')
for nm_ in ['hI_full','hN_full','hI','hins','hi','hn','TGi','TGn','TIi','TIn','cand','bet','se']:
    print(nm_, np.round(np.asarray(g[nm_]),3).tolist())
print('results_core', [str(r)[:110] for r in results_core[:4]])
print('results', [str(r)[:110] for r in results[:4]])
print('keep0', str(keep[0])[:200]); print('allrows0', str(allrows[0])[:200])
print('SAMP0', [ (type(x).__name__, getattr(x,'shape',x)) for x in SAMP[0]][:10])
print('F0', F[:2].round(3).tolist(), 'Lnu0', Lnu[:2].round(3).tolist())
print('code_ uniq', np.unique(code_, return_counts=True))
for c_ in np.unique(code_):
    mm_ = code_==c_
    print(c_, 'res mean %.3f sd %.3f'%(res_all[mm_].mean(), res_all[mm_].std()), 'y mean %.2f'%y_[mm_].mean(), 'mu mean %.2f'%mu_all[mm_].mean())
print('am/fm', am[:5], fm[:5], 'rs_all n stays', len(np.unique(rs_all)))
print('funcs', [f[:60] for f in funcs[:3]], len(funcs))
try:
    print(inspect.getsource(run_stay)[:300])
except Exception as ex: print('no source', ex)
```
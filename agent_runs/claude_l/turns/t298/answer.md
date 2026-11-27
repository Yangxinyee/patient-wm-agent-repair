Function sources can't be recovered: they were defined in exec'd cells. The /work files and the saved data still tell most of the story:
- /work/inj.py holds `bstep_z`, my exact core step with a pluggable z-token function (`zfn`), and `tf_stay_z`, an exact teacher-forced replica that uses it.
- tf_train.npz holds teacher-forced contexts and action encodings for about 57k bundles from 4,000 training stays.
- Later variables point to the pieces I was building: random gelu features of ctx for the expected dose (`A_rf`, `b_rf`, `Frf`), linear programs for gating units (`linprog`, `Aub`), and Jacobians of the prelude output with respect to the z token (`JEk`).

The variable list was still cut off, and the newest variables are the most informative. Next I'll print the rest of it, plus the signatures and referenced names/constants of the key builder and evaluation functions. Function signatures don't need source code. That should be enough to resume the z-token construction without rebuilding it. If the builder functions turn out unusable, I'll write one compact builder myself from the data that's already in memory.

```python
import inspect
s_all = ' | '.join(data_items)
print('DATA(rest2):', s_all[11000:15000])
def sig(fn):
    try: return str(inspect.signature(fn))
    except Exception: return '?'
for name in ['mkW3','mkW2','zpost_np','mk_zfn','make_zfn','zf','zbase','zrec','copyz','copyK','rollout','run_batch','compare','tf_run','tf_all','tf_collect','true_shift','u_syn','gnp','desc','rres','nextmeans','regp','cvpred','cap_prelude','bstep_c','pred4','report','collect_drug','fitr']:
    fn = globals().get(name)
    if fn is None: print(name, 'MISSING'); continue
    co = fn.__code__
    consts = [c for c in co.co_consts if isinstance(c, str) and len(c) < 40][:8]
    print(f"{name}{sig(fn)} N={list(co.co_names)[:18]} C={consts}")
```
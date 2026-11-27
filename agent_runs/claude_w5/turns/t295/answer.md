The tool now matches my local edited model almost exactly: median relative error 2e-6 for both drug and other bundles, and the edit's effect agrees in size and direction (cosine 0.99). With the kv tensor saved, the tool's teacher-forced core is my local core.

Oddly, my local original model (`W64`) is further from the tool's original (1.3e-4). So `W64` probably differs slightly from the true original weights, which I need to check.

Since the core matches, the hour-13 asymmetry (x2 −3.08, x1/2 +0.09) must come from the head's reading of the edited state or from something specific to the rollout. My single-bundle core response is symmetric. Next I'll compute the hour-13 predicted glucose mean locally for selection-A stays under the factual, x2 and x1/2 doses with my head functions. First I need their signatures, since I can't see their code.

- If the local hour 13 reproduces the asymmetry, the head is the cause.
- If it doesn't, the tool's rollout does something I'm not modelling.

```python
import inspect, numpy as np
for nm in ['head_P','head_eval','head_inputs','head_probs','pred_mean','view0','v0_of','fixed_of','last_feat','stay_views','probs_from_v0','head_mean_rows','cmp_head','last_before','prior_mean','gapv','gapf','ctx_of','one_step_from_tool','free_run','eval_full','tf_states']:
    f_ = globals().get(nm)
    if f_ is None: continue
    try: print(nm, inspect.signature(f_))
    except Exception: print(nm, '?')
bad64 = [n for n in NAMES if np.abs(to64(W64[n]) - load_original(n).double().numpy()).max() > 1e-7]
print('W64 differs from original on:', bad64[:10], len(bad64))
for n in bad64[:5]: print('  ', n, np.abs(to64(W64[n]) - load_original(n).double().numpy()).max())
print('H type', type(H).__name__, getattr(H,'shape',None), list(H.columns)[:20] if hasattr(H,'columns') else '')
print('fixed', type(fixed).__name__, getattr(fixed,'shape',None), '| TRr', getattr(TRr,'shape',None), '| V0', getattr(V0,'shape',None), '| cen', getattr(cen,'shape',None))
```
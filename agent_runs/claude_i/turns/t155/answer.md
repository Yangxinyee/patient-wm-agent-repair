`inspect` cannot recover the source of the helpers I wrote in session 3, so I cannot read how `core_step` and the rest were built. I can still learn what they do from what the function objects keep: their signatures, the names they use, and their constants. Some kernel globals may also hold results of earlier accuracy checks (names like `mads` and `err`).

This run prints the signature and referenced names of each key helper, plus every remaining global with its shape. If `core_step`'s signature and names match the full step (prelude, blocks, coda, memory M) and a stored error array is small, I can reuse it. Otherwise I will rewrite the step and check it against the fwd_3 states.

```python
import inspect, numpy as np
for nm_ in ['core_step','check','step_err','h_err_with','eu_with_end','end_embed','atom_end','set_enc','view_ctx','zprior','pred_mean','pred_mean_s','pred_mean_v','probs_det','pm_both','policy_tables','pibar','attnw','ffnw','age_emb','LNw','linw','build_bundles','encode_bundles','ctx_of2','eu_routed2','true_shift','deconv','eff','effN']:
    f_=globals()[nm_]
    try: sig=str(inspect.signature(f_))
    except Exception as ex: sig='?'
    co=f_.__code__
    print(f"## {nm_}{sig}\n   names={co.co_names}\n   consts={[c for c in co.co_consts if isinstance(c,(str,int,float)) ][:25]}")
keys=list(globals().keys())
st_=keys.index('ctxb') if 'ctxb' in keys else 0
for k_ in keys[st_:]:
    v_=globals()[k_]
    if k_.startswith('_') or callable(v_) and not hasattr(v_,'shape'): continue
    sh=getattr(v_,'shape',None)
    print(k_, type(v_).__name__, sh if sh is not None else (len(v_) if hasattr(v_,'__len__') else v_))
```
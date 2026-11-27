Run 292 showed that the z-token channel computes the dose residual correctly (correlation 1.0). It still has no effect on the rollout: raising the gains from 3/8 to 8/20 left the readouts unchanged (−3.0 / +1.3 / +0.7 / −0.9). So the prelude route is blocked. My explanation is that the layer norm before attention removes the injected signal, and what gets through changes the attention keys.

I have 8 Python runs left, so I am switching to a simpler route on the head side:
- `z_prior` and `z_head` only feed the value head's view, and `z_head` is currently zeroed, so `z_prior` is unused and free to repurpose.
- Make `z_prior` deterministic and have it output a linear decode of the dose residual from the context.
- Map that through `z_head` along a direction that moves only the glucose predictions (insulin) or only the MAP predictions (norepinephrine).
- The rollout carries the model's own drawn readings forward with about 57% persistence. A larger one-step effect should therefore build up over hours 13–24, and I will calibrate the gain on the offline rollout.

This run only checks feasibility: the input and shape of `z_prior`, whether my replica code uses `z_head`/`z_prior`, the helper function signatures, and the contents of the saved data files.
- **Supports the plan:** `z_prior` takes a 192-dimensional context and the replica includes the `z_head` term.
- **Refutes it, or forces a patch:** the replica ignores `z_head`.

```python
import inspect, numpy as np
o=[f"{n}{tuple(SHAPES[n])}" for n in NAMES if any(k in n for k in ['z_prior','z_head','query_norm','type_emb','value_bin','gap'])]
print(' | '.join(o)[:1400])
for fn in ['evalW','tfnll','rollout','run_stay_steps','mkW3','build_post','tf_stay_z','bstep','enc_eu2']:
    f=globals().get(fn)
    try: print(fn, inspect.signature(f) if f is not None else None)
    except Exception as e: print(fn,'err',str(e)[:60])
for p in ['/work/wm2.py','/work/wm2_rollout.py']:
    L=open(p).read().split('\n')
    hits=[f"{i}:{l.strip()[:100]}" for i,l in enumerate(L) if ('z_head' in l or 'z_prior' in l or l.startswith('def '))]
    print(p,len(L)); print('\n'.join(hits[:40]))
for p in ['tf_train','s1data','jac']:
    try:
        d=np.load(f'/work/{p}.npz',allow_pickle=True); print(p,{k:d[k].shape for k in list(d.keys())[:14]})
    except Exception as e: print(p,'err',str(e)[:60])
print('dicts>100:',[k for k,v in list(globals().items()) if isinstance(v,dict) and len(v)>100][:10])
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 7 (run 293+)\n- Run 292: z-token message channel decodes r exactly but has NO effect on readouts at any gain -> prelude route abandoned.\n"
            "- Plan B: z_prior (unused since z_head=0) -> deterministic linear decode of dose residual from ctx; z_head maps it along glucose-only / MAP-only view directions. Relies on rollout persistence (~57%) of model-drawn readings; calibrate gain with evalW, check tfnll.\n")
```
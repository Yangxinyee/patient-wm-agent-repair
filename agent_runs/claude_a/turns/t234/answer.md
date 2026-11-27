The kernel still has the full offline toolkit: a teacher-forced loop and NLL (`tf_loop`, `tf_nll`), an offline rollout (`rollout_stay`), my target estimator (`true_shift`) and prelude internals. The data is a long event table: stay, hour, event, value, rate, dose.

One risk in the readout design: the head's code and bin embeddings may be shared with the input encoder. If so, editing them would also change every encoded bundle. Next I'll check how the head builds its query and with_code input, the shapes of the relevant embeddings, and the signatures of my key helpers.

- **Safe to use as a gate:** `type_emb` is per-code and head-only.
- **Must avoid:** shared embeddings.

```python
import re, inspect
A=open('/model/ARCHITECTURE.md').read()
i=A.find('## Heads'); print(A[i:i+1400].replace('\n',' ')[:1400])
for n in NAMES:
    if any(k in n for k in ['type_emb','embed.','bins.','last_','query_norm','status']): print(n,SHAPES[n])
def sig(f):
    c=f.__code__; a=c.co_varnames[:c.co_argcount]; d=f.__defaults__ or ()
    return f.__name__+str(a)+(' def='+str(d)[:60] if d else '')
for nm in ['prelude_parts','tf_loop','tf_nll','tf_eval','rollout_stay','head_value','tf_views','true_shift','build_ds','internals','capture_wc','head3_vals','o3_slots','fitdec','make_scaffold','scaffold_v']:
    print(sig(globals()[nm]))
```
Run 147 shows that every dim in D has a slightly negative kill cost, at most about 1e-2 nats on any of the three cost rows. Zeroing any one of them alone does not hurt factual fit. So D is a safe pool for the dims I need: insulin and norepinephrine memories, one or two write-staging dims, one gate dim and one zero-content reference dim.

Each value is a single-dim estimate on a small sample, so it is noisy. Before any edit I will measure the joint cost of the chosen set on a large stay set, using paired random numbers.

The remaining shapes fit the plan:
- `embed_transition.value_bin` (221×17) is a logit table indexed by code×17 + last value bin. A per-last-bin correction of the memory's factual mean can go there exactly.
- `query_norm`, `out_norm` and `prelude_norm` are ordinary (192,) gain/bias LayerNorms.
- There are tensors I never noted: `filter_gate`, `filter_norm` and `pending_mlp`. I need to check whether they touch the h-stream or the value path, because any reader of a repurposed dim has to be cut.

The helpers whose source is gone are `kill_dims`, `rollout_gen`/`fn` (rollout with "specs") and `tf_loop`. I'll inspect their bytecode constants and names to recover what they do. I'll also grep ARCHITECTURE.md for the pending and filter components and for how the rollout samples. If `kill_dims` already cuts every reader and writer of a dim, I can reuse it for the joint-cost check. If not, I'll write a full kill function next.

```python
import dis, re
for fnm in ['kill_dims','rollout_gen','fn','tf_loop','stay_inputs','rates_by_bundle']:
    co = g[fnm].__code__
    cs = [c for c in co.co_consts if isinstance(c,(str,int,float)) and not (isinstance(c,str) and len(c)>80)]
    print(f'{fnm}: consts={cs[:60]}\n   names={co.co_names[:60]}\n')
txt_ = open('/model/ARCHITECTURE.md').read()
print('ARCH len', len(txt_))
lines_ = txt_.split('\n')
for i,l in enumerate(lines_):
    if re.search(r'pending|filter|rollout|sample|dose_edit|z_prior|z_post|gap', l, re.I):
        print(i, l[:220])
```
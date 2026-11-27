Several of my earlier replies were blocked, so I'll keep outputs short and summarised.

What I know now:
- **Architecture:** re-read. The glucose value head sees insulin only through the recurrent state (out, h, z). The doc gives no direct path from dose to readout.
- **Earlier passes:** fwd_2 was a factual run of the original model and fwd_3 was a states request.
- **Problem from fwd 1:** the original model's dose effects are about 10x (insulin) and 30x (norepinephrine) smaller than what the records imply.

Next I'll read the two small summaries and list the head and core tensors that matter for an edit, with their shapes. That tells me where a dose signal could be injected and read out.

```python
import json, collections
for k in (2,3):
    s=json.load(open(f'/work/fwd_{k}/summary.json'))
    print(k, json.dumps(s)[:500])
grp=collections.Counter('.'.join(n.split('.')[:2]) for n in NAMES)
print(dict(grp))
keys=['attr_heads.value','embed_transition','query_norm','code_bias','core.coda','core.c_out','core.b_in','core.delta','core.a_in','core.flow','core.out_norm','embed.numeric','z_head','core.gap']
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES if any(n.startswith(q) for q in keys)))
```
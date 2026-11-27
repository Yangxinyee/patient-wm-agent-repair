fwd_3 is a factual pass of the original model on 100 training stays, the same ones as fwd_2. Its overall value NLL is 1.436 per reading (glucose_finger 1.54, MAP arterial 1.33). It gives per-reading predicted means. The NLL is categorical over the 17 fine bins, so a dose effect has to show up as moved probability mass.

My kernel holds an unfinished head check, `cmp_head(type_fn, status_idx, ln_layout, ...)`. I never settled which `type_emb` index and which 8-input `last_numeric` layout the value head uses. Before building the dose channel I need an exact local head, so I can test edits offline and save forward passes.

Next I will look at:
- the fwd_3 CSV columns;
- the vocab config entries that might define the 15 types and the 8 register inputs;
- whether `cmp_head` already has a recorded result.

If I find a types list, I can fix those two settings directly; otherwise I will test candidate layouts against the tool's predicted means.

```python
import pandas as pd, json
f3 = pd.read_csv('/work/fwd_3/factual_original.csv')
print(f3.shape, list(f3.columns)[:20])
print(f3.head(3).to_string()[:800])
V = VOCAB
print(list(V.keys()))
cfg = V.get('config', {})
print({k: (v if not isinstance(v,(list,dict)) or len(str(v))<200 else str(v)[:200]) for k,v in cfg.items()})
for k in V:
    if k not in ('config',) and ('type' in k.lower() or 'status' in k.lower() or 'last' in k.lower()):
        print(k, str(V[k])[:300])
```
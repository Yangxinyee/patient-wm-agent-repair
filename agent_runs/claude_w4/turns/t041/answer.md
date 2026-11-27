The last run loaded a `states` file: 5,218 bundles × 992 numbers, with flags for which bundles held insulin or norepinephrine. Drug bundles sit at hour h+0.0833, just after the hourly readings, so the data layout is as expected.

My notebook stops at run 29. It does not record what forward passes 1–3 contained or what they showed. My estimates so far from the records:
- **Insulin:** about −2 mg/dL glucose per unit/h one hour later, building to about −10 mg/dL per unit/h over 12 hours. The effect looks linear and the same across glucose levels.
- **Norepinephrine:** about +3.7 mmHg MAP per mcg/kg/min one hour later, building to about +10 over 12 hours.

Before planning an edit I need two things:
1. The architecture again, especially how dose enters through encode_action and how the readings head works.
2. What the earlier forward passes showed, in particular the original model's dose_edit answers, which tell me how far the model is from the truth.

I will print ARCHITECTURE.md, list /work, and summarise the forward outputs. Nothing is tested yet; this run only re-establishes context.

```python
import os, json, glob
print(open('/model/ARCHITECTURE.md').read())
for d in sorted(glob.glob('/work/fwd_*')):
    print('==', d, os.listdir(d))
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
            print(f, open(p).read()[:1500])
print(open('/work/sel100_both.json').read()[:600])
print([k for k in globals().keys() if not k.startswith('_')][:200])
```
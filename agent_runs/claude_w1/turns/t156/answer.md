The regression settles the main question:
- **Glucose:** residuals load strongly on the lagged insulin innovations, about −3.1 and −1.6 mg/dL per unit/hour at lags 1 and 2. They load little on the lagged policy part, which sums to about −0.7 across lags. So the model does use the readings-driven part of the dose. What it misses is the effect of the random part. Amplifying the dose response as a whole would therefore bias factual predictions, by about 3 × 0.94 ≈ 2.8 mg/dL rms, well over budget. The edit must act like c × (dose − policy expectation).
- **MAP:** the same picture holds, but noisier.

My candidate design:
1. Inject c × dose exactly at the action token, through the prelude value projection. It is linear in dose and zero for other tokens.
2. Subtract c × f(last reading) at the value head. The head knows the last reading exactly, and for insulin the policy is a step function of the reading taken 5 minutes before the dose.

This works best if the injected signal acts mainly on the next reading in teacher-forced mode. To build it I need my exact local replica with an injection hook. This run inspects what run_stay and core_step reference, especially how the edit argument works, so I can reuse them rather than rewrite them.

```python
for f in [run_stay, core_step, prelude_attn, bundles_of_stay, head_mean]:
    c = f.__code__
    print('==', f.__name__, 'args', c.co_varnames[:c.co_argcount])
    print('  names', ' '.join(c.co_names)[:420])
    print('  locals', ' '.join(c.co_varnames[c.co_argcount:])[:300])
    print('  consts', str([x for x in c.co_consts if isinstance(x,(str,int,float)) and x is not None])[:250])
```
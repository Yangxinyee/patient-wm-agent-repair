The helper functions were defined through exec, so their source cannot be shown. That is no problem: the analysis I need (doses, policy expectation, tool residuals) is quick to rebuild. This run only lists the argument signatures of my key replica helpers, so I can reuse them, plus the tool's factual-output columns. Nothing here tests a hypothesis yet. It just sets up the residual-versus-dose-level regression that comes next.

```python
import inspect, pandas as pd
for f in [run_stay, bundles_of_stay, core_step, encset, avec, gen_samples, true_shift, truth_rows, nu_of, ne_expect, base_ins, prelude_attn, head_mean, run_head]:
    try: sig = str(inspect.signature(f))
    except Exception as e: sig = 'n/a'
    print(f.__name__, sig[:150])
F2df = pd.read_csv('/work/fwd_2/factual_original.csv')
print('F2 cols', list(F2df.columns), F2df.shape)
print(F2df.head(2).to_string()[:500])
```
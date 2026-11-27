The data layout fits the injection plan. Each infusion is a one-hour record that starts 5 minutes after the hour, so the dose arrives every hour as a fresh action token. That token encodes the dose exactly.

The policy facts are the key for factual safety:
- **Insulin:** the rate is a step function of the glucose reading taken 5 minutes earlier, times independent noise.
- **Norepinephrine:** each new rate is the previous rate times a step that depends on MAP, plus independent noise.

The model currently uses the noise part of the dose only weakly. Doses also track readings, so simply amplifying the dose response would push factual predictions off. The open question is whether the original model's factual residuals depend on the dose level (amplification is then safe) or only on the noise part (a policy-based compensation is then needed). To test that, I first need my existing helpers for the policy and innovations and the layout of the tool's factual output. This run prints short source excerpts and column names only.

```python
import inspect, pandas as pd
for f in [nu_of, ne_expect, add_pol, base_ins, build_panel, true_shift]:
    s = inspect.getsource(f)
    print('---', f.__name__, len(s)); print(s[:520])
F2df = pd.read_csv('/work/fwd_2/factual_original.csv')
print('F2 cols', list(F2df.columns), F2df.shape)
print(F2df.head(3).to_string()[:600])
```
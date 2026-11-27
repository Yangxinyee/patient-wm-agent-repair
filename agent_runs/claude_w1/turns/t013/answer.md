Within narrow bins of current finger glucose, each extra u/h of insulin goes with 1.5–2.5 mg/dL lower glucose one hour later. The slope grows somewhat with the glucose level: about −1.5 near 150 and −2.5 near 280. That hints at an effect proportional to glucose, about −1% of glucose per u/h per hour. So insulin does lower glucose here.

What matters for the readout is whether this effect persists and accumulates over hours. That depends on the glucose dynamics, and on whether finger readings are a noisy measure of a latent true glucose.

Before modelling that, I want the exact dosing rule: all the steps of the base-rate function and how the noise behaves. Then I can test whether the dose noise is truly random. That means checking it against past readings, past dose noise and other variables. If it depends on anything else, the simple regression would be confounded.

```python
# full base function: mean insulin by exact glucose reading (finger at hour h), all on-hours
cur = Qon[Qon.glucose_finger.notna()]
m = cur.groupby(pd.cut(cur.glucose_finger, np.arange(140, 400, 5))).insulin.agg(['mean','median','size'])
print(m[m['size']>20].to_string())
# lab-driven hours
lab = Qon[Qon.glucose_finger.isna() & Qon.glucose_lab.notna()]
m2 = lab.groupby(pd.cut(lab.glucose_lab, np.arange(140, 400, 10))).insulin.agg(['mean','median','size'])
print(m2[m2['size']>5].to_string())
```
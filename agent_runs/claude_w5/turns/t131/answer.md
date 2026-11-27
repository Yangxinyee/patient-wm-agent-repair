The last run settled the initial state. Starting from h = 0 with a first-bundle gap of 1 second reproduces the first row to 9e-8, and the last row matches to 1.4e-6. So the local float64 core reimplementation is now exact from the start of a stay.

My current view:
- **Data:** the training records give clean, randomised dose variation. g-estimation puts insulin x2 near −14 mg/dL and norepinephrine x2 near +4 mmHg on the tuning sample.
- **Model:** the original model nearly ignores doses (−0.5 and +0.2).
- **What a correct edit needs:** the model's next-reading prediction should equal the true conditional mean given glucose history plus doses, i.e. m0 + Σ_j γ_j·(dose residual at lag j). If it does, the model's own rollout reproduces the causal effect, and the factual fit should improve slightly rather than degrade.
- **Hard part:** this needs a persistent dose memory in h, which the model lacks.

Before designing the edit, I need the exact h-update, head and rollout definitions again: how abar, delta_in and b_in depend on E, and whether the decays are input-independent. I also want to know what fwd_3 contains.

If the h decay is per-dimension and input-independent, a leaky-integrator channel is feasible. If it depends on E, I will need a different mechanism.

```python
import os, json
print(os.listdir('/work/fwd_3'))
for k in ['fwd_1','fwd_2','fwd_3']:
    p='/work/'+k
    for f in os.listdir(p):
        print(k, f, os.path.getsize(os.path.join(p,f)))
txt=open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt)
```
The model badly under-responds to dose, compared with the dose-response kernels I estimated from the records:

| Edit | Model shift | Kernel-implied true shift | Ratio |
|---|---|---|---|
| Insulin x2 | −2.0 mg/dL | −13.6 mg/dL | ~7× |
| Insulin x1/2 | +1.2 | +6.8 | ~6× |
| NE x2 | +0.37 mmHg | +4.5 | ~12× |
| NE x1/2 | −0.06 | −2.2 | ~35× |

The model's insulin effect also builds slowly and keeps growing over the 12 hours. The kernel-implied effect saturates after about 4 hours.

The core difficulty: in the records, the dose is almost a fixed function of the preceding reading (band × noise). A bigger response to dose magnitude must be centred on the dose the policy would give for that reading. Otherwise factual predictions shift after every infusion and the value NLL breaks the 0.005 limit. So the edit must act on (dose − expected dose given reading). That needs a place where the action signal meets the reading.

Next, I will list the tensor names and shapes. That shows which components I can edit precisely (action encoder, numeric dose/rate columns, bins tables, prelude, heads). It also shows how much of my local replica (z-posterior correlation 0.976 with ReLU) I can build on.

```python
for nm in NAMES:
    print(nm, tuple(SHAPES[nm]), end=' | ')
print()
print(json.dumps(VOCAB.get('config',{}))[:1500])
print([k for k in VOCAB.keys()])
```
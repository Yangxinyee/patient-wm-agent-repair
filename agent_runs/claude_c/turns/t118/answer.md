Pass 10's predictions differ from the original by only 0.009 mg/dL rms for glucose_finger and 0.001 for MAP_arterial. They do not track the previous surprise at all (slope 0.00002). So the slots do not use the z token, and the z route is closed.

That points to the real bottleneck:
- Readings reach the state through the recurrent blocks' cross-attention to the memory M of bundle vectors, which adds directly to h.
- Drugs are not in M. They enter only through the prelude's action token, whose message is renormalised by `prelude_norm` against h (norm about 842) before `b_in` injects it.

That would explain both the weak dose response and its ceiling near −2.5.

Before concluding that no selective lever exists, I want to see every tensor I have not yet looked at. A parameter that acts per drug, per time-gap or on the drug-history pathway would be a direct handle.

```python
seen = set(sel)
rest = [nm for nm in NAMES if nm not in seen]
print(len(rest))
print('; '.join(f'{nm}{tuple(SHAPES[nm])}' for nm in rest))
```
The vocabulary shows that insulin rates come in 0.5-unit steps with a median of 2 U/h. Norepinephrine rates seem capped at 1.0 mcg/kg/min, since all the top bin edges are 1.0. The norepinephrine dose equals rate × 80 kg × 60 / 1000 in every case shown, so body weight looks fixed. Two consequences for the edit:
- Doubling a rate pushes many records into bins and z-scores the model never saw in training, for example norepinephrine above 1.0 lands in bin 16.
- The model's x2 response is therefore partly an extrapolation.

Next I will print the remaining numeric entries and the tensor shapes. That shows exactly which weights carry dose information, including whether the dose and rate bin tables are per-code or shared between the two drugs.

```python
for e in V['numeric']:
    print(e)
print()
import collections
for n in NAMES:
    print(n, SHAPES[n])
```
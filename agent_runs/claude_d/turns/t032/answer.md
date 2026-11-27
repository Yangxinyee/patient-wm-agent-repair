The subgroup checks support a linear, homogeneous response for both drugs.
- **Insulin by base** (pre-dose glucose level, rates about 1, 2 and 3+ u/h): the per-unit effect is the same, −1.92 / −1.97 / −2.03 at lag 1 and −2.29 / −2.40 / −2.41 at lag 2. So the effect is linear in dose across the range x2 uses.
- **Insulin by realised rate:** these splits are invalid, because they select on the instrument itself.
- **Norepinephrine:** the lag-1 effect is about 3.0–3.5 by previous rate and 3.2–3.9 by MAP band. It shows no saturation within 0–1, so linear extrapolation past the cap is the evidence-based choice.

I now trust the targets, roughly insulin −14.5 / +7.3 and norepinephrine +4.6 / −2.3 on these stays.

The harder part is the edit. Even the first-hour response, which depends only on the teacher-forced record plus the doubled 12:05 dose, is about 1/6 of the truth (−0.61 vs −3.64). So the model has underlearned the dose effect itself.

To design an edit I need the model's exact computation, ideally a local CPU copy I can check against the tool. Next I will read tensors.json for extra detail, look at the bin and embedding tables involved, and draft a local implementation.

```python
tj = json.load(open('/model/tensors.json'))
print(type(tj), len(tj))
s = json.dumps(tj)
print(s[:3000])
```
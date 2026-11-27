The last run showed that, at insulin bundles, the model's z posterior mean carries almost none of the insulin dose surprise (CV R² 0.046). It does carry the predictable part of the dose (R² about 0.31). This agrees with the earlier finding: the model's state keeps only the dose information that the glucose band already predicts. It ignores the part of the dose that actually changes glucose, which is why its original dose_edit answers are near zero. The crash afterwards came from taking log(0) of norepinephrine rates and does not matter.

My records-based estimates are:
- **Insulin:** lowers glucose by about 1.5–2 mg/dL per unit/h in each of the next 5 hours, decaying to near zero by about 10 hours. The total is about −10 mg/dL per unit/h-hour.
- **Norepinephrine:** raises arterial MAP by about +3.7, then +2.7, then +1.3 per mcg/kg/min, with a total of about +10.

To design a weight edit that puts the dose surprise into the core, I need the exact computation path, which the restart dropped from my context. My next step is to re-read /model/ARCHITECTURE.md and write the key equations into my notes. That shows me which tensors carry the action and how readings are produced in the rollout. No hypothesis is tested here; this is to recover information I need.

```python
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:15000])
```
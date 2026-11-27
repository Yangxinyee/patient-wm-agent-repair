On the drug-heavy set (every stay has insulin and/or norepinephrine after hour 12), the edited model's factual NLL is slightly better than the original: 1.34146 vs 1.34164 (−0.00018 nats). On the random set it was +0.00001. So dropping dose size is free or mildly helpful for factual prediction, as expected if doses carry no information about future readings, and it stays far inside the 0.005 margin for any stay mix.

Everything is now consistent:
- **The truth:** both drugs are inert. Insulin shows no effect via randomised multipliers, step-ups at five thresholds, DML and the cumulative test (implied x2 readout +0.03 ± 0.93 mg/dL). Norepinephrine shows no start effect, and its small short-lag association is explained by confounding, as the pre-dose negative control shows.
- **The edit:** readouts exactly 0 for all four, factual fit unchanged.

The extra forward passes cannot sharpen the truth further; that limit comes from the records, not the model. So I will write the final notebook entry and then submit.

```python
open('/work/notes.md','a').write("""
- Cumulative randomization test (insulin, hours 13-24 vs summed random dose parts since 12): slope 0.002+-0.065/unit -> implied x2 readout +0.03+-0.93 mg/dL.
- fwd_4 factual both on drug-heavy set100: orig 1.34164 vs edited(A) 1.34146 (-0.00018). Edit A safe.
## FINAL DECISION: submit Edit A (dose-invariance). Expected readouts: all four exactly 0.0; true effects estimated ~0
   (insulin x2 ~0+-1 mg/dL, x1/2 ~0+-0.5; NE x2/x1/2 within +-0.5 mmHg, best estimate 0).
""")
print(open('/work/notes.md').read()[-1500:])
print(edited())
```
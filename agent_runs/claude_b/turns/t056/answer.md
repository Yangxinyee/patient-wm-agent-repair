The second factual check passes. On 100 tuning stays the earlier checks never used, the edited model's NLL per reading is 1.58089, against 1.58099 for the original. That is again a hair better, by event as well, consistent with fwd_2 (−0.00006). The 0.005 limit is not at risk.

All the pieces now agree:
- The training records show no causal effect of either drug once only the clean, randomly varying part of the doses is used.
- The edit makes the model's dose-edit answers exactly zero for all four questions.
- Factual fit is unchanged.

I'll record the final state in the notebook, then give the final answer.

```python
with open('/work/notes.md','a') as f: f.write("""
- fwd_4 (factual both, 100 other tuning stays sel2): orig 1.58099, edited 1.58089 (-0.0001). NLL margin confirmed on independent stays.
## FINAL STATE
Edits: embed.numeric.weight (z_dose, z_rate columns -> 0), embed.bins.dose.weight & embed.bins.rate.weight (rows 1..16 -> row 8).
Drug magnitude invisible -> all four dose-edit shifts exactly 0. Records: insulin and NE have no causal effect
(clean-noise implied x2: insulin +0.31 (se<=0.73) mg/dL; NE +0.15 (se<=0.61) mmHg). Expected readouts: 0, 0, 0, 0. Submit.
""")
print(edited())
```
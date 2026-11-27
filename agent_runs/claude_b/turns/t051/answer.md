Rebuilt with threshold-aligned cells, the titration residual is clean. It no longer depends on the previous MAP (+0.0005 ± 0.0006) or on the cuff–arterial difference.

Its effect on arterial MAP is null at every lag from 1 to 12 hours. All coefficients are within about 1.5 SE of zero with no consistent sign; lag 1 is −0.01 ± 0.12. After deconvolving by the rate persistence, the implied norepinephrine x2 shift is +0.15 mmHg (SE 0.19–0.61) and x1/2 is −0.07.

So the earlier −0.3 "effect" was the threshold-straddling artefact. Both drugs show no effect:
- insulin x2 ≈ +0.19 ± ~0.5 mg/dL;
- norepinephrine x2 ≈ +0.15 ± ~0.4 mmHg.

Both are consistent with exactly zero, which is what the current magnitude-blind edit produces (fwd_3: exactly 0.0). Its factual NLL change is −0.00006 ± 0.00006 per reading, well inside the 0.005 margin.

I'll record all of this in the notebook now, since the conversation may restart. Next I'll do the analogous threshold check for the insulin sliding scale. Split values at 150, 180, … could slightly bias the insulin estimate upward. If exact-integer cells keep the insulin effect near zero, the conclusion is final.

```python
notes_add = """
## Session 2 (runs 38-47): re-check of causal null with sustained (12h) effects
- fwd_2 (factual both, 100 tuning stays): orig 1.33354, edited 1.33348 (per-reading diff mean -6e-5, sd 0.0043, se 6e-5) -> NLL safe.
- fwd_3 (dose_edit both): edited shifts exactly 0.0 for all 4 edits (magnitude-blind edit works; no other dose path).
- CAUTION: naive insulin residual (dose - E[dose|finger bin]) mixes endogenous parts: (B) omissions & (C) doses at reading<150/missing
  depend on prev reading/prev dose -> huge spurious positive 'effects'. Only (A) intensity noise among given doses at reading>=150 is clean.
- Insulin (A) impulse responses with dose-time controls, k=1..12: all ~0 (|b|<0.12, se .04-.11); no persistence in later doses.
  Implied insulin x2 = +0.19 (se 0.22..0.72), half -0.09. Lab outcome also null. => insulin effect = 0.
- NE: titration thresholds are sharp on INTEGER MAP but reading 65 is half up, 75 half down (policy uses unrounded value).
  2-mmHg cells straddled thresholds -> spurious -0.3/unit negative effect. With exact-integer cells x source, excluding 65/75 and prev>=0.75:
  residual exogenous (prevMAP +0.0005+-0.0006), impulse k1..12 null (k1 -0.01+-0.12), persistence rho1 0.81.
  Deconvolved implied NE x2 = +0.15 (se 0.19..0.61), half -0.07. => NE effect = 0.
- CONCLUSION: both drugs have no causal effect; the magnitude-blind edit (shifts exactly 0) is the right answer. NLL unchanged.
"""
with open('/work/notes.md','a') as f: f.write(notes_add)
# insulin tier boundaries at exact integer readings
dI=pd.DataFrame({'d':Ins[:,1:].ravel(),'r':Fg[:,1:].ravel()}).dropna()
for th in [150,180,220,260,300]:
    sub=dI[(dI.r>=th-2)&(dI.r<=th+2)]
    print(th, sub.groupby('r')['d'].agg(['mean','count']).round(3).T.to_string())
```
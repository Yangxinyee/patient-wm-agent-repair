Three facts about the policy are now clear.
- The insulin scale is driven by the finger reading when one exists, and by the lab reading when only lab exists. When both exist, the lab scale adds nothing.
- With no glucose reading in an hour, insulin is never given.
- Every hour after an insulin hour has a glucose reading, so patients on insulin are checked hourly.

The conditional jitter in rate within a scale bin looks random. If it is independent of the hidden state, an adjusted regression identifies the drug effect, and the jumps at 150/180/220/260/300 provide a regression-discontinuity check.

Next I will examine the glucose dynamics. I will regress the next-hour finger glucose change on the current insulin rate, controlling flexibly for the current reading (fine glucose bins) and the previous rate. I will compare three estimates:
1. naive (no adjustment);
2. within-bin, controlling for glucose;
3. RD at the 180 threshold, a local comparison of readings just above versus just below.

Agreement between 2 and 3 would support an unconfounded jitter. A naive estimate that is positive or less negative would show the confounding a model could learn. I will also check the measurement process (P(reading | previous insulin)).

```python
P['gf_next'] = g.gf.shift(-1); P['gl_next']=g.gl.shift(-1)
P['gf_prev'] = g.gf.shift(1)
print("P(has gf at t | ins_{t-1}>0):", P.loc[P.ins_prev>0,'has_gf'].mean(), " | ins_prev==0:", P.loc[P.ins_prev==0,'has_gf'].mean())
print("P(has any glucose at t | ins_{t-1}>0):", P.loc[P.ins_prev>0,'glu_any'].notna().mean())
d = P[P.gf.notna() & P.gf_next.notna()].copy()
d['dG'] = d.gf_next - d.gf
import statsmodels.api as sm
```
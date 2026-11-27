Per unit of insulin, the effect profile is nearly identical across glucose bands at lags 1–4:
- about −1.9 at lag 1, −2.3 at lag 2, −2.2 at lag 3 and −1.7 at lag 4;
- this holds for base 1 (150–180), base 2 (180–220) and base 3+ (above 220);
- later lags are noisier, but all bands give a cumulative effect of about −11 to −15 mg/dL per sustained 1 u/h.

So the insulin effect is additive and roughly linear in dose, not proportional to glucose level. The pooled, variance-reduced profile is:

γ = [−1.96, −2.36, −2.22, −1.69, −1.33, −1.02, −0.63, −0.44, −0.47, −0.29, −0.21, −0.09], cumulative −12.7.

That shape fits a delayed compartment, a convolution of two geometric decays at about 0.77 and 0.43 per hour.

Next I'll analyse norepinephrine and MAP the same way, starting with the dosing policy. In the example stay the norepinephrine rate drifted gradually (0.70, 0.66, 0.74, …), which looks different from the insulin sliding scale. I'll check how the rate depends on current MAP (arterial, else cuff), on the previous rate, and the size of the step. The policy's structure determines where the exogenous variation is.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 15-17: insulin effect (g-estimation, linear SNMM, instrument = rate residual given integer glucose)
- per-unit (1 u/h for 1h) effect on finger glucose at lag j (later doses fixed):
  [-1.96,-2.36,-2.22,-1.69,-1.33,-1.02,-0.63,-0.44,-0.47,-0.29,-0.21,-0.09], cum -12.7 over 12 lags.
- Same at lags 1-4 across glucose bands (150-180, 180-220, >220) -> additive, ~linear, not prop. to G.
- Total effect incl. policy feedback (OLS with g fixed effects) peaks lag 2 at -2.25, decays faster.
""")
P['map_now'] = P.map_arterial.fillna(P.map_cuff)
ne = P[P.norepinephrine.notna()].copy()
print("ne records", len(ne), " has art same h", ne.map_arterial.notna().mean(), " cuff", ne.map_cuff.notna().mean(), " either", ne.map_now.notna().mean())
print("prev ne present", ne.norepinephrine_prev.notna().mean())
print(ne.norepinephrine.round(2).value_counts().sort_index().head(30).to_string())
first = ne.groupby('stay_id').head(1)
print("first ne hour", first.h.describe().to_dict())
print("map at first ne", first.map_now.describe().to_dict())
ne['dr'] = ne.norepinephrine - ne.norepinephrine_prev
print("step dist:", ne.dr.describe().to_dict())
```
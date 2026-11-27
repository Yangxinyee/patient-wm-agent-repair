The effect is real and roughly linear, and it peaks after the next reading. Within band 2 the slope per u/h is about −2.0 at lag 1, −2.4 at lag 2 and −1.9 at lag 3. Lag 3 already includes the policy cutting later doses as glucose falls. So the drug's effect builds over about 2 hours.

The apparent convexity at the lowest rates fits a simple explanation. Rate counts per band match a lognormal (σ ≈ 0.30) around base 1/2/3, rounded to 0.5. In band 1 the shares predicted for 0.5, ≥1.5 and ≥2.0 are 0.168, 0.228 and 0.031, matching what is observed. If the drug acts on the unrounded rate, the lowest recorded category actually averages above its label (≈0.66 for "0.5"). That compresses its gap to the next category, which is what the band 1 and band 2 dummies show at a slope of about 2.0–2.2 per u/h.

To get the direct kernel K_j, the effect of one dose j hours later with later doses held fixed, I'll correct for policy feedback. I'll measure the effect of the random rate on later doses (D_m) and on later glucose (T_k), then deconvolve with T_k = Σ K_{k−m} D_m. This needs an outcome not subject to selection, so I'll use lab glucose, provided its presence is unrelated to insulin and glucose. I'll check that first.

```python
# check gl MCAR
Q2 = P.copy()
Q2['gl_next_present'] = Q2.groupby(level='stay_id').gl.shift(-1).notna()
Q2 = Q2[Q2.index.get_level_values('h')>=0]
print("P(gl next) | ins given:", Q2.groupby(Q2['ins'].notna()).gl_next_present.mean().round(4).to_dict())
print("P(gl next) by g_last band:", Q2.groupby(pd.cut(Q2.g_last,[0,100,130,150,180,220,400]),observed=True).gl_next_present.mean().round(4).to_dict())
# MAP measurement process
for c in ['ma','mc']:
    Q2[c+'_next_present'] = Q2.groupby(level='stay_id')[c].shift(-1).notna()
    print(c, "P(next present) | ne given:", Q2.groupby(Q2['ne'].notna())[c+'_next_present'].mean().round(4).to_dict())
# stay-level: arterial-line stays?
art = P.groupby(level='stay_id').ma.count()
print("ma readings per stay distribution:", art.describe().round(2).to_dict(), (art==0).mean())
print("mc per stay:", P.groupby(level='stay_id').mc.count().describe().round(2).to_dict())
```
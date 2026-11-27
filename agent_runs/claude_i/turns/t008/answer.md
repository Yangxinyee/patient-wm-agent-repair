The insulin policy looks almost deterministic. Insulin is given exactly when the latest recorded glucose is above roughly 150, and the rate follows a sliding scale by band: about 1 u/h for 150–180, 2 for 180–220, 3 for 220–250, and so on, with some scatter. Norepinephrine is always given when the latest MAP is below about 60 and becomes less likely above that. Hospital admission is always at −2 h, so it carries no information.

If dosing depends only on recorded readings, teacher-forced prediction gains little from the dose. In the rollout, though, readings from hour 13 on are hidden while recorded doses are shown. A dose then works as a signal of the hidden glucose, so doubling insulin would make the model infer *higher* glucose. That is my working hypothesis for the confounded answer.

To test whether the policy uses only recorded readings, I'll pin it down exactly. I want to know:
- which glucose drives the decision when finger and lab readings are both present in the same hour;
- whether the threshold is ≥150;
- how the rate is distributed within each band, which would be the random part.

```python
Q = P[P.index.get_level_values('h')>=0].copy()
# which reading drives insulin decision when both gf and gl present at hour h
both = Q[Q.gf.notna() & Q.gl.notna()]
print("both present:", len(both))
for name, g in [('gf',both.gf),('gl',both.gl),('mean',(both.gf+both.gl)/2),('max',np.maximum(both.gf,both.gl))]:
    pred = (g>=150)
    print(name, "agreement with insulin given:", (pred == both['ins'].notna()).mean().round(4))
# when no reading at hour h, does policy use last reading?
nor = Q[Q.g_any.isna()]
print("no reading at h:", len(nor), "agree with g_last>=150:", ((nor.g_last>=150)==nor['ins'].notna()).mean().round(4), " frac given:", nor['ins'].notna().mean().round(3))
# threshold exactness on gf-only hours
gfo = Q[Q.gf.notna() & Q.gl.isna()]
t = gfo.groupby(gfo.gf.round())['ins'].apply(lambda x: x.notna().mean())
print("p(ins|gf) near threshold:", t.loc[140:160].round(3).to_dict())
# rate distributions by glucose band
Q['gband'] = pd.cut(Q['g_last'], [140,150,160,170,180,190,200,210,220,230,240,250,275,300,500], right=False)
print(pd.crosstab(Q['gband'], Q['ins']).iloc[:, :14].to_string())
```
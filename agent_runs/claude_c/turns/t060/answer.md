The lag-1 shifts by recorded dose show why input amplification stalls: the response saturates.

- In the original model, the x2 response peaks for 1.5–3 U/h doses (about −0.9) and is only −0.41 above 3.25 U/h, against a true −7.6 there.
- With the z columns ×5, saturation sets in at lower doses. x2 of 2 U/h or more gives about 0, while x1/2 of 2–3 U/h gives about +1.8, because halving moves the dose back into the responsive range.
- Amplifying the bin deviations only adds noise.

Insulin bundles mostly hold 2 atoms (a start and an end record), sometimes 4 with norepinephrine.

The dose therefore needs a channel that stays linear at high doses. The z posterior is a candidate, because it must encode the rate precisely for the rate head to reconstruct it. Its u-block input in z_post (columns 384–575) is zero at reading bundles, so amplifying it would act only at drug bundles.

Next, a cheap check on existing states: is z26 linear in the recorded insulin rate up to high rates, within glucose tier?
- If yes, z26 is a non-saturating dose code worth routing through.
- If z26 also flattens at high rates, the saturation is upstream of z_post and this route fails too.

```python
x_ = So['x']
z26i = x_[ins_b, 960+26]
dfz = pd.DataFrame({'rate': rate, 'tier': tk, 'z26': z26i, 'ne': hn3[ins_b]})
dfz = dfz[(dfz.tier>0)]
print(dfz.groupby(['tier', pd.cut(dfz.rate, [0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 4.25, 6, 12])], observed=True)['z26'].agg(['mean','count']).round(2).unstack(0).to_string())
# overall z26 vs rate (no NE)
q_ = dfz[~dfz.ne]
print(q_.groupby(pd.cut(q_.rate, [0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 4.25, 6, 12]), observed=True)['z26'].agg(['mean','std','count']).round(2).T.to_string())
```
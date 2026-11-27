The dosing policies are clearly rule-based on the readings.
- **Insulin:** never given below a finger glucose of 140, given about 56% of the time at 140–160, and about 97% above 160. The mean rate steps up at about 180 (1→2 U/h) and 220 (2→3 U/h), so it looks like a sliding scale.
- **Norepinephrine:** given about 93–99% of the time when MAP is below 65, and its probability falls off above 65. The rate seems to depend on more than the current MAP; high-MAP hours still carry high rates, which suggests titration from the previous rate.

Sharp thresholds are promising: if dose jumps at a glucose cut-off while patients just either side are similar, a regression-discontinuity comparison can identify the drug's effect. Next I will look at the policy at single-unit resolution of glucose (130–260) and MAP (55–80). I will also check what drives dosing in hours with no finger-stick: lab glucose, the last reading, or the previous rate.

```python
d = P[P.glucose_finger.notna()].copy()
g = d.groupby(d.glucose_finger.round()).agg(n=('glucose_finger','size'), p_ins=('insulin', lambda x: x.notna().mean()), mean_rate=('insulin','mean'))
pd.set_option('display.width',250)
print(g.loc[135:265].T.round(2).to_string())
# hours without glucose_finger
nd = P[P.glucose_finger.isna() & P.insulin.notna()]
print("insulin given w/o finger reading at that hour:", len(nd), " with lab:", nd.glucose_lab.notna().sum())
print("total insulin records", P.insulin.notna().sum())
nd2 = P[P.glucose_finger.isna()]
print("hours w/o finger:", len(nd2), " p_ins:", nd2.insulin.notna().mean())
```
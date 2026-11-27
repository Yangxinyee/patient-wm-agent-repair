The z-scores are linear in the recorded amount: (x − median)/IQR per code. Insulin rate and dose share median 1.5 and IQR 1.0. Norepinephrine is capped at 1.0, so its fine bins above 1.0 are never seen for norepinephrine in training. A doubled norepinephrine rate at the cap therefore lands in an unfamiliar rate bin, which may explain the model's odd negative response to norepinephrine x2.

The states file has one row per bundle with out, the four h slots and the z posterior mean.

Next I test the hypothesis that the z posterior mean at insulin bundles encodes the dose innovation, i.e. rate minus the expected rate for the glucose tier, rather than the raw rate. I will regress z on the innovation and on the tier's expected rate and look at the explained variance. I will also check how much a dose-coding z direction varies at non-insulin bundles. Equal and opposite coding of raw rate and expected rate, or innovation-only coding, supports the hypothesis. Coding that mostly tracks the expected rate (tier/glucose) refutes it.

```python
S3 = np.load('/work/fwd_3/states_original.npz')
X3 = S3['x']; sid3 = S3['stay_id']; hr3 = S3['hour']; hi3 = S3['has_insulin']; hn3 = S3['has_norepinephrine']
print(Ptu.index[:3], Ptu.index.names)
fr = np.round(hr3 - np.floor(hr3), 4)
print(pd.Series(fr).value_counts().head(8))
print(pd.crosstab(fr, hi3))
# build insulin-bundle table
ins_b = np.where(hi3)[0]
h_int = np.floor(hr3[ins_b]).astype(int)
def lookup(col, sids, hrs):
    return np.array([Ptu.loc[(s, h), col] if (s, h) in Ptu.index else np.nan for s, h in zip(sids, hrs)])
rate = lookup('insulin', sid3[ins_b], h_int)
gf = lookup('glucose_finger', sid3[ins_b], h_int)
gl = lookup('glucose_lab', sid3[ins_b], h_int)
print(np.isnan(rate).mean(), np.nanmean(rate), np.isnan(gf).mean())
print(pd.DataFrame({'rate':rate,'gf':gf,'gl':gl,'h':hr3[ins_b]}).head(10))
```
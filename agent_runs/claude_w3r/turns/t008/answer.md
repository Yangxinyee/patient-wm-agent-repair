The insulin policy is a banded sliding scale on the latest glucose_finger, and within each band the rate does not depend on the glucose value. The bands are:
- below 150: off;
- 150–180: centred on 1 (0.5/1/1.5/2 with probabilities .17/.60/.20/.03);
- 180–220: centred on 2;
- 220 and above: centred on about 3.

Within a band the rate scatters widely, which looks like multiplicative noise rounded to 0.5. Readings of 185 and above were always measured that hour, so monitoring is hourly when glucose is high. The within-band scatter is the natural source of identification. It is only valid if the scatter is unrelated to the patient's hidden state.

Next I will define the band base rate and the log deviation of the actual rate from it. I will test whether the deviation persists within a stay (a patient-level factor) and whether it predicts the glucose trend before dosing (a placebo check). If the deviation is i.i.d. and unrelated to past trends, a simple within-band comparison identifies the effect.

```python
def band_base(g):
    return np.select([g<150, g<180, g<220], [0,1,2], default=3).astype(float)
D['base'] = band_base(D.g.values)
D['dev'] = np.where((D.r>0)&(D.base>0), np.log(D.r.where(D.r>0)/D.base.where(D.base>0)), np.nan)
print('dev stats by base:'); print(D.groupby('base').dev.agg(['count','mean','std']).round(3).to_string())
# check exact base for high band: rate distribution for g>=260 by finer bins
s3 = D[D.g>=220]; s3b = pd.cut(s3.g,[220,240,260,280,300,350,400])
print(s3.groupby(s3b).r.agg(['size','mean','median']).round(2).to_string())
# persistence: dev at t vs dev at t-1 within stay
n=len(stays); dev = D.dev.values.reshape(24,n).T  # rows stays, cols t=1..24
x=dev[:,:-1].ravel(); y=dev[:,1:].ravel(); m=~np.isnan(x)&~np.isnan(y)
print('lag1 corr of dev', np.corrcoef(x[m],y[m])[0,1], 'n', m.sum())
x=dev[:,:-5].ravel(); y=dev[:,5:].ravel(); m=~np.isnan(x)&~np.isnan(y)
print('lag5 corr of dev', np.corrcoef(x[m],y[m])[0,1])
```
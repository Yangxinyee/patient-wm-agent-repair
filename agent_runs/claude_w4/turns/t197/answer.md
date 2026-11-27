`tu` holds the tuning records, `df_s` maps state rows to (stay, hour), and `dr3` indexes the drug bundles within it. I'll now build the true dose surprise at each insulin bundle: rate minus the band mean (0, 1.06, 2.10, 3.15, 4.28) of the latest glucose at or before that hour, preferring finger. Then I'll test how well one linear readout of the prelude's normalised slot residual n3, shared across all 4 slots, recovers the surprise, the rate and the band mean (CV by stay). A high surprise R2 means a single unit-pair injection can carry the correct state-adjusted signal.

```python
from sklearn.model_selection import GroupKFold
dsr = df_s.iloc[dr3].reset_index(drop=True)
g_ = tu[tu.event.isin(['glucose_finger','glucose_lab'])][['stay_id','hour','event','value']].copy()
g_['pri'] = (g_.event=='glucose_finger').astype(int)
g_ = g_.sort_values(['stay_id','hour','pri']).groupby(['stay_id','hour']).tail(1)
q_ = dsr[['stay_id','hour']].copy(); q_['i'] = np.arange(len(q_))
q_ = q_.sort_values('hour'); g_ = g_.sort_values('hour')
m_ = pd.merge_asof(q_, g_[['stay_id','hour','value']].rename(columns={'hour':'gh'}), left_on='hour', right_on='gh', by='stay_id', direction='backward').sort_values('i')
Glat = m_.value.values
BM = np.array([0,1.06,2.10,3.15,4.28])
band_d = np.digitize(Glat, [150.0001,180.0001,220.0001,260.0001])
rI = np.nan_to_num(RI_d.astype(float)); iI = hasI_d==1
s_true = rI - BM[band_d]
print('ins bundles', iI.sum(), 'G nan', np.isnan(Glat).sum(), 'mean s_true on ins %.3f sd %.3f' % (s_true[iI].mean(), s_true[iI].std()))
print('band counts on ins', np.bincount(band_d[iI], minlength=5))
n3s = r2['n3'][:, :4, :].numpy()
def cvr2(X, y, grp, alpha=1.0):
    pr = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, grp):
        pr[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    return 1 - ((y-pr)**2).mean()/y.var()
Xp = n3s[iI].reshape(-1,192); grp = np.repeat(dsr.stay_id.values[iI], 4)
for nm_, t_ in [('s_true', s_true[iI]), ('rate', rI[iI]), ('bandmean', BM[band_d][iI])]:
    print(nm_, 'pooled-slot CV R2 %.3f' % cvr2(Xp, np.repeat(t_, 4), grp))
```
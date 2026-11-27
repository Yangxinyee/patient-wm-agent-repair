The model's insulin response is about four times too small. On the same 669 stay-hours, the record-based estimate of the true shift for insulin x2 is −14.2 mg/dL with a linear response, or −13.2 when the effect is scaled by glucose level. The original model gives −3.25. For x1/2 the estimate is +6.6 to +7.1 against the model's +1.77.

To find the cause I need to know whether the model's teacher-forced dose response is already too weak, or whether only the rollout fails. Both matter for the 0.005-nat budget, since the dose explains only about 1–2% of next-hour glucose variance.

First, though, I'll estimate the NE truth the same way:
1. Run local projections of the iid NE residual on arterial MAP at lags 1..12, plus the feedback on later NE rates.
2. Strip out the feedback to get the open-loop impulse response.
3. Apply it to batch 1's reached map_arterial rows.

A positive estimate several times the model's +0.62 would put NE in the same "under-responsive" situation as insulin. A small one would mean NE needs little change.

```python
n_full = d.merge(n_[['stay_id','ih','ne_hat','nres']], on=['stay_id','ih'], how='left').sort_values(['stay_id','ih']).reset_index(drop=True)
msh = {k: n_full.groupby('stay_id').map_arterial.shift(-k) for k in range(1,13)}
nsh = {k: n_full.groupby('stay_id').norepinephrine.shift(-k) for k in range(1,13)}
bn = n_full[n_full.nres.notna() & n_full.ma_prev.notna()].index
Tn={}; Dn={}
for k in range(1,13):
    y = msh[k].loc[bn]; ok = y.notna(); sub = n_full.loc[bn][ok]
    b_, s_ = ols((y[ok]-sub.map_arterial).values, [sub.map_arterial, sub.ma_prev, sub.ne_hat, sub.nres]); Tn[k]=(b_[-1], s_[-1])
    yi = nsh[k].loc[bn]; ok2 = yi.notna(); sub2 = n_full.loc[bn][ok2]
    b2_, s2_ = ols(yi[ok2].values, [sub2.map_arterial, sub2.ma_prev, sub2.ne_hat, sub2.nres]); Dn[k]=(b2_[-1], s2_[-1])
    print(k, "T=%.3f (%.3f)  D=%.4f (%.4f)  n=%d" % (Tn[k][0], Tn[k][1], Dn[k][0], Dn[k][1], ok.sum()))
hn = {}
for k in range(1,13): hn[k] = Tn[k][0] - sum(Dn[j][0]*hn[k-j] for j in range(1,k))
hnk = np.array([hn[k] for k in range(1,13)]); print("NE open-loop:", hnk.round(3), "sum", hnk.sum().round(2))
PT['m_level'] = PT['map_arterial'].groupby(level=0).ffill().groupby(level=0).bfill()
rows_n = de[(de.edit=='ne_x2')&de.reached][['stay_id','hour']]
estn = [true_shift_lin(PT.loc[s], 2.0, drug='norepinephrine', hk=hnk)[hh] for (s,hh) in rows_n.values]
print("NE x2 est:", np.mean(estn), " NE x1/2 est:", -0.5*np.mean(estn), " (model: +0.616, -0.816)")
```
The evaluation's definition is now settled.
- **Which rows:** there is a row only at hours 13–24 with a recorded reading of the outcome code (glucose_finger for insulin, map_arterial for NE).
- **Reached:** a row is "reached" once an insulin (or NE) record has started at or after hour 12 and before that hour.
- **Mean:** the reported mean is pooled over reached rows (−3.247 matches), not averaged per stay.
- **NE coverage:** NE stays without an arterial line contribute nothing to the NE readout.

Next I'll compute my record-based estimate of the true shift on exactly these rows. Each row's shift is the open-loop impulse response applied to the extra or removed insulin from hour 12 on, scaled by glucose level relative to the estimation sample, since the effect is roughly proportional to G. That gives a first "truth" number to set against the model's −3.25 and +1.77.

Ahead of the run, I expect x2 to come out several times larger in magnitude than the model's −3.25. A result near −3 would mean my impulse response, or how I apply it, is wrong.

```python
hk = np.array([h[k] for k in range(1,13)])
# reference glucose level for the estimated h (OLS weights ~ ires^2)
wts = d.loc[base].ires**2; Gref = np.average(d.loc[base].glucose_finger, weights=wts); print("Gref", Gref)
def true_shift_lin(stay_df_hourly, mult, drug='insulin', hk=hk, scale_col=None, Gref=Gref, start=12):
    # stay_df_hourly: index ih -1..24 with columns insulin, g_level
    r = stay_df_hourly[drug].values; hrs = stay_df_hourly.index.values
    out = {}
    for hh in range(13,25):
        tot = 0.0
        for s in range(start, hh):
            k = hh - s
            if k<=12:
                rs = r[hrs==s][0]
                sc = 1.0 if scale_col is None else stay_df_hourly[scale_col].values[hrs==s][0]/Gref
                tot += hk[k-1]*(mult-1)*rs*sc
        out[hh] = tot
    return out
PT['g_level'] = PT['glucose_finger'].groupby(level=0).ffill().groupby(level=0).bfill()
rows = de[(de.edit=='insulin_x2')&de.reached][['stay_id','hour']]
est = {'lin':[], 'linG':[]}
for (s, hh) in rows.values:
    sd_ = PT.loc[s]
    est['lin'].append(true_shift_lin(sd_, 2.0)[hh]); est['linG'].append(true_shift_lin(sd_, 2.0, scale_col='g_level')[hh])
print("x2 linear:", np.mean(est['lin']), " G-scaled:", np.mean(est['linG']))
print("x1/2 linear:", -0.5*np.mean(est['lin']), " G-scaled:", -0.5*np.mean(est['linG']))
```
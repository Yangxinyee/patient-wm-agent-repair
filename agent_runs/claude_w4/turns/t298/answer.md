The last output showed four things:

- **fwd_8 stays:** fwd_8 ran on a different 100 tuning stays. Only 30 overlap with sel100, so the sel100 truth (−12.96 / +4.94) does not apply to it.
- **Insulin response:** under the v4 edit, the insulin x2 shift is positive in every hour (about +1 to +5). The x½ shift is negative and grows to about −9 by hour 24. Both are the wrong direction, and x2 is the smaller of the two, so the response is not linear.
- **Drift:** the recorded-dose rollout drifts upward in almost every stay, from +4 mg/dL at hour 13 to +26 at hour 24.
- **NE response:** the NE response stays tiny.

Two explanations fit this. One: the injected signal has the opposite effect to what I intended, and the x2 response is squashed because predictions near the top glucose bin cannot rise much further. Two: the drift comes from a part of the edit that acts regardless of dose.

The two can be told apart from the records alone, without a forward pass. Using the recorded doses of these stays, I will regress the drift on the cumulative centred insulin signal, and compare the x2 shift with the true effect computed from my estimated response curve. I will also check whether stays without insulin drift too.

- A positive slope of drift on the centred signal, together with positive x2 shifts that grow with insulin given, means the channel's sign is reversed.
- Drift in insulin-free stays means a dose-independent leak.

```python
import numpy as np, pandas as pd
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 9 (runs 266-)\n- fwd_8 = v4(1,1) dose_edit both on 100 tuning stays (NOT sel100; 30 overlap). EDITED ins x2 +3.30, x1/2 -5.08, NE x2 +0.23, x1/2 -0.18;"
            " ORIGINAL -0.38/+0.21/+0.10/-0.31. Per-hour ins x2 shifts all positive (2-5), x1/2 negative growing to -9.4.\n"
            "  Factual (recorded-dose) rollout drift edited-orig grows +4 (h13) -> +26.5 mg/dL (h24), positive almost everywhere; NE drift +0.3->+2.7.\n"
            "  => v4 insulin effect has WRONG SIGN and/or a dose-independent leak. Checking with record regression (run 269).\n")
tun = pd.read_csv('/data/tuning.csv')
def ratemat(ev):
    d = tun[tun.event==ev][['stay_id','hour','rate']].copy(); d['h'] = np.floor(d.hour+1e-6).astype(int)
    return d.pivot_table(index='stay_id', columns='h', values='rate', aggfunc='sum').reindex(columns=range(26)).fillna(0.)
RI, RN = ratemat('insulin'), ratemat('norepinephrine')
print('IRF arrays present:', [k for k in ['hF','hA','hL','hC','irfF','irfA'] if k in globals()])
hF = np.array([-1.83,-2.12,-1.90,-1.55,-1.08,-.76,-.44,-.28,-.21,-.08,-.12,.06])
hA = np.array([3.71,2.71,1.34,1.06,.43,.79,.17,-.18,0,0,0,0])
def feats(row, R, rbar, h_irf):
    s, h = row.stay_id, int(row.hour)
    r = R.loc[s].values if s in R.index else np.zeros(26)
    ts = np.arange(12, h); rr = r[ts]
    return rr.sum(), (rr>0).sum(), rr.sum()-rbar*(rr>0).sum(), float(np.sum(rr*h_irf[h-1-ts]))
for ed, R, rbar, hh in [('insulin_x2', RI, 1.77, hF), ('ne_x2', RN, 0.52, hA)]:
    mm = m[m.edit==ed].copy()
    F = np.array([feats(r, R, rbar, hh) for r in mm.itertuples()])
    mm['sumr'], mm['cnt'], mm['cen'], mm['truth'] = F.T
    mm['dh'] = mm.hour-12
    X = np.c_[np.ones(len(mm)), mm.cen, mm.dh]; b = np.linalg.lstsq(X, mm.drift.values, rcond=None)[0]
    print(ed, 'drift ~ 1 + cen + dh coefs', b.round(3), ' mean drift where cnt==0:', mm[mm.cnt==0].drift.mean().round(2), 'n', (mm.cnt==0).sum())
    rr = mm[mm.reached]
    X2 = np.c_[np.ones(len(rr)), rr.sumr]; b2 = np.linalg.lstsq(X2, rr.shift_e.values, rcond=None)[0]
    print('   reached: mean shift E', rr.shift_e.mean().round(2), 'O', rr.shift_o.mean().round(2), ' TRUTH(x2)', rr.truth.mean().round(2),
          ' shift_E ~ 1 + sumr:', b2.round(3), ' corr(shift_E, truth)', np.corrcoef(rr.shift_e, rr.truth)[0,1].round(2))
    print('   truth by hour', rr.groupby('hour').truth.mean().round(1).values)
    globals()['mm_'+ed] = mm
```
The insulin readout is not the problem. Its bias is +0.03 at ×0.5 and −0.01 at ×2, and only degrades at ×3 (−0.18).

The insulin channel clearly reaches the rollout, but with the wrong sign:
- x1/2 accumulates steadily from −1.2 at hour 13 to −9.7 at hour 24, so a downward push builds up the way a dose effect should.
- x2 rises to about +5 and then falls back.
- Downward pushes come out about 3× stronger per unit than upward ones.
- NE shows no dose response.

So the "news" target is the wrong state direction. Most likely the value head reads injected news against an unchanged "last value" register and predicts regression back.

The principled target is the direction in `out` that moves the *next prediction* while the register is held fixed. I'll estimate it by ridge regression of the model's own predicted means (fwd_2) on the preceding `out` state. The controls are the register's bin, z-value and age, plus the gap. I'll do this separately for finger, lab, arterial and cuff.

Two checks follow. First, the fitted finger direction should predict a positive immediate effect for v4's insulin injection, matching the sign reversal I saw. If it doesn't, the regression is not trustworthy. Second, I'll build v5 at k = 1. Its insulin target makes finger and lab predictions both fall by 1 mg/dL per unit of (rate − r̄). Its NE target makes arterial and cuff both rise by 1 mmHg per unit of (rate − 0.52). Each is mapped to the mean slot via the median σ, and from there through the coda and `c_out` inversions as before.

```python
f2['hk']=f2.hour.round(4); rowkey = pd.DataFrame({'stay_id':sid,'hk':np.round(hr,4),'row':np.arange(len(sid))})
fr = f2.merge(rowkey, on=['stay_id','hk'], how='left')
recs = tsel[tsel.event.isin(ev4)][['stay_id','hour','event','value']].sort_values('hour')
W = {}; outs_all = X[:,:192]; m_all = H.mean(1)
for e in ev4:
    d = fr[fr.event==e].copy(); d = d[d.row.notna()]; d['row']=d.row.astype(int); d = d[prev_ok[d.row.values]]
    r_e = recs[recs.event==e][['stay_id','hour','value']].rename(columns={'hour':'h_reg','value':'v_reg'})
    d = d.sort_values('hour'); d['hq'] = d.hour - 1e-4
    d = pd.merge_asof(d.sort_values('hq'), r_e.sort_values('h_reg'), left_on='hq', right_on='h_reg', by='stay_id', direction='backward')
    d['has'] = d.v_reg.notna().astype(float); nmd = NUM[(e,'value')]
    d['zr'] = np.where(d.has>0, np.clip((d.v_reg-nmd['median'])/nmd['iqr'],-10,10), 0); d['b'] = np.where(d.has>0, 1+np.searchsorted(nmd['fine_bin_edges'], d.v_reg.fillna(0), side='left'), 0)
    d['lage'] = np.where(d.has>0, np.log1p(d.hour - d.h_reg), 0); d['ldt'] = np.log((hr[d.row.values]-hr[d.row.values-1])*3600+1)
    Ctl = np.c_[np.eye(17)[d.b.values.astype(int)], d.zr, d.lage, d.ldt]; O = outs_all[d.row.values-1]; y_ = d.predicted_mean.values
    Z_ = np.c_[Ctl, O]; lam = np.r_[np.full(Ctl.shape[1],1e-6), np.full(192, 1.0)]
    n_ = len(y_); perm = np.random.RandomState(0).permutation(n_); fold = perm % 5; pr = np.zeros(n_); prc = np.zeros(n_)
    for f in range(5):
        tr, te = fold!=f, fold==f
        bb = np.linalg.solve(Z_[tr].T@Z_[tr] + np.diag(lam), Z_[tr].T@y_[tr]); pr[te] = Z_[te]@bb
        bc = np.linalg.lstsq(Ctl[tr], y_[tr], rcond=None)[0]; prc[te] = Ctl[te]@bc
    bb = np.linalg.solve(Z_.T@Z_ + np.diag(lam), Z_.T@y_); W[e] = bb[-192:]
    print(e, 'n', n_, 'CV R2 ctl-only', round(1-np.var(y_-prc)/np.var(y_),4), 'ctl+out', round(1-np.var(y_-pr)/np.var(y_),4), '|w|', round(np.linalg.norm(W[e]),2))
gout, bout = L('core.out_norm.weight'), L('core.out_norm.bias')
sig_m = np.median(m_all[rows].std(-1)); print('sigma(mean slot) median at drug bundles', sig_m)
def dout_of_dm(dm, sm=sig_m): return gout*(dm - dm.mean())/sm
dmI4, dmN4 = Bfit[0], Bfit[1]
print('v4 predicted immediate pred change per unit: finger', W['glucose_finger']@dout_of_dm(dmI4), 'lab', W['glucose_lab']@dout_of_dm(dmI4), '| NE art', W['map_arterial']@dout_of_dm(dmN4), 'cuff', W['map_cuff']@dout_of_dm(dmN4))
def tgt_from(wa, wb, sgn):
    Wg = np.c_[wa, wb]; do = sgn*Wg@np.linalg.solve(Wg.T@Wg, np.ones(2)); dm = sig_m*do/gout; return dm - dm.mean(), do
tgt5_I, doI = tgt_from(W['glucose_finger'], W['glucose_lab'], -1.0); tgt5_N, doN = tgt_from(W['map_arterial'], W['map_cuff'], +1.0)
print('target check finger/lab', W['glucose_finger']@dout_of_dm(tgt5_I), W['glucose_lab']@dout_of_dm(tgt5_I), 'art/cuff', W['map_arterial']@dout_of_dm(tgt5_N), W['map_cuff']@dout_of_dm(tgt5_N))
print('|tgt5_I|', np.linalg.norm(tgt5_I), 'cos with old tgt_I', uv(tgt5_I)@uv(tgt_I), '| |tgt5_N|', np.linalg.norm(tgt5_N), 'cos old', uv(tgt5_N)@uv(tgt_N))
dc5I, e5I, _ = solve_dc(tgt5_I); dc5N, e5N, _ = solve_dc(tgt5_N); dh5I, dh5N = solve_h(dc5I), solve_h(dc5N)
print('solve errs', e5I, e5N, 'c_out resid', np.linalg.norm(Cout@dh5I-dc5I)/np.linalg.norm(dc5I), np.linalg.norm(Cout@dh5N-dc5N)/np.linalg.norm(dc5N))
tI5 = (sig_ref/c)*dh5I/spd; tN5 = (sig_ref/c)*dh5N/spd
np.savez('/work/v5_parts.npz', tI5=tI5, tN5=tN5, tgt5_I=tgt5_I, tgt5_N=tgt5_N, wF=W['glucose_finger'], wL=W['glucose_lab'], wA=W['map_arterial'], wC=W['map_cuff'], sig_m=sig_m)
```
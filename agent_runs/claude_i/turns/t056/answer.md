Placing has_last in column 0, log-age in column 3 and z_last_value in column 6 (status 0) fits much better. The within-code errors are 1.86 (glucose_finger), 1.40 (glucose_lab), 0.81 (map_arterial) and 1.17 (map_cuff). That is still not exact, so either another register feature is missing or my use of the prior mean for z (the tool averages 4 random draws) adds the gap.

Next, with columns (0, 3, 6) fixed, I will try putting a constant 1 in each remaining column (a has-value flag). I'll also try averaging probabilities over 16 prior z draws in place of the prior mean. Whichever change cuts the error most tells me which source remains. Sampling noise alone should leave a small, roughly symmetric error, with correlation near 0.999 within code.

```python
def run_var(extra_cols=(), zs=None, nsamp=0, seed=0):
    rng_ = np.random.default_rng(seed)
    if nsamp==0:
        mn,_ = head_value2(F, out_, m_, gv_, zp_, extra_cols)
    else:
        acc=0
        for i in range(nsamp):
            zz = zp_[:,:32] + np.exp(zp_[:,32:])*rng_.standard_normal((len(F),32))
            _,p = head_value2(F, out_, m_, gv_, zp_, extra_cols, zsamp=zz); acc = acc + p
        p = acc/nsamp; cen = np.stack([CEN[e] for e in evs]); mn = (p[:,1:]*cen).sum(1)
    mads = [np.abs(mn-y)[evs==e].mean() for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']]
    return np.round(mads,3), mn
def head_value2(F, out, m, g_view, zp, extra_cols=(), zsamp=None):
    codes = np.array([CID[e] for e in F.event])
    z = zp[:, :32] if zsamp is None else zsamp
    view = out + m + g_view + z @ P['zhW'].T + P['zhb']
    q = lnorm(view + P['te'][1], P['qw'], P['qb'])
    hasl = (~np.isnan(F.lastv.values)).astype(float)
    age = np.where(hasl>0, (F.hour.values - F.lasth.values)*3600, 0.0)
    zl = np.zeros(len(F)); lb = np.zeros(len(F), int)
    for ev in np.unique(evs):
        ix = (evs==ev)&(hasl>0); zl[ix] = zsc(ev,'value',F.lastv.values[ix]); lb[ix] = fbin(ev,'value',F.lastv.values[ix])
    feats = np.zeros((len(F),8)); feats[:,0]=hasl; feats[:,3]=np.log1p(age)/10; feats[:,6]=zl
    for c_ in extra_cols: feats[:,c_] = hasl
    wc = q + P['ce'][codes] + feats @ P['LW'].T + P['Lb'] + P['bv'][lb] + P['br'][0] + P['ls'][0]
    lg = mlp2(wc, 'attr_heads.value_bin') + P['tv'][codes*17 + lb]
    lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    cen = np.stack([CEN[e] for e in evs]); mn = (p[:,1:]*cen).sum(1)
    return mn, p
print("base", run_var()[0])
for c_ in [1,2,4,5,7]: print("extra has-col", c_, run_var((c_,))[0])
print("16 z-samples", run_var(nsamp=16)[0])
```
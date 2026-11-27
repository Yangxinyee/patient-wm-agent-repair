The titration rule is now clear. For running infusions (previous rate 0.2–1.0):

| Arterial MAP (sharp cut-offs; 65 and 75 are mixed) | Multiplier on the rate | Stop probability |
|---|---|---|
| < 65 | about ×1.09 | — |
| 65–75 | ×1.00 (hold) | ~1%/h |
| > 75 | about ×0.9 | ~19%/h |

On top of that there is lognormal noise with log-sd about 0.29, then rounding and the 1.0 cap. The noise is large relative to the systematic part, so there is plenty of exogenous variation. However, it persists in all later rates.

Next I will estimate the one-hour effect. For running infusions below the cap, I will regress MAP at t+1 on the NE rate at t, controlling flexibly for the policy inputs: the previous rate and the MAP reading at t by source. Given those inputs the rate varies only through the noise, so the coefficient is causal for the 1-hour lag. I will also run a placebo on MAP at t−1 and t−2. A positive and stable effect would mean norepinephrine raises MAP. A value near zero would point towards "absent".

```python
P['NEp'] = P.groupby(level=0).NE.shift(1)
P['ma'] = P.map_arterial
P['src'] = np.where(P.map_arterial.notna(),1,np.where(P.map_cuff.notna(),2,0))
P['mr'] = P.map_arterial.fillna(P.map_cuff)
def ne_lag_reg(L, outcome='map_arterial', restrict_cap=True):
    Y = P.groupby(level=0)[outcome].shift(-L)
    q = P.assign(Y=Y)
    q = q[(q.index.get_level_values('h')>=1) & q.NEp.notna() & q.NE.notna() & q.Y.notna() & q.mr.notna()]
    if restrict_cap: q = q[(q.NEp<1.0)]
    m = q.mr.values; ne_p = q.NEp.values
    X = [np.ones(len(q))]
    nm=['c']
    # flexible controls: MAP bands (by src) x linear, NEp spline
    for s in [1,2]:
        for lo,hi in [(0,65),(65,75.5),(75.5,300)]:
            ind = ((q.src.values==s)&(m>=lo)&(m<hi)).astype(float)
            X += [ind, ind*m]; nm += [f's{s}_{lo}', f's{s}_{lo}_m']
    X += [ne_p, ne_p**2, np.log(ne_p)]; nm += ['nep','nep2','lnep']
    X = X[0:1]+X[2:]  # drop const (bands cover all) 
    nm = nm[1:]
    X += [q.NE.values]; nm += ['NE']
    r = ols(q.Y.values, X, nm).loc['NE']
    return len(q), r.b, r.se
for L in [-2,-1,1,2,3]:
    print(L, ne_lag_reg(L))
```
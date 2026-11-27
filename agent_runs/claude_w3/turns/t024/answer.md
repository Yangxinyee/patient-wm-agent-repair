Arterial MAP is a clean outcome. Arterial lines are a stay-level feature, present in about 48% of stays, and those stays are sicker (lower MAP, norepinephrine in 46% of hours vs 19%). Within line stays an arterial reading appears in 95% of hours at random, regardless of MAP or norepinephrine. The norepinephrine innovation (sd 0.077 mcg/kg/min) passes both validity checks: a simultaneous cuff reading does not predict it given the arterial MAP (corr 0.01), and neither does the previous hour's MAP (corr 0.004). So the titration noise is as good as random given the recorded history.

Next I will run the IV impulse response for norepinephrine on map_arterial, the same way as for insulin:
- origins: hours with norepinephrine running in line stays;
- outcome: arterial MAP L hours later;
- endogenous regressors: the rates r_t … r_{t+L−1}, instrumented by their innovations;
- fixed effects: hour and the policy cell (MAP bin × previous-rate bin).

A positive coefficient means norepinephrine raises MAP. Near zero means no effect, which would make the model's small answers close to right for norepinephrine. A negative coefficient would be surprising but possible in this simulation.

```python
# arrays per stay for norepi analysis (all stays; outcome only in line stays)
x = x.sort_values(['stay_id','h'])
SS = x.stay_id.nunique()
def A(col): return x[col].values.reshape(SS, 25)
MAPA = A('map_arterial'); NR = A('nr'); ZR = A('zr'); MAPP = A('map_pol'); PREV = A('ne_prev')
KEYC = pd.factorize(key.reindex(x.index))[0].reshape(SS,25)
def iv_ne(L, Y=MAPA):
    rows = []
    for t in range(0, 25-L):
        ok = (NR[:,t]>0) & ~np.isnan(Y[:,t+L])
        if ok.sum()==0: continue
        rows.append(np.column_stack([np.full(ok.sum(), t), KEYC[ok,t], Y[ok,t+L], NR[ok,t:t+L], ZR[ok,t:t+L]]))
    Am = np.vstack(rows)
    grp = pd.Series(list(zip(Am[:,0].astype(int), Am[:,1].astype(int)))).factorize()[0]
    df = pd.DataFrame(Am[:,2:])
    dm = (df - df.groupby(grp).transform('mean')).values
    yv = dm[:,0]; Dv = dm[:,1:1+L]; Zv = dm[:,1+L:]
    PZ = Zv @ np.linalg.lstsq(Zv, Dv, rcond=None)[0]
    b = np.linalg.lstsq(PZ, yv, rcond=None)[0]
    res = yv - Dv@b; V = np.linalg.inv(PZ.T@PZ)*res.var()
    # first-stage strength for the lag-L regressor
    return b, np.sqrt(np.diag(V)), len(yv)
for L in [1,2,3,4,6,8,10,12]:
    b, se, n = iv_ne(L)
    print(f"L={L:2d} n={n:6d} coef r_t (lag L): {b[0]:7.2f} ({se[0]:.2f}); by lag (L..1):", np.round(b,1))
```
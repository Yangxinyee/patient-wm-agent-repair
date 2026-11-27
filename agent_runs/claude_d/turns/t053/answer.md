EPSN is clean. It is uncorrelated with current MAP, previous MAP and the previous rate, and the titration pattern looks like a step-function f plus noise. So both estimates should be unbiased, and the gap between γ₁ = 1.76 (IV) and about 3.3 (run 46) must come from how the samples or regressors were built. Candidates are the cap at 1.0, or the A_log clipping at 0.08 weakening the first stage.

I'll compute the lag-1 reduced form (MA_{t+1} on EPSN_t) and the first stage (A_t on EPSN_t) directly, with and without the uncapped restriction, and compare log and linear treatment codings.

```python
rows=[]
for t in range(1,24):
    s_=(EPSN[:,t]!=0)&~np.isnan(MA[:,t+1])&~np.isnan(MA[:,t])
    idx=np.where(s_)[0]
    rows.append(np.column_stack([EPSN[idx,t], MA[idx,t+1], MA[idx,t], A_log[idx,t], NOR[idx,t], NOR[idx,t-1]]))
Q=np.vstack(rows); eq, y1q, y0q, alq, rq, rqm1 = Q.T
def slope(y,x,ctrl=None):
    X = np.column_stack([x] + ([] if ctrl is None else [ctrl]))
    return ols(y, X)[0][1]
for name, s_ in [('all',np.ones(len(eq),bool)),('uncapped r<.97',rq<0.97),('capped r>=.97',rq>=0.97),('r>0.1', (rq>0.1)&(rq<0.97))]:
    rf = slope(y1q[s_], eq[s_], y0q[s_]); fsl = slope(alq[s_], eq[s_], y0q[s_]); fsr = slope(rq[s_], eq[s_], y0q[s_])
    print('%-16s n=%6d RF=%.2f FSlog=%.3f -> per log %.2f | FSrate=%.3f -> per rate %.2f'%(name, s_.sum(), rf, fsl, rf/fsl, fsr, rf/fsr))
```
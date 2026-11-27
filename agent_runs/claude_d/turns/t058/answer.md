The NE policy is more detailed than I assumed:
- Rates are rounded to 0.01 and can be as low as 0.01, so there is no hard stop at 0.08.
- The cap is exactly 1.0, and 13.6% of running hours sit at it.
- The titration factor has more steps: about e^{+0.25} below MAP 56, e^{+0.14} for 56–64, 1 for 68–76, and e^{−0.13} from 76 up.
- Starting rates depend on MAP but are noisy.

So I'll avoid a hand-made policy model and build the instrument non-parametrically. For each titration hour, Z = g(rate_t) − E[g(rate_t) | cell], where the cells are fine bins of the expected rate crossed with 4-mmHg MAP bands. That is mean-zero given everything the policy uses, so it is valid over the full range of rates, cap and stops included.

To pick the dose-response shape, I'll fit power codings g(r) = r^p (with p = 0 as a log coding). Each fit is an overidentified 2SLS whose instruments are split by expected-rate stratum, and I'll compare the Hansen J statistic across p. The best-fitting shape should have the smallest J. I'll also print the lag-1 effects by stratum.

```python
# nonparametric instrument for titration rows
Ztr = np.zeros_like(NOR); ER = np.full_like(NOR, np.nan)
recs=[]
for t in range(1,Hn):
    pre=(NOR[:,t-1]>0)&~np.isnan(Mdec[:,t])
    idx=np.where(pre)[0]
    er = NOR[idx,t-1]*np.exp(f_ne(Mdec[idx,t])); ER[idx,t]=er
    recs.append(np.column_stack([idx, np.full(len(idx),t), er, Mdec[idx,t], NOR[idx,t]]))
Rr=np.vstack(recs); ri,rtt,rer,rmd,rnow = Rr.T; ri=ri.astype(int); rtt=rtt.astype(int)
erb = np.digitize(np.log(rer), np.quantile(np.log(rer), np.linspace(0,1,41)[1:-1]))
mdb = np.clip((rmd//4).astype(int), 10, 23)
dfr = pd.DataFrame({'c':erb*100+mdb, 'r':rnow})
Zlin = (dfr['r'] - dfr.groupby('c')['r'].transform('mean')).values
Ztr[ri,rtt] = Zlin
strat = np.zeros_like(NOR, int); strat[ri,rtt] = np.digitize(rer,[0.25,0.5])+1
print('titration rows', len(ri), 'Z sd', Zlin.std().round(3), ' strata counts', np.bincount(strat[ri,rtt]))
def iv2(Y, A, Zi, St, K, nst=3):
    ys=[];Xs=[];Zs=[];Cs=[]
    for t in range(K+1, Y.shape[1]):
        ok=~np.isnan(Y[:,t])
        Xs.append(np.column_stack([A[ok,t-k] for k in range(1,K+1)]))
        Zs.append(np.column_stack([Zi[ok,t-k]*(St[ok,t-k]==s) for k in range(1,K+1) for s in range(1,nst+1)]))
        Cs.append(np.column_stack([np.ones(ok.sum()), np.nan_to_num(Y[ok,t-K-1],nan=np.nanmean(Y[:,t-K-1])), np.isnan(Y[ok,t-K-1]), A[ok,t-K-1], np.full(ok.sum(),t)]))
        ys.append(Y[ok,t])
    y=np.concatenate(ys); X=np.vstack(Xs); Zm=np.vstack(Zs); Cm=np.vstack(Cs)
    res=lambda M: M - Cm@lstsq(Cm,M,rcond=None)[0]
    y,X,Zm=res(y),res(X),res(Zm)
    # 2SLS
    PZX = Zm@lstsq(Zm, X, rcond=None)[0]
    b = lstsq(PZX, y, rcond=None)[0]
    u = y - X@b
    g_ = Zm*u[:,None]; S_ = g_.T@g_/len(y); gbar = g_.mean(0)
    J = len(y)*gbar@np.linalg.solve(S_, gbar)
    return b, J
for p in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
    Ap = np.where(NOR>0, (np.log(np.maximum(NOR,0.01)/0.01) if p==0 else NOR**p), 0.0)
    b,J = iv2(MA, Ap, Ztr, strat, 8)
    print('p=%.2f J=%.1f (df=%d) gamma'%(p,J,8*3-8), b.round(2))
```
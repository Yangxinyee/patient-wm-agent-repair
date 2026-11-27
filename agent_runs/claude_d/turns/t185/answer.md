WC0 is exactly the with_code at the prior-mean z (difference 0). But a single prior draw of z moves with_code by rms 0.23, against an rms of 0.82 for with_code itself. That is a lot of noise. The tool's factual NLL and its rollouts use sampled z, so a decoder fitted on the z = mean inputs could pick up z-driven noise and look better offline than it is.

I will refit the decoders on z-sampled with_code: four prior draws per reading, stay-grouped five-fold CV. For each target I report the CV R² and the slope of the decoded value on the truth (this slope is my estimate of the counterfactual sensitivity κ):

- insulin surprise at lags 1 and 2, glucose rows;
- NE deviation and NE surprise at lag 1, MAP rows.

If the R² values hold up under z noise (about 0.3 for insulin lag 1), the head-edit plan stays viable. If they fall a lot, I have to project out the z-sensitive directions.

```python
def aug_wc(rows, nd=4, seed=1):
    rg_ = np.random.default_rng(seed)
    out = []
    for d in range(nd):
        z = HI['zmu'][rows] + np.exp(HI['zls'][rows])*rg_.standard_normal((len(rows),32))
        out.append(wc_from(z, HI['ctx'][rows], HI['reg'][rows]))
    return np.stack(out)  # (nd, n, 192)
def cvdec_aug(rows, Y, alpha, nd=4):
    Xa = aug_wc(rows, nd)            # nd,n,192
    grp = RD2['stay_id'].values[rows]
    mu = Xa.reshape(-1,192).mean(0); sd = Xa.reshape(-1,192).std(0)+1e-6
    Yh = np.zeros((nd,)+Y.shape)
    for trn,tst in GroupKFold(5).split(Y, Y, grp):
        Xt = ((Xa[:,trn]-mu)/sd).reshape(-1,192); Yt = np.tile(Y[trn],(nd,1))
        mdl = Ridge(alpha=alpha).fit(Xt, Yt)
        for d in range(nd): Yh[d,tst] = mdl.predict((Xa[d,tst]-mu)/sd)
    r2 = 1-((Y-Yh)**2).sum((0,1))/(nd*((Y-Y.mean(0))**2).sum(0))
    kap = [np.polyfit(np.tile(Y[:,j],nd), Yh[:,:,j].ravel(), 1)[0] for j in range(Y.shape[1])]
    return np.round(r2,3), np.round(kap,3)
YgI = RD2.loc[rg, ['ei1','ei2']].values
YmN = RD2.loc[rm, ['D_ne_1','eps_ne_1']].values
for a in (1, 10):
    print('a',a,'G', cvdec_aug(rg, YgI, a), 'M', cvdec_aug(rm, YmN, a))
```
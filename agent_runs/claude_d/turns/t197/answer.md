The coverage numbers are useful: 58% of glucose readings follow an insulin hour and 36% of MAP readings follow an NE hour. So presence gating would remove a large share of the injected noise.

The AND gates themselves came out wild, though, with values from -20 to +20 instead of about ±1. I fitted the joint gates by unregularised least squares on the z = mean with_code. They then extrapolate badly once z is sampled (the earlier family gate happened to survive this). The decoder call also failed on a single-column shape issue.

Next I refit:

1. the presence gates on z-sampled with_code with ridge regularisation, using a simple "drug ran in the previous hour" target on all rows (a state property, not a code property);
2. the family gate in the same way.

I then check the AND margins (they need to stay clearly positive on target rows and clearly negative elsewhere). I also refit the decoders on present-drug rows only, with the shape fix.

```python
def ridge_gate(target_pos, alpha=10.0):
    y = np.tile(np.where(target_pos, 1.0, -1.0), 4)
    X = WCd.reshape(-1,192); mu = X.mean(0); sd = X.std(0)+1e-6
    m = Ridge(alpha=alpha).fit((X-mu)/sd, y)
    w = m.coef_/sd; c = m.intercept_ - mu@w
    return np.concatenate([w,[c]]), WCd@w + c
for al in (10., 1000.):
    vF, gF = ridge_gate(ism, al); vI, gI = ridge_gate(ins1, al); vN, gN = ridge_gate(ne1p, al)
    andI = -gF + gI - 1; andN = gF + gN - 1
    for nm, a_, pos in [('I', andI, isg&ins1), ('N', andN, ism&ne1p)]:
        print(al, nm, 'pos min/p1', round(float(a_[:,pos].min()),2), round(float(np.percentile(a_[:,pos],1)),2), 'neg max/p99', round(float(a_[:,~pos].max()),2), round(float(np.percentile(a_[:,~pos],99)),2))
def cvdec_aug2(rows, Y, alpha):
    Y = Y.reshape(len(rows), -1)
    Xa = aug_wc(rows, 4); grp = RD2['stay_id'].values[rows]
    mu = Xa.reshape(-1,192).mean(0); sd = Xa.reshape(-1,192).std(0)+1e-6
    Yh = np.zeros((4,)+Y.shape)
    for trn,tst in GroupKFold(5).split(Y, Y, grp):
        mdl = Ridge(alpha=alpha).fit(((Xa[:,trn]-mu)/sd).reshape(-1,192), np.tile(Y[trn],(4,1)))
        for d in range(4): Yh[d,tst] = mdl.predict((Xa[d,tst]-mu)/sd).reshape(len(tst),-1)
    r2 = 1-((Y-Yh)**2).sum((0,1))/(4*((Y-Y.mean(0))**2).sum(0))
    kap = [np.polyfit(np.tile(Y[:,j],4), Yh[:,:,j].ravel(), 1)[0] for j in range(Y.shape[1])]
    return np.round(r2,3), np.round(kap,3)
print('present rows dec I', cvdec_aug2(ri, RD2.loc[ri,['ei1','ei2']].values, 1.0), 'N', cvdec_aug2(rn, RD2.loc[rn,['D_ne_1']].values, 1.0))
```
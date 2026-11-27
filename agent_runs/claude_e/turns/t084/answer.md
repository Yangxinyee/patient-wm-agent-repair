On the tuning stays the model's coefficients are [−0.16, 0.09, 0.32, 0.14, −0.08, −0.37]. Pooled with the training stays they are [−0.52, 0.31, 0.52, 0.23, −0.01, −0.27] (se 0.13). In either case the model misses most of the true effect, so the missing part is roughly γ ≈ [−1.5, −1.25, −0.4, 0.3, 0.7, 0.7].

For the rollout to reproduce the true response, the correction must use dose coefficients conditional on the full glucose history the model sees. Otherwise the model's own propagation of a glucose shift double-counts or under-counts the effect.

Next I will check how stable the coefficients are when I add more glucose history (gf at t−2 and t−3, lab at t−1), for both the true outcomes and the model's predictions. I will also compare the glucose-history coefficients between truth and model. Stable dose coefficients and matching history coefficients would support the one-step correction design.

```python
def build_ins_design2(G, stays, nlag=6):
    D = build_ins_design(G, stays, nlag)
    gf = G['gf'].loc[stays]; gl = G['gl'].loc[stays]
    hrs_ = np.array(G['gf'].columns); idx = {h:i for i,h in enumerate(hrs_)}
    gfv = gf.values; glv = gl.values
    ii = np.array([idx[h] for h in D.hour.values])  # outcome hour index; t = ii-1
    s = D.s.values
    D['g2'] = gfv[s, ii-3]; D['g3'] = gfv[s, ii-4]; D['lab1'] = glv[s, ii-2]
    return D
def ins_regress2(D, ycol='y', nlag=6, extra=True):
    Xc = []
    for k in range(5):
        m = (D.tier==k).astype(float).values; c = D.g0.values - [140,165,200,260,330][k]
        Xc += [m, m*c, m*c**2]
    for col in (['g1','lab'] + (['g2','g3','lab1'] if extra else [])):
        v = D[col].values; Xc += [np.nan_to_num(v - D.g0.values), np.isnan(v).astype(float)]
    Xd = [D[f'dev{j+1}'].values for j in range(nlag)]
    Xm = np.column_stack(Xc + Xd); yv = D[ycol].values
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    res = yv - Xm@beta; cov = res.var()*np.linalg.pinv(Xm.T@Xm)
    return beta, np.sqrt(np.abs(np.diag(cov)))
Dtr2 = build_ins_design2(Gtr, ins_stays_tr)
for ex in [False, True]:
    b_, se_ = ins_regress2(Dtr2, 'y', extra=ex); print('TRUE extra=%s'%ex, 'dev', b_[-6:].round(2), ' hist coefs', b_[15:-6:2].round(3))
DA2 = build_ins_design2(Gtr, selA); DA2['stay']=np.array(selA)[DA2.s.values]; DA2 = DA2.merge(RA[RA.event=='glucose_finger'][['stay','hour','pm']], on=['stay','hour'])
D12 = build_ins_design2(Gtu, s1); D12['stay']=np.array(s1)[D12.s.values]; D12 = D12.merge(R1[R1.event=='glucose_finger'][['stay','hour','pm']], on=['stay','hour'])
DP2 = pd.concat([DA2, D12])
for ex in [False, True]:
    b_, se_ = ins_regress2(DP2, 'pm', extra=ex); bo_, _ = ins_regress2(DP2, 'y', extra=ex)
    print('MODEL extra=%s'%ex, 'dev', b_[-6:].round(2), '+-', se_[-6:].round(2), ' hist', b_[15:-6:2].round(3), '| obs same rows hist', bo_[15:-6:2].round(3))
```
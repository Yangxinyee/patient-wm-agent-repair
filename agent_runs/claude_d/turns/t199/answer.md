On present-drug rows the deficits are:

- **Insulin:** -1.6, -1.33, -0.39 at lags 1 to 3.
- **NE:** jointly 3.15, 0.53, -2.13; alone, the lag-1 deviation term is 2.2.

Surprise variances on these rows are 0.38 (insulin lag 1), 0.47 (lag 2) and 0.059 (NE deviation).

My cost model is ΔMSE ≈ κ·var·a·(a - 2·c_factual). For NE it says the full lag-1 response (3.2 per unit) would cost about +0.004 overall. About 68% of it would cost about +0.0015.

So I build edit E2:

- **Insulin channel** (units 31 and 143): gated to glucose readings with insulin in the previous hour (threshold 0.135, M = 100). Decoders for lag-1 and lag-2 surprise use weights -1.6/0.57 and -1.33/0.24.
- **NE channel** (units 54 and 9): gated to MAP readings with NE in the previous hour (threshold 0, M = 30). The lag-1 deviation decoder has weight 8.7.
- **Tilts:** exponential, fitted on the present rows.

I evaluate the four-draw NLL change by code and the residual slopes after the edit. If total ΔNLL is around +0.003 or less, I will test the rollout with the tool.

```python
vF, _ = ridge_gate(ism, 10.); vI, _ = ridge_gate(ins1, 10.); vN, _ = ridge_gate(ne1p, 10.)
gI_vec = np.concatenate([-vF[:192] + vI[:192], [-vF[192] + vI[192] - 1]])   # AND gluc & ins1
gN_vec = np.concatenate([ vF[:192] + vN[:192], [ vF[192] + vN[192] - 1]])   # AND map & ne1
Wd_i2, w0_i2 = fit_dec(ri, RD2.loc[ri,['ei1','ei2']].values)
Wd_n2, w0_n2 = fit_dec(rn, RD2.loc[rn,['D_ne_1']].values); Wd_n2 = np.asarray(Wd_n2).reshape(192,-1); w0_n2 = np.ravel(w0_n2)
t_Gp = exp_tilt2(ri); t_Mp = exp_tilt2(rn)
def build2(a_ins=(-2.8,-5.5), a_ne=8.7, MI=100., thI=0.135, MN=30., thN=0.0, lam_g=5., lam_m=2., units=((31,143),(54,9))):
    W1e, b1e, W2e, b2e = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    for j in sum(units,()): b2e += W2[:,j]*mh4[j]
    w_ins = Wd_i2@np.array(a_ins); c_ins = w0_i2@np.array(a_ins)
    w_ne = Wd_n2[:,0]*a_ne; c_ne = w0_n2[0]*a_ne
    for (ua,ub), gv, M, th, w, c, lam, t in [(units[0], gI_vec, MI, thI, w_ins, c_ins, lam_g, t_Gp), (units[1], gN_vec, MN, thN, w_ne, c_ne, lam_m, t_Mp)]:
        W1e[ua] = M*gv[:192] + w/lam; b1e[ua] = M*(gv[192]-th) + c/lam
        W1e[ub] = M*gv[:192];         b1e[ub] = M*(gv[192]-th)
        W2e[:,ua] = lam*t; W2e[:,ub] = -lam*t
    return W1e, b1e, W2e, b2e
E2 = build2()
ne2, dm2 = eval_edit(E2)
print('dNLL all', round(float(ne2.mean()-base4),5), {c: round(float((ne2-nb)[evc==c].mean()),4) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
print('dm sd present I/N', round(float(dm2[ri].std()),2), round(float(dm2[rn].std()),2), 'absent gluc/map', round(float(dm2[isg&~ins1].std()),3), round(float(dm2[ism&~ne1p].std()),3))
r_new = RD2['observed'].values - (RD2['pred0'].values + dm2)
RD2['res2'] = r_new
print('I resid slopes after', reg(isg&ins1, ['ei1','ei2','ei3'], y='res2')[1], 'N', reg(ism&ne1p, ['D_ne_1','D_ne_2','D_ne_3'], y='res2')[1])
```
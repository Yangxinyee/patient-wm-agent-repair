Each code has one fixed set of 16 bin centres. The arterial MAP bins run from 33 to 101 mmHg, 3 to 7 mmHg wide in the normal range, with a top bin at 101. So a logit tilt can shift the MAP mean smoothly in the relevant range.

Now I build the complete gated two-channel edit offline:

- **Channels.** Units 31 and 143 carry insulin (glucose-gated). Units 54 and 9 carry NE (MAP-gated). In each pair, one unit is signal plus gate and the other is gate only.
- **Gate multiplier.** M = 40.
- **Tilts.** Per-family tilt vectors are fitted so that a unit of channel output shifts that family's mean by about one unit.
- **Decoders.** z-augmented ridge fits of the insulin surprise at lags 1 and 2 and of the NE deviation at lag 1.

For a first version I use full-deficit scaling (-1.4/κ for each insulin lag, 3.2/κ for NE). Then I evaluate under the four-draw z sampling:

1. the NLL change by code;
2. how linear the mean shift is for injected shifts of -15 to +15;
3. that the other family stays untouched.

```python
def fit_dec(rows, Y, alpha=1.0, nd=4, seed=11):
    Xa = aug_wc(rows, nd, seed); X = Xa.reshape(-1,192); Yt = np.tile(Y,(nd,1))
    mu = X.mean(0); sd = X.std(0)+1e-6
    m = Ridge(alpha=alpha).fit((X-mu)/sd, Yt)
    W = (m.coef_/sd).T if Y.ndim>1 else m.coef_/sd     # 192 x k
    w0 = m.intercept_ - mu@W
    return W, w0
Wd_ins, w0_ins = fit_dec(rg, YgI)                    # eps lag1, lag2
Wd_ne, w0_ne = fit_dec(rm, RD2.loc[rm,['D_ne_1']].values)
P4 = np.exp(lgd - lgd.max(2,keepdims=True)); P4 /= P4.sum(2,keepdims=True)
def fit_tilt(rows, rho=1e-4):
    m = (P4[:,rows,1:]*cen[rows][None]).sum(2,keepdims=True)
    Cm = (P4[:,rows,1:]*(cen[rows][None]-m)).reshape(-1,16)
    t = np.linalg.solve(Cm.T@Cm/len(Cm)+rho*np.eye(16), Cm.T@np.ones(len(Cm))/len(Cm))
    return np.concatenate([[0.],t])
t_G = fit_tilt(rg); t_M = fit_tilt(rm)
def build(a_ins=(-4.4,-7.4), a_ne=21.3, M=40., lam_g=5., lam_m=2., units=((31,143),(54,9))):
    W1e, b1e, W2e, b2e = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    for j in sum(units,()): b2e += W2[:,j]*mh4[j]
    w_ins = Wd_ins@np.array(a_ins); c_ins = w0_ins@np.array(a_ins)
    w_ne = Wd_ne[:,0]*a_ne; c_ne = w0_ne[0]*a_ne
    for (ua,ub), gs, w, c, lam, t in [(units[0], -1, w_ins, c_ins, lam_g, t_G), (units[1], 1, w_ne, c_ne, lam_m, t_M)]:
        W1e[ua] = gs*M*vg[:192] + w/lam; b1e[ua] = gs*M*vg[192] + c/lam
        W1e[ub] = gs*M*vg[:192];         b1e[ub] = gs*M*vg[192]
        W2e[:,ua] = lam*t; W2e[:,ub] = -lam*t
    return W1e, b1e, W2e, b2e
def eval_edit(E):
    W1e,b1e,W2e,b2e = E
    lge = gelu(WCd@W1e.T+b1e)@W2e.T + b2e + HI['trans'][None]
    ne_ = nll_avg(lge)
    pe = np.exp(lge-lge.max(2,keepdims=True)); pe/=pe.sum(2,keepdims=True)
    me = (pe[:,:,1:]*cen[None]).sum(2).mean(0); m0_ = (P4[:,:,1:]*cen[None]).sum(2).mean(0)
    return ne_, me-m0_
E1 = build()
ne1, dm1 = eval_edit(E1)
print('dNLL all', round(float(ne1.mean()-base4),5), {c: round(float((ne1-nb)[evc==c].mean()),4) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
print('dmean sd gluc/map', round(float(dm1[isg].std()),2), round(float(dm1[ism].std()),2))
```
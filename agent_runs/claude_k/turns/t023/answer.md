The deconvolved responses are smooth and fit simple forms.

**Insulin:** h_k = -1.99, -2.37, -2.10, -1.71, -1.21, -0.93, -0.68, -0.52 is almost exactly -1.985 × k × 0.6^(k-1). That is a two-stage first-order process with both rates about 0.6 per hour. It implies about -12.4 mg/dL per sustained unit/h.

**Norepinephrine:** h_k = 3.33, 2.62, 1.75, 1.03, 0.44, 0.28, ~0, 0.1 decays faster, about +9.5 mmHg per sustained 1 mcg/kg/min.

Before computing counterfactuals I'll check whether these effects are homogeneous. Averaging h × dose over stays assumes the effect doesn't vary with things correlated with the doses. Using the double-ML residuals at lags 1–2, I'll compare the insulin effect by norepinephrine status, arterial line, hour, glucose level and dose size. The same goes for norepinephrine by rate level (linearity up to the cap), arterial line and glucose. I'll also test cross-effects (insulin→MAP, norepinephrine→glucose). Similar effects across subgroups would let me use one linear impulse response per drug. A clear dependence on rate level, especially near norepinephrine's cap, would change the x2 and x3 extrapolations.

```python
def dml_sub(B, X, rd, ycol, masks):
    out={}
    m0 = B[ycol].notna().values
    ry = np.full(len(B), np.nan); ry[m0] = crossfit_resid(X[m0], B[ycol].values[m0])
    for name, mk in masks.items():
        m = m0 & mk
        b = (rd[m]@ry[m])/(rd[m]@rd[m]); e = ry[m]-b*rd[m]; se = np.sqrt((e**2*rd[m]**2).sum())/(rd[m]@rd[m])
        out[name]=(int(m.sum()), round(b,2), round(se,2))
    return out
Bi = build('ins'); Xi = Bi[featcols]; rdi = crossfit_resid(Xi, Bi.dose.values)
mi = {'all':np.ones(len(Bi),bool),'ne_on':(Bi.rp>0).values,'ne_off':(Bi.rp==0).values,'art':(Bi.has_art>0).values,'noart':(Bi.has_art==0).values,
      't<12':(Bi.t<12).values,'t>=12':(Bi.t>=12).values,'g<180':(Bi.glast<180).values,'g180-220':((Bi.glast>=180)&(Bi.glast<220)).values,'g>=220':(Bi.glast>=220).values}
print("insulin lag1:", dml_sub(Bi, Xi, rdi, 'y1', mi))
print("insulin lag2:", dml_sub(Bi, Xi, rdi, 'y2', mi))
Bn = build('ne'); Xn = Bn[featcols]; rdn = crossfit_resid(Xn, Bn.dose.values)
mn = {'all':np.ones(len(Bn),bool),'rp<.3':(Bn.rp<.3).values,'rp.3-.6':((Bn.rp>=.3)&(Bn.rp<.6)).values,'rp.6-.9':((Bn.rp>=.6)&(Bn.rp<.9)).values,'rp>=.9':(Bn.rp>=.9).values,
      'ins_on':(Bn.ip>0).values,'ins_off':(Bn.ip==0).values,'t<12':(Bn.t<12).values,'t>=12':(Bn.t>=12).values,'m<65':(Bn.mlast<65).values,'m>=65':(Bn.mlast>=65).values}
print("norepi lag1:", dml_sub(Bn, Xn, rdn, 'y1', mn))
print("norepi lag2:", dml_sub(Bn, Xn, rdn, 'y2', mn))
```
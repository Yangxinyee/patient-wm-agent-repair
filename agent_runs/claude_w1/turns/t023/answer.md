No cross-effects were found:
- Norepinephrine has no detectable effect on glucose; all estimates are within about 1.5 SE of 0.
- Insulin has no detectable effect on MAP (about 0 ± 0.03).

The norepinephrine effect is linear across rate levels. The one-hour effect is about +3.3 to +3.6 mmHg per mcg/kg/min, the same at low and high rates, so there's no sign of saturation up to 1.0.

Next I'll turn the total effects into fixed-dose impulse responses h(k) by deconvolution: T_out(k) = Σ_j h(k−j)·T_dose(j), with T_dose(0) = 1. The readout holds later doses at recorded ×2, so h is what's needed. I'll bootstrap over stays for standard errors and save the result to the notebook. I'll also check that cuff MAP has no bias relative to arterial MAP.

```python
def lp_series(df, outcol, shockcol, ctrl_fn, dosecol, K=12):
    TG=[]; TI=[]
    for k in range(1,K+1):
        mk = df[f'{outcol}+{k}'].notna()
        sub=df[mk]; X=ctrl_fn(sub); X=np.column_stack([X, sub[shockcol]])
        b=np.linalg.lstsq(X, sub[f'{outcol}+{k}'].values, rcond=None)[0]; TG.append(b[-1])
        mk2 = df.h+k<=24
        sub2=df[mk2]; X2=ctrl_fn(sub2); X2=np.column_stack([X2, sub2[shockcol]])
        b2=np.linalg.lstsq(X2, sub2[f'{dosecol}+{k}'].fillna(0).values, rcond=None)[0]; TI.append(b2[-1])
    return np.array(TG), np.array(TI)
def deconv(TG, TI):
    K=len(TG); h=np.zeros(K); TIf=np.r_[1.0, TI]
    for k in range(1,K+1):
        h[k-1] = TG[k-1] - sum(h[k-j-1]*TIf[j] for j in range(1,k))
    return h
# insulin
S['G+0']=S.g_now
ctrl_ins = lambda sub: np.column_stack([np.ones(len(sub)), sub.g_now]+[(sub.ins_base==b).astype(float) for b in [2,3,4,6]])
for k in range(1,13): S[f'I+{k}']=S[f'I+{k}'] if f'I+{k}' in S else np.nan
TGi, TIi = lp_series(S, 'G', 'shock', ctrl_ins, 'I')
hi = deconv(TGi, TIi)
# NE: innovation-based; restrict to art-line rows with MAP observed
T2 = T[T.map_arterial.notna()].copy()
C_all = cells.loc[T2.index].values
ctrl_ne = lambda sub: np.column_stack([cells.loc[sub.index].values, sub.map_arterial, sub.ne_prev])
TGn, TIn = lp_series(T2, 'MA', 'norepinephrine', ctrl_ne, 'NE')
hn = deconv(TGn, TIn)
print(pd.DataFrame({'TG_ins':TGi,'TI_ins':TIi,'h_ins':hi,'cum_h_ins':np.cumsum(hi),'TG_ne':TGn,'TI_ne':TIn,'h_ne':hn,'cum_h_ne':np.cumsum(hn)}, index=range(1,13)).round(3).to_string())
both = P[P.map_arterial.notna() & P.map_cuff.notna()]
dd = both.map_cuff - both.map_arterial
print("cuff - art: mean", dd.mean(), "sd", dd.std(), "n", len(dd))
print(both.groupby(pd.cut(both.map_arterial,[0,40,50,60,70,80,90,200])).apply(lambda s: pd.Series({'d':(s.map_cuff-s.map_arterial).mean(),'sd':(s.map_cuff-s.map_arterial).std(),'n':len(s)})).round(2).to_string())
```
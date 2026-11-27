Norepinephrine has a clear pressor effect in these records. A 1 mcg/kg/min increase for one hour raises map_arterial by +3.45 at 1 h, +2.79 at 2 h, +1.79 at 3 h, +1.09 at 4 h and +0.50 at 5 h, then about 0. The cumulative effect is about +10 mmHg per (mcg/kg/min·h), so a sustained +Δr settles at about +10·Δr mmHg after roughly 5 h. The original model's answers (−1.33 for ×2, −0.48 for ×1/2) have the wrong sign and size.

Next I will save these findings to the notes. I will then redo the insulin estimate with the same rigour: cross-fitted flexible policy model, richer controls and 12 lags. That firms up the long tail, which matters for hours up to 12 h after the edit.

```python
with open('/work/notes.md','a') as f:
    f.write("""- NOREPI POLICY: titration AR-like: r_t ~ 0.27+0.74 r_{t-1}+0.04 r_{t-2}-0.003 MAP; start if MAP<65; stop sometimes MAP>=75; cap 1.0.
  Simple cell-mean policy failed placebo (x vs m_{t-1}: -1.73). Cross-fitted HGB policy on (m0,m1,m2,r1,r2,r3,mc0,t): R2 .93, innov sd .092;
  placebos m3,m4 pass (-0.09±.10, 0.07±.10).
- NOREPI EFFECT on map_arterial, structural IRF per 1 mcg/kg/min for 1h, lags1..12:
  3.45,2.79,1.79,1.09,0.50,0.22,0.26,-0.14,0.07,-0.11,0.20,-0.18; cum ~10 mmHg (reached by lag 5-6).
  Model original: ne_x2 -1.33, ne_half -0.48 -> wrong sign/size.
""")
rows=[]
for t in range(1,T):
    s_idx=np.where(~np.isnan(gf_last[:,t]))[0]; t_idx=np.full(len(s_idx),t)
    d=pd.DataFrame({'stay':s_idx,'t':t,'r':ins_r[s_idx,t]})
    d['gl0']=gf_last[s_idx,t]; d['gage']=gf_age[s_idx,t]
    for L in range(1,4): d['g%d'%L]=lagcol(gf_last,s_idx,t_idx,L)
    for L in range(1,4): d['r%d'%L]=lagcol(ins_r,s_idx,t_idx,L)
    d['lab0']=gl_last[s_idx,t]
    rows.append(d)
IP = pd.concat(rows, ignore_index=True)
featsI=['gl0','gage','g1','g2','r1','r2','r3','lab0','t']
IP['rhat']=np.nan; fold=IP.stay%2
for k in [0,1]:
    mdl=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
    mdl.fit(IP.loc[fold!=k,featsI], IP.loc[fold!=k,'r']); IP.loc[fold==k,'rhat']=mdl.predict(IP.loc[fold==k,featsI])
IP['x']=IP.r-IP.rhat
print('insulin policy R2', 1-IP.x.var()/IP.r.var(), 'innov sd', IP.x.std())
IP['g02']=(IP.gl0-160)**2/100
IP['g3p']=lagcol(gf_last, IP.stay.values, IP.t.values, 4)
b,s_,nn=ols(IP,'g3p',['x','gl0','g02','g1','g2','r1','r2']); print('placebo g(t-4) coef %.3f se %.3f'%(b.x,s_.x))
```
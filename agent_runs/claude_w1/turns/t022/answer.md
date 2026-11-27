The norepinephrine innovation is essentially random. Its correlation with the previous innovation is −0.014, and with glucose, insulin and hour about 0. There is a small −0.04 correlation with current MAP, so the adjustment depends somewhat on MAP within each band. My regressions control for MAP linearly, which covers that.

The run then failed because the insulin table S was built before I added the arterial-MAP lead columns. I'll rebuild S from the panel and rerun the cross-effect and saturation checks.

```python
S = P[(P.ins_on==1)&P.g_now.notna()&(P.h<=23)].copy()
S['shock'] = S.insulin - S.ins_base.map(mb)
res=[]
for k in [1,2,3,4,6]:
    mk = T[f'G+{k}'].notna() & T.g_now.notna()
    sub=T[mk]; C=cells[mk].values
    X=np.column_stack([C, sub.g_now, sub.ne_prev, sub.norepinephrine])
    r,_=ols(sub[f'G+{k}'],X)
    sub2 = S[S[f'MA+{k}'].notna() & S.map_arterial.notna()]
    X2=np.column_stack([np.ones(len(sub2)), sub2.g_now, sub2.map_arterial, sub2.shock]+[(sub2.ins_base==b).astype(float) for b in [2,3,4,6]])
    r2,_=ols(sub2[f'MA+{k}'],X2)
    res.append((k, r.b.iloc[-1], r.se.iloc[-1], r2.b.iloc[3], r2.se.iloc[3]))
print(pd.DataFrame(res,columns=['k','NE->G','se','INS->MAP','se2']).to_string())
for lo,hi in [(0,0.15),(0.15,0.35),(0.35,0.6),(0.6,0.9)]:
    mk = T['MA+1'].notna() & T.map_arterial.notna() & (T.ne_prev>lo)&(T.ne_prev<=hi)&(T.norepinephrine<1.0)
    sub=T[mk]; C=cells[mk].values
    X=np.column_stack([C, sub.map_arterial, sub.ne_prev, sub.norepinephrine])
    r,_=ols(sub['MA+1'],X)
    print(lo,hi,mk.sum(), round(r.b.iloc[-1],3), round(r.se.iloc[-1],3))
```
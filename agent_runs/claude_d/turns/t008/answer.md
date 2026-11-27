The previous run had already added columns to `P` before it failed, so the rename did not fit. I am rebuilding `P` from the clean panel `PT` and running the same policy tables. These show how often insulin is given and at what rate by current glucose, and the same for norepinephrine by current arterial MAP.

```python
def prep(PT):
    P = PT.copy(); P.columns = ['gf','gl','ma','mc','ins','nor']
    g = P.groupby(level='stay_id')
    P['gf_last'] = g.gf.ffill(); P['gl_last']=g.gl.ffill(); P['ma_last']=g.ma.ffill(); P['mc_last']=g.mc.ffill()
    P['ins0'] = P.ins.fillna(0); P['nor0']=P.nor.fillna(0)
    g = P.groupby(level='stay_id')
    P['ins_prev'] = g.ins0.shift(1); P['nor_prev']=g.nor0.shift(1)
    return P
P = prep(PT); PUu = prep(PU)
cur = P[P.gf.notna()]
bins = np.arange(60,420,20)
tab = cur.groupby(pd.cut(cur.gf,bins), observed=True).agg(n=('ins0','size'), p_ins=('ins', lambda x: x.notna().mean()), mean_rate=('ins','mean'), sd_rate=('ins','std'))
print(tab.to_string())
cur2 = P[P.ma.notna()]
tab2 = cur2.groupby(pd.cut(cur2.ma,np.arange(20,140,5)), observed=True).agg(n=('nor0','size'), p_ne=('nor', lambda x: x.notna().mean()), mean_rate=('nor','mean'), sd_rate=('nor','std'))
print(tab2.to_string())
```
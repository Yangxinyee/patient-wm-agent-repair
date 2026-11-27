Within each tier, an extra 1 u/h of insulin lowers the next-hour finger glucose by about 2 mg/dL: −1.92 (SE 0.11), −1.97 (0.07) and −2.06 (0.10) in the 150–180, 180–220 and 220–300 tiers. This is a real lowering effect, but small, and it is the same at different glucose levels. That points to a linear effect in dose that does not depend on the glucose level.

What matters for the readout is whether the effect persists: does a dose at hour s still lower glucose at s+2, s+3 and later? Under x2 from hour 12, extra insulin accumulates over 12 hours. A persistent effect could add up to tens of mg/dL; a one-hour-only effect would stay around −4.

Next I'll estimate the response at lags 1–8. I'll regress the reading at t+k on the dose at t, with fixed effects for the reading at t, restricted to on-insulin rows. Later doses respond only weakly to a 2 mg/dL change, so feedback through the protocol should be small. I'll also check that readings at t+k are about equally available whatever the dose, because missing readings could bias this.

```python
GF = Gtr['gf']
res = []
for k in range(1,9):
    rows=[]
    for h in range(0,25-k+1):
        if h+k>24: continue
        rows.append(pd.DataFrame({'stay':stays_tr,'h':h,'g0':GF[h].values,'ins':Gtr['ins'][h].values,'gk':GF[h+k].values}))
    D = pd.concat(rows); D = D[D.ins.notna() & D.g0.notna()]
    D['tier'] = pd.cut(D.g0,[0,150,180,220,300,1000],right=False,labels=False)
    D = D[(D.g0%10)!=0]
    avail = D.gk.notna().mean()
    # availability vs dose within tier
    D['fe'] = D.g0
    D['avail']=D.gk.notna().astype(float)
    ba,_,_ = fe_reg(D,'avail','ins','fe')
    b,se,n = fe_reg(D,'gk','ins','fe')
    res.append((k, round(b[0],3), round(se[0],3), n, round(avail,3), round(ba[0],4)))
print(pd.DataFrame(res, columns=['lag','coef','se','n','avail','avail_coef']).to_string())
```
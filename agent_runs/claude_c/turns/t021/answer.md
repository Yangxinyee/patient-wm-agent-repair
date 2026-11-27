Norepinephrine raises MAP. An innovation of +1 mcg/kg/min at t is followed by MAP higher by 3.4 at t+1, 5.2 at t+2 and 5.9 mmHg at t+3. The placebo on earlier MAP is small but slightly positive (+0.19 and +0.29, SE 0.10). That suggests my controls do not fully capture the policy. A likely cause is that the multiplicative rule implies a band × previous-rate interaction, which I left out.

These numbers are reduced-form: they include the innovation carrying into later rates. Next I will:
1. Refit with band × source × previous-rate interactions and check that the placebo falls to about 0.
2. Estimate how an innovation at t propagates to the rates at t+k (stops count as 0).
3. Do the same for insulin.
4. Solve the triangular system RF_MAP(L) = Σ_k b(L−k)·RF_NE(k) to get the structural per-hour response b.

A clean placebo together with a smooth b would confirm the approach.

```python
def build_ctrl_ne(q):
    m = q.mr.values; ne_p = q.NEp.values; X=[]
    for s in [1,2]:
        for lo,hi in [(0,64.5),(64.5,65.5),(65.5,74.5),(74.5,75.5),(75.5,300)]:
            ind = ((q.src.values==s)&(m>=lo)&(m<hi)).astype(float)
            X += [ind, ind*m, ind*ne_p, ind*ne_p**2, ind*np.log(ne_p)]
    return X
def ne_rf(L, outcome):
    # outcome: column name; L lag; returns coefficient on NE_t (innovation) with policy controls
    Pg = P.groupby(level=0)
    if outcome=='NEfill':
        Y = Pg.NE.shift(-L).fillna(0.0) if L>0 else P.NE
        # stays end at h=24 -> mark missing beyond
        hh = P.index.get_level_values('h').values
        Y = Y.where(hh+L<=24)
    else:
        Y = Pg[outcome].shift(-L)
    q = P.assign(Y=Y)
    q = q[(q.index.get_level_values('h')>=1) & q.NEp.notna() & q.NE.notna() & q.Y.notna() & q.mr.notna() & (q.NEp<1.0)]
    X = build_ctrl_ne(q) + [q.NE.values]
    nm = [f'x{i}' for i in range(len(X)-1)]+['NE']
    r = ols(q.Y.values, X, nm).loc['NE']
    return len(q), r.b, r.se
rows=[]
for L in [-3,-2,-1]+list(range(1,13)):
    n,b,se = ne_rf(L,'map_arterial')
    rows.append((L,n,b,se) + ((ne_rf(L,'NEfill')[1:]) if L>=1 else (np.nan,np.nan)))
ne_tab = pd.DataFrame(rows, columns=['L','n','map_b','map_se','ne_b','ne_se'])
print(ne_tab.round(3))
```
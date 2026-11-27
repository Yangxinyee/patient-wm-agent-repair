The CSV gives, for each stay-hour, the model's shift, its standard error, the factual predicted mean and whether the hour was "reached". The shifts build up slowly over hours: for stay 2020000000 they go from −0.3 at hour 16 to −3.3 at hour 20.

Next I recover the model's implied impulse response. I regress each stay-hour shift on the recorded dose at each lag (the dose change under ×2 equals the recorded dose, from hour 12 on), for insulin ×2 and NE ×2. Then I compare the coefficients with the true open-loop impulse responses I estimated from the records. Insulin: −2.06, −2.45, −2.18, −1.77, … NE: +3.30, +2.65, +1.86, …

A lag-1 coefficient near the truth with later coefficients collapsing would mean missing memory. A small lag-1 coefficient would mean the immediate response is too weak.

```python
def dose_grid(df, drug):
    d=df[df.event==drug].copy(); d['h']=np.floor(d.hour).astype(int)
    g=d.pivot_table(index='stay_id',columns='h',values='rate',aggfunc='sum').reindex(columns=range(-3,25)).fillna(0.0)
    return g
gi_tu=dose_grid(tu,'insulin'); gn_tu=dose_grid(tu,'norepinephrine')
def irf_fit(edit, grid, L=10, mult=1.0):
    d=de1[de1.edit==edit]
    rows=[];y=[];w=[]
    for r in d.itertuples():
        t=int(r.hour); x=np.zeros(L)
        for k in range(1,L+1):
            hh=t-k
            if hh>=12 and r.stay_id in grid.index: x[k-1]=grid.loc[r.stay_id,hh]*mult
        rows.append(x); y.append(r.shift)
    Xr=np.array(rows); y=np.array(y)
    m=Xr.sum(1)>0
    b,*_=np.linalg.lstsq(Xr[m],y[m],rcond=None)
    return b, m.sum()
for ed,gr,mu in [('insulin_x2',gi_tu,1.0),('insulin_half',gi_tu,-0.5),('ne_x2',gn_tu,1.0),('ne_half',gn_tu,-0.5)]:
    b,n=irf_fit(ed,gr,L=10,mult=mu)
    print(ed,n,np.round(b,2),'sum',round(b.sum(),2))
```
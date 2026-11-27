`true_shift` fails on an argument it apparently expects in another form, and its source is lost. The truth calculation is simple, so I will write it out directly.

For each reached row at hour h:

truth = Σ_k IR[k] × (m − 1) × recorded dose at hour h − k, counting only dose hours ≥ 12.

Here m is 2 or ½. For insulin I use the deconvolved impulse responses d_f, d_l and d_avg. For norepinephrine I use dN_use, which is linear in the rate. The grid's NE range will show whether it stores rates or doses.

I will also show truth, edited model and original by hours since the first dose. That will tell whether the insulin shortfall is in the early build-up or in the long run, and what the norepinephrine ×2 problem looks like over time.

```python
print('Gtu keys',list(Gtu.keys()),'ne max %.3f ins max %.2f'%(np.nanmax(Gtu['ne'].values),np.nanmax(Gtu['ins'].values)), 'index sample',list(Gtu['ins'].index[:2]),'cols',list(Gtu['ins'].columns[:4]))
def truth_rows(D,edit,ir,key,mult):
    r=D[(D.edit==edit)&D.reached.astype(bool)].copy(); vals=[]; first=[]
    G_=Gtu[key]
    for s,h in zip(r.stay_id.values,r.hour.values):
        row=G_.loc[s]; tot=0.0
        for k in range(1,len(ir)+1):
            t=h-k
            if t>=12 and t in row.index:
                d=row.loc[t]
                if np.isfinite(d): tot+=ir[k-1]*(mult-1)*d
        vals.append(tot)
        hrs=[t for t in row.index if t>=12 and np.isfinite(row.loc[t]) and row.loc[t]>0]; first.append(min(hrs) if hrs else np.nan)
    r['truth']=vals; r['since']=r.hour.values-np.array(first); return r
T10={}
for ed,key,mult,vecs in [('insulin_x2','ins',2.0,['d_f','d_l','d_avg']),('insulin_half','ins',0.5,['d_f','d_l','d_avg']),('ne_x2','ne',2.0,['dN_use']),('ne_half','ne',0.5,['dN_use'])]:
    for vn in vecs:
        r=truth_rows(de,ed,np.asarray(globals()[vn],dtype=float),key,mult); T10[(ed,vn)]=r
        print(ed,vn,'truth %.3f'%r.truth.mean(),'n',len(r))
r=T10[('insulin_x2','d_avg')].merge(do[['stay_id','hour','edit','shift']].rename(columns={'shift':'orig'}),on=['stay_id','hour','edit'])
print(r.groupby('since').agg(n=('truth','size'),truth=('truth','mean'),edited=('shift','mean'),orig=('orig','mean')).round(2).T.to_string())
r=T10[('ne_x2','dN_use')].merge(do[['stay_id','hour','edit','shift']].rename(columns={'shift':'orig'}),on=['stay_id','hour','edit'])
print(r.groupby('since').agg(n=('truth','size'),truth=('truth','mean'),edited=('shift','mean'),orig=('orig','mean')).round(2).T.to_string())
```
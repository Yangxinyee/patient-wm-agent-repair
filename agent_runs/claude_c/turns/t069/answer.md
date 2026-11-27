Run 61 shows that z26 is the only z dimension that varies at any drug bundle: insulin-only, norepinephrine-only and both. So whatever the posterior says about a dose passes through that one number. z26 also varies at reading-only bundles (variance 2.48), together with many other dimensions. Scaling `z_in`'s response to z26 would therefore also act on reading bundles, but the size of that side effect is still unknown.

Next I will measure how z26 encodes dose at each kind of drug bundle, using the records of the same 100 tuning stays:
- regress z26 on the insulin rate, the norepinephrine rate, the previous rates (whose END atoms share the bundle) and the last glucose and MAP readings;
- check whether z26 at reading bundles is linearly predictable from the other z dimensions.

A tight, mostly linear dependence on dose at drug bundles would make z26 a clean carrier to amplify. A high R² at reading bundles would make an amplification direction partly selective for drug bundles.

```python
tu_=pd.read_csv('/data/tuning.csv')
sid_=So_['stay_id']; hr_=So_['hour']; na_=So_['n_atoms']
def lookup_rate(df, ev):
    d_=df[df.event==ev][['stay_id','hour','rate']].copy()
    return {(int(a),round(float(b),3)):float(c) for a,b,c in d_.values}
ins_map=lookup_rate(tu_,'insulin'); ne_map=lookup_rate(tu_,'norepinephrine')
def last_val(df, ev):
    d_=df[df.event==ev][['stay_id','hour','value']].sort_values(['stay_id','hour'])
    return {s:(g.hour.values,g.value.values) for s,g in d_.groupby('stay_id')}
gfin=last_val(tu_,'glucose_finger'); mart=last_val(tu_,'map_arterial')
def lastv(dct,s,h):
    if s not in dct: return np.nan
    hh,vv=dct[s]; k=np.searchsorted(hh,h+1e-6)-1
    return vv[k] if k>=0 else np.nan
rows_=[]
for i_ in range(len(sid_)):
    s=int(sid_[i_]); h=round(float(hr_[i_]),3)
    ri=ins_map.get((s,h),0.0); rn=ne_map.get((s,h),0.0)
    rip=ins_map.get((s,round(h-1,3)),0.0); rnp=ne_map.get((s,round(h-1,3)),0.0)
    rows_.append((ri,rn,rip,rnp,lastv(gfin,s,h),lastv(mart,s,h)))
R_=np.array(rows_)
z26=zo_[:,26]
for k_,msk in [('ins_only',grp_['ins_only']),('ne_only',grp_['ne_only']),('both',grp_['both'])]:
    m=msk&np.isfinite(R_[:,4])&np.isfinite(R_[:,5])
    X_=np.column_stack([np.ones(m.sum()),R_[m,0],R_[m,1],R_[m,2],R_[m,3],R_[m,4]/50,R_[m,5]/10,(R_[m,2]>0),(R_[m,3]>0),na_[m]])
    b_,res__,_,_=np.linalg.lstsq(X_,z26[m],rcond=None)
    pred=X_@b_; r2=1-((z26[m]-pred)**2).mean()/z26[m].var()
    print(k_, m.sum(), 'R2 %.3f'%r2, 'coef [1,ins,ne,ins_prev,ne_prev,g/50,map/10,has_ip,has_np,n_atoms]:', np.round(b_,3))
    print('   z26 mean %.2f sd %.2f; ins rate mean %.2f; ne mean %.3f'%(z26[m].mean(),z26[m].std(),R_[m,0].mean(),R_[m,1].mean()))
mr=grp_['read']
others=[j for j in range(32) if j!=26]
Xr=np.column_stack([np.ones(mr.sum()),zo_[mr][:,others]])
br,_,_,_=np.linalg.lstsq(Xr,z26[mr],rcond=None)
print('read: R2 z26 ~ other z dims: %.3f'%(1-((z26[mr]-Xr@br)**2).mean()/z26[mr].var()))
print('read z26 mean %.2f; drug z26 mean %.2f'%(z26[mr].mean(), z26[hi_|hn_].mean()))
```
`code_id` is keyed by code name, not number, so `atom_vec` needs `'norepinephrine'` rather than 9. Most of the "T" hits are just `.T` transposes, so restoring `T` as a callable dict is safe. I am rerunning the u(r) diagnostic with the correct code name. The question stays: does the action encoding leave the in-range dose trend for doubled NE rates above 1.0, and does bin-capping repair it?

```python
aS=lambda r: np.asarray(atom_vec('norepinephrine',0,2,4,3,None,4.8*r,r),dtype=np.float64)
aE=np.asarray(atom_vec('norepinephrine',2,2,4,3),dtype=np.float64)
print('check vs uenc', np.abs(np.asarray(uenc([aS(.5),aE]),dtype=np.float64)-uenc2([aS(.5),aE])).max())
Br,Bd=Wt('embed.bins.rate'),Wt('embed.bins.dose')
ed=[d for d in VOCAB['numeric'] if d['code']=='norepinephrine']; print([(d['attribute'],d['median'],d['iqr'],d['fine_bin_edges'][-5:]) for d in ed])
er=[d for d in ed if d['attribute']=='rate'][0]['fine_bin_edges']; edd=[d for d in ed if d['attribute']=='dose'][0]['fine_bin_edges']
bt_r=1+np.searchsorted(er,1.0,'left'); bt_d=1+np.searchsorted(edd,4.8,'left'); print('top bins',bt_r,bt_d)
rg=np.linspace(.05,2.0,40); U=np.array([uenc2([aS(r),aE]) for r in rg])
dB=Br[16]+Bd[16]-Br[bt_r]-Bd[bt_d]
Ucap=np.array([uenc2([aS(r)-dB*(r>1+1e-9),aE]) for r in rg])
m=(rg>=.1)&(rg<=1.0); X=np.c_[np.ones(m.sum()),rg[m]]; cf=np.linalg.lstsq(X,U[m],rcond=None)[0]; g=cf[1]
for nm,UU in [('orig',U),('cap',Ucap)]:
    pr=(UU-cf[0])@g/(g@g); res=np.linalg.norm(UU-cf[0]-np.outer(pr,g),axis=1)
    print(nm,' '.join(f'{r:.2f}:{p:.2f}/{q:.1f}' for r,p,q in zip(rg[::3],pr[::3],res[::3])))
print('|g|',np.linalg.norm(g).round(2))
```
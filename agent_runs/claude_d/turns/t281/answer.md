`code_id` covers only measurement codes, so `atom_vec` was a measurement-atom builder and cannot make NE atoms. I will build NE atoms directly from the embedding tables, using the verified formula and z = (x − median)/IQR clipped to ±10. The END atom gets kind 2, absent bins and zero numerics.

Then I run the u(r) diagnostic over rate 0.05–2.0. If the original encoding stops tracking the in-range dose direction above rate 1 and the bin-capped variant keeps tracking it, the mlp1 correction should map bin-16 NE atoms to the capped-bin encoding (or an extrapolation of it).

```python
C_,K_,EK_=Wt('embed.code.weight'),Wt('embed.kind.weight'),Wt('embed.ekind.weight')
Bv,Br,Bd=Wt('embed.bins.value.weight'),Wt('embed.bins.rate.weight'),Wt('embed.bins.dose.weight')
Wn,bn=Wt('embed.numeric.weight'),Wt('embed.numeric.bias'); UN,RU=Wt('embed.categorical.unit.weight'),Wt('embed.categorical.rate_unit.weight')
ed={d['attribute']:d for d in VOCAB['numeric'] if d['code']=='norepinephrine'}; print({k:(v['median'],v['iqr'],v['fine_bin_edges'][-5:]) for k,v in ed.items()})
def zz(x,d): return float(np.clip((x-d['median'])/d['iqr'],-10,10))
def bb(x,d): return int(1+np.searchsorted(d['fine_bin_edges'],x,'left'))
def aNE(r,cap=False):
    br,bd=bb(r,ed['rate']),bb(4.8*r,ed['dose'])
    if cap and br==16: br,bd=bb(1.0,ed['rate']),bb(4.8,ed['dose'])
    return C_[9]+K_[0]+EK_[2]+Br[br]+Bd[bd]+Bv[0]+Wn@np.array([0,0,zz(4.8*r,ed['dose']),1,zz(r,ed['rate']),1])+bn+UN[4]+RU[3]
aE=C_[9]+K_[2]+EK_[2]+Br[0]+Bd[0]+Bv[0]+bn+UN[4]+RU[3]
rg=np.linspace(.05,2.0,40)
U=np.array([uenc2([aNE(r),aE]) for r in rg]); Ucap=np.array([uenc2([aNE(r,True),aE]) for r in rg])
m=(rg>=.1)&(rg<=1.0); cf=np.linalg.lstsq(np.c_[np.ones(m.sum()),rg[m]],U[m],rcond=None)[0]; g=cf[1]
for nm,UU in [('orig',U),('cap',Ucap)]:
    pr=(UU-cf[0])@g/(g@g); res=np.linalg.norm(UU-cf[0]-np.outer(pr,g),axis=1)
    print(nm,' '.join(f'{r:.2f}:{p:.2f}/{q:.1f}' for r,p,q in zip(rg[::3],pr[::3],res[::3])))
print('|g|',np.linalg.norm(g).round(2), 'bins at 1.0,1.01:', bb(1.0,ed['rate']), bb(1.01,ed['rate']))
```
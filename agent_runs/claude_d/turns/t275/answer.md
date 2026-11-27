The END atom of an NE record carries no rate or dose; only the START atom (akind 0) does. So the dose edit changes only START atoms. A typical NE bundle is {new START, previous END}, both going through `encode_action`.

Next I will compute the action encoding u(r) for that typical bundle over r ∈ (0, 2]. I fit the in-range trend (r ≤ 1) and check whether r > 1 continues it, comparing:
- the original encoding,
- a "bin-capped" variant: bin 16 replaced by the bin of rate 1.0, numeric z still extended.

If the original's projection collapses or turns away for r > 1 while the capped variant keeps rising, that confirms the mechanism and gives a correction target for gated mlp1 units. I also check my own encoder function against the verified `uenc` helper.

```python
from scipy.special import erf
def Wt(n): return load(n).numpy().astype(np.float64)
def gel(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def mlpP(x,p,W0=None,b0=None,W2=None,b2=None):
    W0=Wt(p+'.0.weight') if W0 is None else W0; b0=Wt(p+'.0.bias') if b0 is None else b0
    W2=Wt(p+'.2.weight') if W2 is None else W2; b2=Wt(p+'.2.bias') if b2 is None else b2
    return gel(x@W0.T+b0)@W2.T+b2
def lnP(x,p): mu=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*Wt(p+'.weight')+Wt(p+'.bias')
def uenc2(atoms,**kw):
    n=len(atoms); s=mlpP(np.asarray(atoms),'encode_action.mlp1',**kw).sum(0)/np.sqrt(n)+Wt('encode_action.size.weight')[:,0]*np.log(1+n)+Wt('encode_action.size.bias')
    return lnP(mlpP(s,'encode_action.mlp2'),'encode_action.norm')
aS=lambda r: np.asarray(atom_vec(9,0,2,4,3,None,4.8*r,r),dtype=np.float64)
aE=np.asarray(atom_vec(9,2,2,4,3),dtype=np.float64)
try: print('check vs uenc', np.abs(np.asarray(uenc([aS(.5),aE]))-uenc2([aS(.5),aE])).max())
except Exception as ex: print('uenc ERR',repr(ex)[:150])
Br,Bd=Wt('embed.bins.rate'),Wt('embed.bins.dose')
ed=[d for d in VOCAB['numeric'] if d['code']=='norepinephrine']; print([(d['attribute'],d['median'],d['iqr'],d['fine_bin_edges'][-5:]) for d in ed])
er=[d for d in ed if d['attribute']=='rate'][0]['fine_bin_edges']; edd=[d for d in ed if d['attribute']=='dose'][0]['fine_bin_edges']
bt_r=1+np.searchsorted(er,1.0,'left'); bt_d=1+np.searchsorted(edd,4.8,'left'); print('top bins',bt_r,bt_d)
rg=np.linspace(.05,2.0,40); U=np.array([uenc2([aS(r),aE]) for r in rg])
Ucap=np.array([uenc2([aS(r)-(Br[16]+Bd[16]-Br[bt_r]-Bd[bt_d])*(r>1+1e-9),aE]) for r in rg])
m=(rg>=.1)&(rg<=1.0); X=np.c_[np.ones(m.sum()),rg[m]]; cf=np.linalg.lstsq(X,U[m],rcond=None)[0]; g=cf[1]
for nm,UU in [('orig',U),('cap',Ucap)]:
    pr=(UU-cf[0])@g/(g@g); res=np.linalg.norm(UU-cf[0]-np.outer(pr,g),axis=1)
    print(nm,' '.join(f'{r:.2f}:{p:.2f}/{q:.1f}' for r,p,q in zip(rg[::3],pr[::3],res[::3])))
print('|g|',np.linalg.norm(g).round(2))
```
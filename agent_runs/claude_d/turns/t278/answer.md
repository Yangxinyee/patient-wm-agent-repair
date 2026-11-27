`atom_vec` reads `T['embed.code.weight'][code_id[code]]`, so `T` used to be a dict of tensors and was later overwritten by a function `T(n)`. In run 248 I also redefined `gel`, which other helpers may call with torch tensors.

Repairs in this run:
- Make `T` a dict subclass that is also callable, so both old and new users work.
- Make `gel` dispatch exact GELU for both torch and numpy inputs.
- Scan which helpers reference these names.

Then I rerun the u(r) diagnostic for NE START atoms over rate 0.05–2.0, original versus bin-capped.

```python
Tf=T
class TD(dict):
    def __call__(self,n): return Tf(n)
T=TD({n:load(n) for n in NAMES}); print(type(T['embed.code.weight']), list(code_id.items())[:4])
def gel(x):
    if isinstance(x,torch.Tensor): return torch.nn.functional.gelu(x)
    return 0.5*x*(1+erf(x/np.sqrt(2)))
users={k:[n for n in ('gel','Wt','T','mlpP') if n in v.__code__.co_names] for k,v in list(globals().items()) if callable(v) and getattr(v,'__module__',None)=='__main__' and hasattr(v,'__code__')}
print({k:v for k,v in users.items() if v})
aS=lambda r: np.asarray(atom_vec(9,0,2,4,3,None,4.8*r,r),dtype=np.float64)
aE=np.asarray(atom_vec(9,2,2,4,3),dtype=np.float64)
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
Two results from run 292:

- **The head-1 value edit is purely additive** (error 1e-8), and the original model's weights along those value directions are tiny (0.24% of the new ones). So cutting out the original response along them should do little harm.
- **The kernel still holds the replica's tokens** for all 5,118 bundles after each stay's first: `Td` (drug bundles, 7 tokens) and `Tm` (other bundles, 6 tokens). The prefix `P` could not be restored, so I will write my own layer norm from `norm1` and check it against the stored values.

Plan for this run:

1. Check my layer norm and my head-1 attention against the stored drug-bundle attention, and identify which token types `Tm` holds.
2. Measure the current readout B on attention-weighted token mixtures at non-drug bundles, before and after the first drug.
3. Refit B by ridge regression. Targets are the drug features at drug bundles and zero at all other bundles, including synthetic first bundles.
4. Rebuild with `build7` at about 0.6 of full insulin gain and 0.5 of full norepinephrine gain, but only if the attention check passes.

The leak hypothesis is confirmed if the current B gives large mixed readouts in pre-drug bundles and small ones afterwards. It is refuted if the readouts are small before the first drug as well; then the leak must come from elsewhere, such as the first bundle or feedback through the state.

```python
import numpy as np, torch
S=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
f64=lambda t:(t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)).astype(np.float64)
g1=f64(load_original('core.prelude.0.norm1.weight')); b1=f64(load_original('core.prelude.0.norm1.bias'))
def LNe(x,eps):
    x=np.asarray(x,np.float64); m=x.mean(-1,keepdims=True); vv=x.var(-1,keepdims=True); return (x-m)/np.sqrt(vv+eps)*g1+b1
errs={eps:np.abs(LNe(u,eps)-LN1u).max() for eps in (1e-5,1e-6,1e-12)}; EPS=min(errs,key=errs.get)
LNa=lambda x: LNe(x,EPS)
kvO=f64(load_original('core.prelude.0.attn.kv.weight')); qO=f64(load_original('core.prelude.0.attn.q.weight'))
WoO=f64(load_original('core.prelude.0.attn.out.weight')); WbO=f64(load_original('core.b_in.weight'))
say('LN err',{k:float('%.2g'%v) for k,v in errs.items()},'| glob==orig Wkv %.1g Wo %.1g Wbin %.1g'%(np.abs(Wkv-kvO).max(),np.abs(Wo-WoO).max(),np.abs(Wbin-WbO).max()))
def attn_h(T,h,sc):
    L=LNa(T); Q=L[:,:4]@qO[h*48:(h+1)*48].T; K=L@kvO[h*48:(h+1)*48].T
    lg=np.einsum('nid,njd->nij',Q,K)*sc; lg-=lg.max(-1,keepdims=True); w=np.exp(lg); return w/w.sum(-1,keepdims=True),L
scs={'1/sqrt48':48**-.5,'1/48':1/48,'1':1.0}
ae={k:np.abs(attn_h(Td,1,s)[0]-att[:,1]).max() for k,s in scs.items()}; SC=scs[min(ae,key=ae.get)]
say('attn err head1', {k:float('%.2g'%v) for k,v in ae.items()})
st=np.load('/work/fwd_15/states_original.npz'); X15=np.asarray(st['x'],np.float64); sd=np.asarray(st['stay_id'])
dr=np.asarray(st['has_insulin']).astype(bool)|np.asarray(st['has_norepinephrine']).astype(bool)
new=np.r_[True,sd[1:]!=sd[:-1]]; pre=np.zeros(len(sd),bool); started=False
for i in range(len(sd)):
    if new[i]: started=False
    if dr[i]: started=True
    pre[i]=not started
hp=np.zeros((len(sd),4,192)); hp[1:]=X15[:-1,192:960].reshape(-1,4,192); hp[new]=0
zp=np.zeros((len(sd),32)); zp[1:]=X15[:-1,960:992]; zp[new]=0
slot_tok=hp+slots[None]; ztok=zp@zinW.T+zinB
drw=np.where(dr)[0]; mrw=np.where(~dr&~new)[0]
say('counts',len(drw),len(mrw),'Td/Tm sizes',len(Td),len(Tm))
if len(mrw)==len(Tm): say('Tm slot err %.2g'%np.abs(Tm[:,:4]-slot_tok[mrw]).max())
if len(drw)==len(Td): say('Td slot err %.2g'%np.abs(Td[:,:4]-slot_tok[drw]).max())
for j in range(4,Tm.shape[1]):
    say(f'Tm tok{j}: d_uconst {np.abs(Tm[:,j]-u_const).max():.2g} d_ztok {np.abs(Tm[:,j]-ztok[mrw]).max() if len(mrw)==len(Tm) else -1:.2g} spread {Tm[:,j].std(0).mean():.2g}')
for j in range(4,Td.shape[1]):
    say(f'Td tok{j}: d_u {np.abs(Td[:,j]-u).max():.2g} d_ztok {np.abs(Td[:,j]-ztok[drw]).max() if len(drw)==len(Td) else -1:.2g} spread {Td[:,j].std(0).mean():.2g}')
```
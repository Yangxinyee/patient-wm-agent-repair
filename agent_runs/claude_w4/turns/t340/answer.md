The replicas check out exactly: my layer norm matches the stored values, and my head-1 attention matches the stored drug-bundle attention to 3e-16. The globals `Wkv`, `Wo` and `Wbin` are still the original weights.

The stored tokens are not simply h + core.slots (that check fails by about 200), so the slot tokens must be a decayed copy of h. I will use the stored token arrays directly:
- `Td` holds [4 slots, event token, drug token, latent token].
- `Tm` holds [4 slots, u_const, one more token]. One token type is missing from `Tm`, and I need to find out which.

This run will:
1. Check that the `Tm` and `Td` rows line up with the fwd_15 bundles, using cosines to h.
2. Identify the 6th `Tm` token.
3. Measure the current head-1 leak readout at non-drug bundles before and after the first drug, and at a synthetic first bundle.
4. Refit B by weighted ridge regression: drug features as targets at drug bundles, zero at all other bundles and the first bundle.
5. Rebuild with `build7` at insulin gain 2.91 and norepinephrine gain 3.45, about 0.6 and 0.5 of the full v7 gains.

The leak hypothesis holds if the current readout is large before the first drug and the refit brings it near zero while the drug-bundle fit stays good.

```python
import numpy as np, torch
S=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
def rowcos(Aa,Bb):
    a=Aa.reshape(len(Aa),-1); b=Bb.reshape(len(Bb),-1)
    return float(np.median((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)))
say('map cos Tm %.3f Td %.3f | F0~rI corr %.3f'%(rowcos(Tm[:,:4]-slots[None],hp[mrw]),rowcos(Td[:,:4]-slots[None],hp[drw]),np.corrcoef(F[:,0],rI)[0,1]))
dz=np.linalg.norm(Tm[:,5].mean(0)-Td[:,6].mean(0)); de_=np.linalg.norm(Tm[:,5].mean(0)-Td[:,4].mean(0))
zlike = dz < de_
say('Tm5 mean dist to Td-z %.2f, to Td-e %.2f -> Tm5 is %s'%(dz,de_,'z' if zlike else 'e'))
if zlike: Tm7=Tm                      # e missing (no proxy available)
else:     Tm7=np.concatenate([Tm, ztok[mrw][:,None,:]],1)
Mi=np.linalg.inv(M); Bo=(Mi@(A.T@kvO[240:288])).T
def mixes(T):
    w,L=attn_h(T,1,SC); return np.einsum('nij,njd->nid',w,L), w
Xd,wd=mixes(Td); Xm,wm=mixes(Tm7)
rng=np.random.default_rng(0); nf=300
fb=[np.repeat(slots[None],nf,0), np.repeat(u_const[None,None],nf,0)]
fb.append((Tm[rng.integers(0,len(Tm),nf),5] if not zlike else np.repeat(zinB[None],nf,0))[:,None])
if not zlike: fb.append(np.repeat(zinB[None,None],nf,0))
Tf=np.concatenate(fb,1); Xf,wf=mixes(Tf)
prem=pre[mrw]
def stats(Bt,tag):
    fd=Xd@Bt; fm=Xm@Bt; ff=Xf@Bt
    r=lambda x: np.round(np.sqrt((x**2).mean((0,1))),3)
    say(f'{tag}: drugErr {r(fd-F[:,None])} measPre {r(fm[prem])} measPost {r(fm[~prem])} first {r(ff)}')
stats(B+Bo,'curB')
say('h1 att meas pre: slots %.3f uconst %.3f | post: slots %.3f uconst %.3f | first: slots %.3f'%(wm[prem][:,:,:4].sum(-1).mean(),wm[prem][:,:,4].mean(),wm[~prem][:,:,:4].sum(-1).mean(),wm[~prem][:,:,4].mean(),wf[:,:,:4].sum(-1).mean()))
Xa=np.concatenate([Xd.reshape(-1,192),Xm.reshape(-1,192),Xf.reshape(-1,192)])
Ya=np.concatenate([np.repeat(F,4,0),np.zeros((Xm.shape[0]*4,3)),np.zeros((Xf.shape[0]*4,3))])-Xa@Bo
Wt=np.concatenate([np.ones(Xd.shape[0]*4),3*np.ones(Xm.shape[0]*4),20*np.ones(Xf.shape[0]*4)])
G_=Xa.T@(Xa*Wt[:,None]); R_=Xa.T@(Ya*Wt[:,None]); md=np.trace(G_)/192
cands={}
for lam in (1e-7,1e-6,1e-5,1e-4):
    Bn=np.linalg.solve(G_+lam*md*np.eye(192),R_); cands[lam]=Bn; stats(Bn+Bo,f'lam{lam:g}')
np.savez('/work/v10_cands.npz', **{f'B_{k:g}':v for k,v in cands.items()})
with open('/work/notes.md','a') as f:
    f.write('\n- run293: LN1 replica exact (eps1e-5), head-1 attention replica exact (1/sqrt48). Td=[4 slots,e,u,z]; Tm=[4 slots,u_const,tok5]; slot tokens != h+slots (decayed h). Globals Wkv/Wo/Wbin = original.\n- run294: refit B on attention-weighted mixtures (drug targets F; zero at meas bundles x3, synthetic first bundles x20); cands saved /work/v10_cands.npz\n'+'\n'.join('  '+s for s in S)+'\n')
```
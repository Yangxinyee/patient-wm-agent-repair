I now have the exact signatures, and they tell me the plan:
- The rollout replica builds the z token from `W`, so I can test a weight edit offline before using forward passes.
- Stage 1: rebuild `z_post`/`z_in` so the z token equals a copy of u plus `v_j·g_j·(dose_j − m_j(ctx))`.
  - u is the action embedding; drug j is code 8 or 9.
  - `m_j(ctx)` is the dose the model's own state expects.
  - The copy makes attention split evenly between u and the z token, so factual outputs should not change. The extra term carries only the unexpected part of the dose.

First check: can u be copied by a 30-component linear basis, and can dose and presence be decoded linearly from u? I will test this on a synthetic dose grid reaching 3× the observed maximum. Small reconstruction and decode errors (≲1e-3 relative, about 0.05 dose units) confirm the plan. Large errors mean I need nonlinear decoders.

```python
import torch, math
print({n:SHAPES[n] for n in NAMES if n.startswith(('z_post','z_in','core.prelude.0.attn'))})
ex={}
for s,B_ in BUN.items():
    for t,atoms in B_:
        for a in atoms:
            if a.get('act') and a['kind']==0 and a['code'] not in ex: ex[a['code']]=a
    if 8 in ex and 9 in ex: break
def rate_of(a):
    nm=DRUGN[a['code']]; return a['num'][4]*NUMV[(nm,'rate')]['iqr']+NUMV[(nm,'rate')]['median']
def u_syn(r8,r9):
    ua=[]
    if r8 is not None: ua.append(edit_atom(ex[8], r8/rate_of(ex[8])))
    if r9 is not None: ua.append(edit_atom(ex[9], r9/rate_of(ex[9])))
    return set_enc(W0,[fix_atom2(a) for a in ua],'encode_action')
G=[(r,None) for r in np.arange(0,20.01,0.25)]+[(None,r) for r in np.arange(0,3.001,0.02)]+[(a,b) for a in np.arange(0,20.01,1.0) for b in np.arange(0,3.001,0.1)]
Ug=torch.stack([u_syn(*x) for x in G]).numpy()
Lg=np.array([[x[0] or 0.0, x[1] or 0.0, x[0] is not None, x[1] is not None] for x in G],float)
TT=np.load('/work/tf_train.npz'); DAm=TT['DA_meta']; DAu=TT['DA_u']; DAc=TT['DA_ctx']
print('DA_meta rows',DAm[:2].round(3).tolist(),'|u| mean',np.linalg.norm(DAu,axis=1).mean().round(2))
ue=u_empty.numpy()[None]; Uf=np.vstack([Ug,DAu[::5],ue]); mu_u=Uf.mean(0)
_,sv,Vt=np.linalg.svd(Uf-mu_u,full_matrices=False)
for K in [20,26,30]:
    P=Vt[:K].T; re=lambda X: np.abs((X-mu_u)-((X-mu_u)@P)@P.T).max(1)
    print(K,'sv ratio %.2e'%(sv[K-1]/sv[0]),'maxerr grid %.2e data %.2e empty %.2e'%(re(Ug).max(),re(DAu).max(),re(ue).max()))
Xg=np.hstack([Ug,np.ones((len(Ug),1))]); Wd=np.linalg.lstsq(Xg,Lg,rcond=None)[0]
print('decoder grid maxerr [d8,d9,p8,p9]',np.abs(Xg@Wd-Lg).max(0).round(4))
Xd=np.hstack([DAu,np.ones((len(DAu),1))])@Wd
print('decoder data maxerr d8 %.4f d9 %.4f'%(np.abs(Xd[:,0]-DAm[:,2]).max(),np.abs(Xd[:,1]-DAm[:,3]).max()))
```
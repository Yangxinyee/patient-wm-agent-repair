Run 288 settled the formats. `CAPT` is a list of 798 drug-bundle captures (hf, e, u, ctx as double tensors). Each has `info` = {di, hi: insulin dose / present; dn, hn: norepinephrine dose / present}. `keygrads` returns the 16 attention-score gradients (4 slots × 4 heads) with respect to the u token; because the z-token is a copy of u, these are also its gradients. The failure was only a float32-vs-float64 mismatch: the replica runs in double precision.

The prelude is one self-attention layer plus a feed-forward layer on each token, and E takes only the 4 slot rows. So the z-token affects E through two routes:
1. Its attention scores: 16 numbers per bundle, the route that saturates.
2. Its values.

This run builds the new injection direction separately for each drug:
- Target: the change in E that one extra unit of dose causes in the original model at each recorded bundle, using the u-direction `VD_`.
- Restrict candidate directions to the complement of the top-32 attention-score-gradient subspace, so the first-order effect on attention is zero.
- Fit, by ridge least squares over all drug bundles, the direction whose effect on E through the copy token best reproduces that target.

I'll report the fit's R² and check linearity at amplitudes up to 10. Then I'll swap the new directions into `VD_`, confirm `mkW3` uses them as I expect, and run quick rollout and NLL tests at gains 1, 2 and 4.

If R² is high and the response stays linear, the readouts should finally grow with gain, symmetrically for ×2 and ×½, approaching my estimated effects of about −12.7 / +6.4 / +5.8 / −2.9. A low R² would mean the value route alone cannot carry the dose effect.

```python
import time
td9=lambda x: (x if torch.is_tensor(x) else torch.tensor(np.asarray(x))).to(T)
if 'VD_old' not in globals(): VD_old={k9:np.array(v9) for k9,v9 in VD_.items()}
Wc9=mkW3(0.0,0.0,W0z)
def ztok(W,ctx,e,u): return blin(W,bmlp(W,torch.cat([ctx,e,u],-1),'z_post')[:,:32],'z_in')
grp9={'ins':[q for q in CAPT if q['info'].get('hi',0)>0],'ne':[q for q in CAPT if q['info'].get('hn',0)>0]}
VDN={}; eps9=1e-3
for d9,cl9 in grp9.items():
    t9=time.time(); nb9=len(cl9)
    hf9=torch.stack([td9(q['hf']) for q in cl9]); e9=torch.stack([td9(q['e']) for q in cl9]); u9=torch.stack([td9(q['u']) for q in cl9]); x9=torch.stack([td9(q['ctx']) for q in cl9])
    vd9=td9(VD_old[d9])[None]
    E0o=E_of(W0z,hf9,0.0,e9,u9,ztok(W0z,x9,e9,u9)); E1o=E_of(W0z,hf9,0.0,e9,u9+eps9*vd9,ztok(W0z,x9,e9,u9+eps9*vd9))
    Tb9=((E1o-E0o)/eps9).reshape(nb9,-1)
    z09=ztok(Wc9,x9,e9,u9)
    KG9=torch.cat([td9(keygrads(q)[1]) for q in cl9],0); _,S9,V9=torch.linalg.svd(KG9,full_matrices=True)
    C9=V9[32:].T; nc9=C9.shape[1]
    Ez9=E_of(Wc9,hf9,0.0,e9,u9,z09)
    JC9=torch.zeros(nb9,768,nc9,dtype=T)
    for b9 in range(nb9):
        Ep9=E_of(Wc9,hf9[b9:b9+1].repeat(nc9,1,1),0.0,e9[b9:b9+1].repeat(nc9,1),u9[b9:b9+1].repeat(nc9,1),z09[b9:b9+1]+eps9*C9.T)
        JC9[b9]=((Ep9-Ez9[b9:b9+1])/eps9).reshape(nc9,-1).T
    A9=JC9.reshape(-1,nc9); y9=Tb9.reshape(-1); AtA9=A9.T@A9
    w9=torch.linalg.solve(AtA9+1e-4*float(torch.diagonal(AtA9).mean())*torch.eye(nc9,dtype=T),A9.T@y9)
    r29=1-float(((y9-A9@w9)**2).sum()/(y9**2).sum()); v9=C9@w9; VDN[d9]=v9.numpy().copy()
    idx9=torch.arange(min(nb9,80)); lin9=[]
    for Am in [1.0,4.0,10.0]:
        for nmv,dv in [('new',v9[None]),('old',vd9)]:
            dE9=(E_of(Wc9,hf9[idx9],0.0,e9[idx9],u9[idx9],z09[idx9]+Am*dv)-Ez9[idx9]).reshape(len(idx9),-1); tg9=Am*Tb9[idx9]
            lin9.append((nmv,Am,round(float(((dE9-tg9)**2).sum()/(tg9**2).sum()),3)))
    print(d9,'nb',nb9,'copy cos %.4f'%float(F.cosine_similarity(z09,u9).mean()),'R2 %.4f'%r29,'|v| %.2f |vd| %.2f |u-mu| %.1f'%(float(v9.norm()),float(vd9.norm()),float((u9-u9.mean(1,keepdim=True)).norm(dim=1).mean())),
          'relerr(new/old,A):',lin9,'%.0fs'%(time.time()-t9))
VD_.update({k9:VDN[k9] for k9 in VDN})
Wa9=mkW3(1.0,0.0,W0z); Dz9=(Wa9['z_in.weight']-Wc9['z_in.weight']).numpy(); cn9=np.linalg.norm(Dz9,axis=0); j9=int(np.argmax(cn9))
print('z_in col changed', j9, 'n nonzero cols', int((cn9>1e-9).sum()), 'cos with VDN ins %.4f ratio %.3f'%(Dz9[:,j9]@VDN['ins']/cn9[j9]/np.linalg.norm(VDN['ins']), cn9[j9]/np.linalg.norm(VDN['ins'])))
st40=list(BUN.keys())[:40]
for g9 in [1.0,2.0,4.0]:
    Wn9=mkW3(g9,g9,W0z); rr9=evalW(Wn9,st40); print('NEWDIR gain',g9,{kk:vv[0] for kk,vv in rr9.items()},'tfnll %.5f'%tfnll(Wn9,st40)[0])
```
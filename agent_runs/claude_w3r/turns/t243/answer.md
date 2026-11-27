Facts recovered from ARCHITECTURE.md:
- The action encoder is a DeepSet, `u = norm(mlp2(Σ mlp1(atom)/√n + size(log(1+n))))`. Each drug atom carries the numbers `[z_rate, has_rate, z_dose, …]` through `embed.numeric`.
- The dose edit changes those z-scores and the rate/dose bins.
- `pending_mlp` and the filter only feed the action heads, so there is no direct path from running infusions to the value head.

**Conclusion:** the saturation I saw comes from the bins and the encoder's nonlinearity. But z_rate may be linear in the rate, and it extrapolates under ×2.

**New option for stage 1:** inside `encode_action`, use dead hidden units of mlp1 as a gate on the drug code (an exact, clean indicator). They would pass z_rate straight through, and mlp2 would carry it linearly into u. That would give a rate channel that stays linear under the dose edit. If that direction is also nearly empty for all other prelude tokens, head2 could read it with no prelude gating at all.

**Checks this run:**
1. Encoder shapes and dead units on a grid of drug atoms.
2. Whether z_rate is linear in the rate.
3. How much the pre-norm scale of u varies.
4. Whether all-token normalised inputs have near-empty directions.

**Test:** dead units present, linear z, and small scale variation would confirm the encoder route. Otherwise I stay with the prelude-gating route.

```python
for n in ['encode_action.mlp1.0.weight','encode_action.mlp1.2.weight','encode_action.mlp2.0.weight','encode_action.mlp2.2.weight','encode_action.size.weight','encode_action.norm.weight']:
    print(n, SHAPES[n])
cks=[k for k in CODE if ('insulin' in k or 'norepi' in k)]; print('codes', cks)
ci=[k for k in cks if 'insulin' in k][0]; cn=[k for k in cks if 'norepi' in k][0]
try: print('zsc ins rate', [round(float(zsc(ci,'rate',x)),3) for x in [0.5,1,2,4,8]], 'ne', [round(float(zsc(cn,'rate',x)),3) for x in [0.1,0.5,1,2,4]])
except Exception as e: print('zsc err', e)
E_=lambda n: Wt['encode_action.'+n].numpy()
gelu=lambda x: 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
try:
    av=lambda c,r: np.asarray(atom_vec(c,float(r),0)).reshape(-1)
    A=[av(ci,r) for r in np.arange(0.25,9.01,0.25)]+[av(cn,r) for r in np.arange(0.02,3.01,0.02)]
    A=np.array(A); H1=A@E_('mlp1.0.weight').T+E_('mlp1.0.bias'); print('mlp1 pre max per unit: dead(<-2)=',int((H1.max(0)<-2).sum()),'of',H1.shape[1])
    m1=lambda X: gelu(X@E_('mlp1.0.weight').T+E_('mlp1.0.bias'))@E_('mlp1.2.weight').T+E_('mlp1.2.bias')
    S=[]
    for ri in [0,0.5,1,2,3,4,6,8]:
        for rn in [0,0.1,0.3,0.6,1.0,2.0]:
            at=([av(ci,ri)] if ri>0 else [])+([av(cn,rn)] if rn>0 else [])
            if not at: continue
            n_=len(at); s=m1(np.array(at)).sum(0)/np.sqrt(n_)+E_('size.weight').reshape(-1)*np.log(1+n_)+E_('size.bias'); S.append(s)
    S=np.array(S); H2=S@E_('mlp2.0.weight').T+E_('mlp2.0.bias'); print('mlp2 pre: dead(<-2)=',int((H2.max(0)<-2).sum()),'of',H2.shape[1])
    P2=gelu(H2)@E_('mlp2.2.weight').T+E_('mlp2.2.bias'); sd=P2.std(1); print('pre-norm sd across combos: mean %.3f, cv %.3f'%(sd.mean(), sd.std()/sd.mean()))
    uchk2=u_of(2.0,0.6); print('check u_of vs local:', 'n/a')
except Exception as e: print('enc err', repr(e))
Xall=n1.reshape(-1,192); Cv=np.cov(Xall.T); ev_=np.linalg.eigvalsh(Cv)
print('all-token n1 cov eig: smallest 8', np.round(ev_[:8],6), '| #<1e-4', int((ev_<1e-4).sum()), '#<1e-3', int((ev_<1e-3).sum()), '#<1e-2', int((ev_<1e-2).sum()))
```
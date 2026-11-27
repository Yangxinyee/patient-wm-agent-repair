All measurements fall on the hour (minute 0) and all drug starts five minutes later (minute 5). They never share a bundle. Each hour is a measurement bundle, then a drug bundle that also holds the previous hour's END atoms. So the dose at h:05 follows the glucose read at h:00, and its first effect shows at (h+1):00.

That shapes the design:
- **Write:** a gated write at insulin bundles only, of the raw rate. Dims needed: memory, a reference, a raw staging dim and a gated staging dim (NE would add three).
- **Compensation:** done at the readout, not in the core. The glucose_finger readout unit, in its linear regime, can add any linear function of the head's query vector q. Fitting that function by regression on factual data removes the part of the dose memory the model already knows (the policy-implied dose), which keeps factual fit. In the rollout the extra dose still passes through.

I'll put a kill function in /work/lib1.py with all rules written out. A larger evaluation set (300 fresh train stays) will give tighter cost estimates. Then I'll compare the factual cost of killing the old role dims against the smallest-rms dims, for 4 and 7 dims.

```python
code=r'''
import torch, numpy as np
BLK=['core.blocks.0.0','core.blocks.0.1']
def cl(P0): return {k:(v.clone() if torch.is_tensor(v) else torch.as_tensor(np.asarray(v)).clone()).float() for k,v in P0.items()}
def kill(P0, J, mem=()):
    """Cut h dims J out of the model (no writers, no readers). mem: dims that will carry memory (also cut from ctx->z)."""
    P=cl(P0); J=list(J)
    P['core.slots'][:,J]=0
    pre='core.prelude.0'
    P[pre+'.attn.out.weight'][J,:]=0
    P[pre+'.ffn.2.weight'][J,:]=0; P[pre+'.ffn.2.bias'][J]=0
    P[pre+'.ffn.0.weight'][:,J]=0
    P['core.prelude_norm.weight'][J]=0; P['core.prelude_norm.bias'][J]=0
    for b in BLK:
        for nm in ['attn.out','cross.out']: P[f'{b}.{nm}.weight'][J,:]=0
        P[f'{b}.ffn.2.weight'][J,:]=0; P[f'{b}.ffn.2.bias'][J]=0
        for n in ['norm1','norm2','norm3']: P[f'{b}.{n}.weight'][J]=0
    P['core.b_in.weight'][J,:]=0; P['core.b_in.weight'][:,J]=0
    P['core.c_out.weight'][J,:]=0; P['core.c_out.weight'][:,J]=0
    c='core.coda.0'
    P[c+'.attn.out.weight'][J,:]=0; P[c+'.ffn.2.weight'][J,:]=0; P[c+'.ffn.2.bias'][J]=0
    P[c+'.norm1.weight'][J]=0; P[c+'.norm3.weight'][J]=0
    P['core.out_norm.weight'][J]=0; P['core.out_norm.bias'][J]=0
    P['gap.2.weight'][J,:]=0; P['gap.2.bias'][J]=0
    P['z_head.weight'][J,:]=0; P['z_head.bias'][J]=0
    P['type_emb.weight'][:,J]=0
    P['query_norm.weight'][J]=0
    for m in mem:
        P['z_post.0.weight'][:,m]=0; P['z_prior.0.weight'][:,m]=0
    return P
'''
open('/work/lib1.py','w').write(code); exec(code)
rng=np.random.default_rng(1); allst=np.unique(sid); s2=set(sel2)
ev_st=[int(s) for s in rng.choice(allst,400,replace=False) if s not in s2][:300]
t0=time.time(); dsE=build_ds(tr, ev_st, 0); print('build %.1fs'%(time.time()-t0))
def cost(P):
    r=tf_eval(P, dsE, False); return r[0]
t0=time.time(); base=cost(PO); print('PO nll %.5f n=%d  %.1fs'%(float(base.mean()), len(base), time.time()-t0))
small=[1,17,4,132,33,36,2,101,91,12,9,57]
for nm,J in [('roles8',JALL),('small4',small[:4]),('small7',small[:7]),('roles4',[132,73,105,155])]:
    d=(cost(kill(PO,J))-base).numpy(); print(nm, J, 'dNLL %.5f se %.5f'%(d.mean(), d.std()/np.sqrt(len(d))))
```
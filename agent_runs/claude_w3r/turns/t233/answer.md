`u_of(ins, ne)` builds the atoms with `atom_vec`, applies mlp1 to each, sums, divides by √n, adds `size(log(1+n))`, then applies mlp2 and the norm. I will rebuild this so that I can see mlp2's 768 hidden activations. I will check the rebuild against `u_of`, then evaluate it on all 2,406 recorded action bundles plus their four dose-edited versions, to match the evaluation rollouts.

On that set I will fit linear reads φ (768-dim) of the insulin and norepinephrine rates, and measure how much the encoder norm's scale σ_pre varies. Success means residual sd well under 0.05 rate units and σ_pre varying by less than about 10%, so that the injected signal's gain stays nearly constant.

```python
import torch, torch.nn.functional as Fnn
def enc_parts(ins=None, ne=None):
    A=[]
    if ins is not None: A.append(atom_vec('insulin',ins))
    if ne is not None: A.append(atom_vec('norepinephrine',ne))
    X=torch.stack([torch.as_tensor(a,dtype=torch.float64) for a in A]); n=len(A)
    x=mlp_t(X,'encode_action.mlp1').sum(0)/math.sqrt(n)+Wt['encode_action.size.weight'][:,0]*math.log(1+n)+Wt['encode_action.size.bias']
    h2=Fnn.gelu(Wt['encode_action.mlp2.0.weight']@x+Wt['encode_action.mlp2.0.bias'])
    pre=Wt['encode_action.mlp2.2.weight']@h2+Wt['encode_action.mlp2.2.bias']
    return h2.numpy(), pre.numpy()
import math
h2t,pret=enc_parts(1.5,0.3); ut=u_of(1.5,0.3); ut=np.asarray(ut.detach() if hasattr(ut,'detach') else ut).ravel()
pre_t=torch.tensor(pret); uchk=Fnn.layer_norm(pre_t,(192,),Wt['encode_action.norm.weight'],Wt['encode_action.norm.bias'],1e-5).numpy()
print('recon err', np.abs(uchk-ut).max(), np.abs(ut).max())
pa=PD[act][['ins_s','ne_s','has_ins_s','has_ne_s']].values
TOKS=[]
for (ri,rn,hi,hn) in pa:
    ri=ri if hi else None; rn=rn if hn else None
    for (mi,mn) in [(1,1),(2,1),(.5,1),(1,2),(1,.5)]:
        TOKS.append((None if ri is None else ri*mi, None if rn is None else rn*mn))
TOKS=list(dict.fromkeys(TOKS)); print('n unique tokens', len(TOKS))
H2E=np.zeros((len(TOKS),768)); PREE=np.zeros((len(TOKS),192))
for i,(a,b) in enumerate(TOKS): H2E[i],PREE[i]=enc_parts(a,b)
yI=np.array([0 if a is None else a for a,b in TOKS]); yN=np.array([0 if b is None else b for a,b in TOKS])
sig=PREE.std(1); print('sigma_pre mean %.3f cv %.3f min %.3f max %.3f'%(sig.mean(), sig.std()/sig.mean(), sig.min(), sig.max()))
for nm,y in [('ins',yI),('ne',yN)]:
    for lam in [1e-6,1e-4]:
        Xa=np.c_[H2E,np.ones(len(H2E))]; c=np.linalg.solve(Xa.T@Xa+lam*len(Xa)*np.eye(769),Xa.T@y); r=y-Xa@c
        print(nm,lam,'resid sd %.4f max %.3f |phi| %.2f'%(r.std(), np.abs(r).max(), np.linalg.norm(c[:-1])))
```
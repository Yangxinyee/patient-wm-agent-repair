The states file lines up with `BUN` exactly. Same stays and hours, one row per bundle, the state after the bundle. Alignment is not the problem.

The clue is `zerr`, the function that produced the earlier match. It applied the z_post MLP with a **ReLU** default. So the activation differs by module, and I wrongly assumed SiLU everywhere.

Next I make the activation configurable per module family: encoder MLPs, gap MLP, z_post, and later the core FFNs and the age MLP. Then I grid-search the z-posterior match over SiLU, GELU, GELU-tanh and ReLU, using the stored states as context. One combination giving errors near 1e-5 at all bundles would fix the encoder, gap and z_post activations. After that I can search the core FFN activations against `out`/`h`.

```python
code3 = r'''
ACTMAP = {}
_ACTS = {'silu':F.silu,'gelu':F.gelu,'gelu_tanh':(lambda x: F.gelu(x,approximate='tanh')),'relu':F.relu}
def act_for(n):
    for p,a in ACTMAP.items():
        if n.startswith(p): return _ACTS[a]
    return F.silu
def mlp(W, x, n): return lin(W, act_for(n)(lin(W, x, n+'.0')), n+'.2')
'''
open('/work/wm.py','a').write(code3); exec(code3)
s=2020000000; rows=row_of[s]; bl=BUN[s]; ks=list(range(1,30))
res=[]
import itertools
names_a=['silu','gelu','gelu_tanh','relu']
pre={}
for a1,a2 in itertools.product(names_a,names_a):
    ACTMAP.clear(); ACTMAP.update({'encode':None}); ACTMAP['encode.mlp1']=a1; ACTMAP['encode.mlp2']=a2; ACTMAP['encode_action.mlp1']=a1; ACTMAP['encode_action.mlp2']=a2
    del ACTMAP['encode']
    E=[]
    for k in ks:
        meas=[a for a in bl[k][1] if not a['act']]; act=[a for a in bl[k][1] if a['act']]
        e=set_enc(W0,meas,'encode') if meas else torch.zeros(192,dtype=T)
        u=set_enc(W0,act,'encode_action') if act else torch.zeros(192,dtype=T)
        E.append((e,u))
    pre[(a1,a2)]=E
ctxs={}
for ga in names_a:
    ACTMAP.clear(); ACTMAP['gap']=ga
    C=[]
    for k in ks:
        prev=torch.tensor(X[rows[k-1]],dtype=T); out=prev[:192]; h=prev[192:960].reshape(4,192)
        dt=(bl[k][0]-bl[k-1][0])*3600
        hf=h*torch.exp(-dt*torch.exp(W0['core.flow.a']))
        C.append(out+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap'))
    ctxs[ga]=C
for (a1,a2),E in pre.items():
    for ga,C in ctxs.items():
        for za in names_a:
            ACTMAP.clear(); ACTMAP['z_post']=za
            errs=[ (mlp(W0,torch.cat([C[i],E[i][0],E[i][1]]),'z_post')[:32]-torch.tensor(X[rows[k],960:],dtype=T)).abs().max().item() for i,k in enumerate(ks)]
            res.append((np.median(errs),max(errs),a1,a2,ga,za))
res.sort()
for r in res[:8]: print(r)
```
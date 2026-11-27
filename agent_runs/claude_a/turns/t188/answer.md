`tf_loop(P, ds, keep)` returns h states, out states and a store dict, which holds only z by default. The sel2 store `ST2` also has prelude attention weights (aw), E, hf and hn, so `keep` presumably selects what to store. The ds dict holds the encoded e and u per bundle, time gaps, and atom/bundle/reading tables.

I'll write my own explicit prelude function to /work/lib2.py and validate it against `tf_loop`'s stored attention weights and E on the 20-stay set. It returns tokens, attention weights, per-head outputs, residuals and E. A close match (errors near float precision) means I can compute exactly the internals the write channel and gates must be fitted on.

```python
code2=r'''
import torch, math, torch.nn.functional as F
def LNx(x, P, pre): return F.layer_norm(x, (x.shape[-1],), P[pre+'.weight'], P[pre+'.bias'], 1e-5)
def prelude(P, hf, e, u, z):
    zt = F.linear(z, P['z_in.weight'], P['z_in.bias'])
    X = torch.cat([hf + P['core.slots'][None], e[:,None], u[:,None], zt[:,None]], 1)
    pre='core.prelude.0'; B=X.shape[0]; H=4; dh=48
    n1 = LNx(X, P, pre+'.norm1')
    q = F.linear(n1, P[pre+'.attn.q.weight']); k,v = F.linear(n1, P[pre+'.attn.kv.weight']).split(192,-1)
    sh=lambda t: t.view(B,7,H,dh).transpose(1,2)
    att = torch.softmax(sh(q)@sh(k).transpose(-1,-2)/math.sqrt(dh), -1)
    o = att@sh(v)
    X1 = X + F.linear(o.transpose(1,2).reshape(B,7,192), P[pre+'.attn.out.weight'])
    n3 = LNx(X1, P, pre+'.norm3')
    X2 = X1 + F.linear(F.gelu(F.linear(n3, P[pre+'.ffn.0.weight'], P[pre+'.ffn.0.bias'])), P[pre+'.ffn.2.weight'], P[pre+'.ffn.2.bias'])
    E = LNx(X2[:,:4], P, 'core.prelude_norm')
    return dict(X=X, n1=n1, att=att, o=o, X1=X1, n3=n3, X2=X2, E=E, vh=sh(v))
'''
open('/work/lib2.py','w').write(code2); exec(code2)
r=tf_loop(PO, ds, keep=('aw','E','hf','hn','z'))
st=r[2]; print('store keys', list(st.keys()))
if 'E' in st:
    pr=prelude(PO, st['hf'], ds['e'], ds['u'], st['z'])
    print('E err %.3g (scale %.3g)'%(float((pr['E']-st['E']).abs().max()), float(st['E'].abs().mean())))
    print('aw shape', tuple(st['aw'].shape), 'att err %.3g'%float((pr['att']-st['aw']).abs().max()))
```
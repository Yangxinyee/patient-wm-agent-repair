Next I'll measure, for each of the 4 prelude heads, how much the slot queries attend to the action token at drug bundles. Injecting through attention only works if that weight is large and stable. I'll write a fresh small prelude-attention computation from the saved states and local encoders.

```python
import torch
T=lambda n: load(n).double()
fa_=T('core.flow.a'); print('flow.a range', float(fa_.min()), float(fa_.max()))
def lnorm(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdim=True); v=x.var(-1,unbiased=False,keepdim=True); return (x-m)/torch.sqrt(v+eps)*w+b
def pre_tokens(hprev,dt,e,u,z):
    hf=hprev*torch.exp(-dt*torch.exp(fa_))
    zt=z@T('core.z_in.weight').T+T('core.z_in.bias') if 'core.z_in.weight' in NAMES else z@T('z_in.weight').T+T('z_in.bias')
    return torch.cat([hf+T('core.slots'), e[None], u[None], zt[None]],0)
def pre_probs(X):
    n=lnorm(X,T('core.prelude.0.norm1.weight'),T('core.prelude.0.norm1.bias'))
    q=n@T('core.prelude.0.attn.q.weight').T; kv=n@T('core.prelude.0.attn.kv.weight').T; k=kv[:,:192]
    q=q.view(-1,4,48).transpose(0,1); k=k.view(-1,4,48).transpose(0,1)
    return torch.softmax(q@k.transpose(1,2)/np.sqrt(48),-1)  # heads x tok x tok
print('ok')
```
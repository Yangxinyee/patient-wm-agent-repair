All 2,319 drug bundles have a previous bundle in the same stay. core.flow.a is a 192-vector, so the time flow is a per-dimension decay applied to every slot.

Now I run the prelude replica. I'll report the slots' mean attention on the e_k, u_k and z tokens (other slots get the rest), and how much each token's value moves the slot outputs. This tells me which token can carry an injection into E.

```python
Wd = {n: load_original(n) for n in NAMES if n.startswith('core.prelude') or n.startswith('z_in') or n in ('core.slots','core.flow.a','core.prelude_norm.weight','core.prelude_norm.bias')}
def LNt(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdim=True); v = x.var(-1, unbiased=False, keepdim=True)
    return (x-m)/torch.sqrt(v+eps)*w + b
hprev = torch.tensor(X3[prev][:, 192:960]).reshape(-1,4,192)
dt = torch.tensor((hr[rows]-hr[prev])*3600.0, dtype=torch.float32)[:,None,None]
fa = Wd['core.flow.a']
htf = hprev*torch.exp(-dt*torch.exp(fa))
ek = EU3[:, :192].float(); uk = EU3[:, 192:].float()
zk = torch.tensor(Z3[rows]).float()
ztok = zk @ Wd['z_in.weight'].T + Wd['z_in.bias']
TOK = torch.cat([htf + Wd['core.slots'][None], ek[:,None], uk[:,None], ztok[:,None]], 1)  # (N,7,192)
def prelude(TOK, W=Wd, ret=False):
    p='core.prelude.0.'
    x = TOK
    n1 = LNt(x, W[p+'norm1.weight'], W[p+'norm1.bias'])
    q = n1 @ W[p+'attn.q.weight'].T
    kv = n1 @ W[p+'attn.kv.weight'].T
    k, v = kv[..., :192], kv[..., 192:]
    N, T, _ = x.shape
    qh = q.reshape(N,T,4,48).transpose(1,2); kh = k.reshape(N,T,4,48).transpose(1,2); vh = v.reshape(N,T,4,48).transpose(1,2)
    att = torch.softmax(qh @ kh.transpose(-1,-2)/math.sqrt(48), -1)  # N,4,T,T
    o = (att @ vh).transpose(1,2).reshape(N,T,192) @ W[p+'attn.out.weight'].T
    x = x + o
    n3 = LNt(x, W[p+'norm3.weight'], W[p+'norm3.bias'])
    f = torch.nn.functional.gelu(n3 @ W[p+'ffn.0.weight'].T + W[p+'ffn.0.bias']) @ W[p+'ffn.2.weight'].T + W[p+'ffn.2.bias']
    x = x + f
    E = LNt(x[:, :4], W['core.prelude_norm.weight'], W['core.prelude_norm.bias'])
    if ret: return E, att, vh, x
    return E
E0, att0, vh0, x0 = prelude(TOK, ret=True)
a_s = att0[:, :, :4, :]  # slots as queries
print('mean attn from slots to [e,u,z] per head:')
for hh in range(4):
    print(hh, np.round(a_s[:,hh,:,4:].mean((0,1)).numpy(),3), 'to slots', np.round(a_s[:,hh,:,:4].sum(-1).mean().item(),3))
```
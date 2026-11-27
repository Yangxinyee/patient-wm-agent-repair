The filtered view with `pending_mlp` and `filter_gate` only feeds the action (dose) heads. It doesn't feed the glucose/MAP value heads, so it's no shortcut. For value predictions, dose information must still pass through the core: the action encoding `u`, then the prelude attention, then the slot state.

Next I'll write my own teacher-forced loop around the verified `cstep` and a standalone prelude that exposes its attention weights. I'll check it two ways: the loop's states against the fwd_2 states, and my prelude's `E` against `cstep`'s `E`. Then I'll report how strongly the slot queries attend to the `u` and `z` tokens at insulin bundles. Large, stable attention to `u` in some head would make the prelude a usable write point for a dose signal. Near-zero attention would explain why the model ignores doses and send me to a different write route.

```python
import torch, math, numpy as np
W = W64
print("W64 type:", type(W['z_in.weight']), W['z_in.weight'].dtype)
def LNt(x, n, eps=1e-5):
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return (x-mu)/torch.sqrt(var+eps)*W[n+'.weight'] + W[n+'.bias']
def prelude_full(hf, e, u, z):
    P='core.prelude.0.'
    zt = z @ W['z_in.weight'].T + W['z_in.bias']
    tok = torch.cat([hf + W['core.slots'], e[None], u[None], zt[None]], 0)
    a = LNt(tok, P+'norm1')
    q = a[:4] @ W[P+'attn.q.weight'].T
    kv = a @ W[P+'attn.kv.weight'].T
    k, v = kv[:, :192], kv[:, 192:]
    qh = q.view(4,4,48).transpose(0,1); kh = k.view(7,4,48).transpose(0,1); vh = v.view(7,4,48).transpose(0,1)
    att = torch.softmax(qh @ kh.transpose(1,2)/math.sqrt(48), -1)
    o = (att @ vh).transpose(0,1).reshape(4,192) @ W[P+'attn.out.weight'].T
    x = tok[:4] + o
    f = LNt(x, P+'norm3')
    f = torch.nn.functional.gelu(f @ W[P+'ffn.0.weight'].T + W[P+'ffn.0.bias']) @ W[P+'ffn.2.weight'].T + W[P+'ffn.2.bias']
    x = x + f
    return LNt(x, 'core.prelude_norm'), att, x, tok
def run_tf(sid, kmax=None):
    B = bundles_B[sid]
    out = torch.zeros(192, dtype=torch.float64); h = torch.zeros(4,192, dtype=torch.float64)
    Mv, Mt, recs = [], [], []
    t_prev = None
    for k,(t,ats) in enumerate(B[:kmax] if kmax else B):
        dt_s = 1.0 if t_prev is None else (t - t_prev)*3600.0
        e,u = enc2(ats)
        Mv.append(e); Mt.append(t)
        Mv_ = torch.stack(Mv[-128:]); Mage = torch.tensor([(t-tt)*3600.0 for tt in Mt[-128:]], dtype=torch.float64)
        out_n, y, z, E, hf = cstep(out, h, dt_s, e, u, Mv_, Mage)
        recs.append(dict(t=t, ats=ats, out_p=out, h_p=h, dt=dt_s, e=e, u=u, z=z, E=E, hf=hf, out=out_n, y=y))
        out, h = out_n, y; t_prev = t
    return recs
sid0 = SIDS[0] if 'SIDS' in globals() else list(bundles_B.keys())[0]
R0 = run_tf(sid0)
rows0 = np.where(st_sid==sid0)[0]
Xs0 = np.stack([np.concatenate([r['out'].numpy(), r['y'].numpy().reshape(-1), r['z'].numpy()]) for r in R0])
print("n bundles", len(R0), "rows", len(rows0), "max abs err vs fwd_2:", np.abs(Xs0 - X_[rows0]).max() if len(rows0)==len(R0) else 'len mismatch')
errE = max(float((prelude_full(r['hf'], r['e'], r['u'], r['z'])[0] - r['E']).abs().max()) for r in R0)
print("max |E_mine - E_cstep|:", errE)
```
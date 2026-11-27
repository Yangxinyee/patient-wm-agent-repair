The prelude's slot tokens use a learned embedding (core.slots), so slot tokens are hf + core.slots. The gap, age and z networks are two-layer MLPs that take a scalar input.

Next I code the transparent step from the documented architecture and compare it with cstep_h's stored outputs on every bundle of one stay. The step has hooks for adding to the slot residual P (before prelude_norm) and to h_new. If the maximum relative error is around 1e-9 on out, y, z and E, and the sixth output equals h_new, the function is exact and becomes my design tool. A mismatch would point to a specific mis-documented detail, such as age scaling or the gap input.

```python
def Wt(n): return W64[n]
def LNn(x, pre, eps=1e-5): return F_.layer_norm(x, (x.shape[-1],), Wt(pre+'.weight'), Wt(pre+'.bias'), eps)
def linb(x, pre):
    y = x @ Wt(pre+'.weight').T
    b = W64.get(pre+'.bias')
    return y + b if b is not None else y
def mlp2_(x, pre): return linb(F_.gelu(linb(x, pre+'.0')), pre+'.2')
def mha_(xq, xkv, pre, nh=4):
    q = xq @ Wt(pre+'.q.weight').T; kv = xkv @ Wt(pre+'.kv.weight').T
    k, v = kv[:, :192], kv[:, 192:]
    qh = q.view(-1, nh, 48).transpose(0,1); kh = k.view(-1, nh, 48).transpose(0,1); vh = v.view(-1, nh, 48).transpose(0,1)
    att = torch.softmax(qh @ kh.transpose(1,2) / math.sqrt(48), -1)
    return (att @ vh).transpose(0,1).reshape(-1, 192) @ Wt(pre+'.out.weight').T, att
def mystep(d, R=None, dP=None, dhn=None):
    out0, h0, dt, e, u, Mv, Ma = d['out0'], d['h0'], d['dt'], d['e'], d['u'], d['Mv'], d['Ma']
    hf = h0 * torch.exp(-dt * torch.exp(Wt('core.flow.a')))
    gp = mlp2_(torch.tensor([math.log(dt)/10.0], dtype=torch.float64), 'gap')
    ctx = out0 + LNn(hf.mean(0), 'core.out_norm') + gp
    z = mlp2_(torch.cat([ctx, e, u]), 'z_post')[:32]
    toks = torch.cat([hf + Wt('core.slots'), e[None], u[None], linb(z, 'z_in')[None]])
    n1 = LNn(toks, 'core.prelude.0.norm1'); ao, att = mha_(n1, n1, 'core.prelude.0.attn')
    x1 = toks + ao
    n3 = LNn(x1, 'core.prelude.0.norm3'); pre_h = linb(n3, 'core.prelude.0.ffn.0'); ph = F_.gelu(pre_h)
    P = (x1 + linb(ph, 'core.prelude.0.ffn.2'))[:4]
    if dP is not None: P = P + dP
    E = LNn(P, 'core.prelude_norm'); xb = hf + E
    Mk = Mv + mlp2_(torch.log1p(Ma)[:, None], 'core.age')
    for l in (0, 1):
        p = f'core.blocks.0.{l}'
        n = LNn(xb, p+'.norm1'); a_, _ = mha_(n, n, p+'.attn'); xb = xb + a_
        n = LNn(xb, p+'.norm2'); c_, _ = mha_(n, Mk, p+'.cross'); xb = xb + c_
        n = LNn(xb, p+'.norm3'); xb = xb + linb(F_.gelu(linb(n, p+'.ffn.0')), p+'.ffn.2')
    sp_d = F_.softplus(Wt('core.delta_in')); abar_ = torch.exp(-sp_d * torch.exp(Wt('core.a_in')))
    h_new = abar_ * hf + sp_d * (E @ Wt('core.b_in.weight').T) + (xb - (hf + E))
    if dhn is not None: h_new = h_new + dhn
    c = h_new @ Wt('core.c_out.weight').T
    n = LNn(c, 'core.coda.0.norm1'); a_, _ = mha_(n, n, 'core.coda.0.attn'); c = c + a_
    n = LNn(c, 'core.coda.0.norm3'); y = c + linb(F_.gelu(linb(n, 'core.coda.0.ffn.0')), 'core.coda.0.ffn.2')
    out = LNn(y.mean(0), 'core.out_norm')
    if R is not None: R.update(dict(hf=hf, toks=toks, n1=n1, att=att, x1=x1, n3=n3, pre_h=pre_h, ph=ph, P=P, E=E, h_new=h_new, y=y, z=z))
    return out, y, z, E, hf, h_new
stepsB = run_tf(B, keep=True)
errs = np.zeros((len(stepsB), 6))
for i, d in enumerate(stepsB):
    o = mystep(d)
    for j in range(6):
        ref = d['res'][j]; errs[i, j] = float((o[j]-ref).norm() / (ref.norm() + 1e-12))
print('n', len(stepsB), 'max rel err per output', np.round(errs.max(0), 10))
print('dtype', W64['core.slots'].dtype, stepsB[0]['h0'].dtype)
```
The architecture's last section confirms how the evaluation rollout works. From hour 13 the model samples its own glucose and MAP values and feeds them back. Every insulin atom from hour 12 on has its rate and dose scaled. The readout is the edited-minus-factual predicted mean, averaged over 20 paths with common random numbers.

With the head verified, I can now build the whole pipeline locally, but I need one more piece: a core step I wrote myself that takes a weights dictionary and accepts an injected perturbation. The kernel's `cstep` is opaque, so I can't apply candidate edits or test write paths inside it. I'll reimplement the full core step:
- flow, `ctx`, z posterior;
- the prelude;
- two blocks, each with self-attention, cross-attention to memory plus age embedding, and an FFN;
- the gated h update, then `c_out` and the coda.

I'll check it against `cstep` on cached bundles. The age-embedding input scale is the one uncertain detail, so I'll test both `log1p(age)` and `log1p(age)/10`. A match near 1e-12 would confirm the reimplementation.

```python
def mha_t(xq, xkv, pre, Wd, nh=4):
    q = xq @ Wd[pre+'.q.weight'].T; kv = xkv @ Wd[pre+'.kv.weight'].T
    k, v = kv[:, :192], kv[:, 192:]
    nq, nk = xq.shape[0], xkv.shape[0]
    qh = q.view(nq,nh,48).transpose(0,1); kh = k.view(nk,nh,48).transpose(0,1); vh = v.view(nk,nh,48).transpose(0,1)
    att = torch.softmax(qh @ kh.transpose(1,2)/math.sqrt(48), -1)
    return (att @ vh).transpose(0,1).reshape(nq,192) @ Wd[pre+'.out.weight'].T, att
def LNw(x, n, Wd, eps=1e-5):
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return (x-mu)/torch.sqrt(var+eps)*Wd[n+'.weight'] + Wd[n+'.bias']
def ffn_w(x, pre, Wd):
    return gelu(x @ Wd[pre+'.0.weight'].T + Wd[pre+'.0.bias']) @ Wd[pre+'.2.weight'].T + Wd[pre+'.2.bias']
def core_step(Wd, out_p, h_p, dt, e, u, Mv, Mage, age_scale=1.0, dx_pre=None):
    hf = h_p * torch.exp(-dt*torch.exp(Wd['core.flow.a']))
    g_ = Wd['gap.2.weight'] @ gelu(Wd['gap.0.weight'][:,0]*(math.log(dt)/10) + Wd['gap.0.bias']) + Wd['gap.2.bias']
    ctx = out_p + LNw(hf.mean(0)[None], 'core.out_norm', Wd)[0] + g_
    zz = gelu(torch.cat([ctx, e, u]) @ Wd['z_post.0.weight'].T + Wd['z_post.0.bias']) @ Wd['z_post.2.weight'].T + Wd['z_post.2.bias']
    z = zz[:32]
    zt = z @ Wd['z_in.weight'].T + Wd['z_in.bias']
    tok = torch.cat([hf + Wd['core.slots'], e[None], u[None], zt[None]], 0)
    P='core.prelude.0'
    ao, att = mha_t(LNw(tok, P+'.norm1', Wd)[:4], LNw(tok, P+'.norm1', Wd), P+'.attn', Wd)
    x = tok[:4] + ao
    if dx_pre is not None: x = x + dx_pre
    x = x + ffn_w(LNw(x, P+'.norm3', Wd), P+'.ffn', Wd)
    E = LNw(x, 'core.prelude_norm', Wd)
    xb = hf + E
    ag = gelu(Wd['core.age.0.weight'][:,0][None]*(torch.log1p(Mage)[:,None]*age_scale) + Wd['core.age.0.bias']) @ Wd['core.age.2.weight'].T + Wd['core.age.2.bias']
    Mk = Mv + ag
    for l in range(2):
        B_='core.blocks.%d'%l
        xb = xb + mha_t(LNw(xb, B_+'.norm1', Wd), LNw(xb, B_+'.norm1', Wd), B_+'.attn', Wd)[0]
        xb = xb + mha_t(LNw(xb, B_+'.norm2', Wd), Mk, B_+'.cross', Wd)[0]
        xb = xb + ffn_w(LNw(xb, B_+'.norm3', Wd), B_+'.ffn', Wd)
    spd = torch.nn.functional.softplus(Wd['core.delta_in'])
    ab_ = torch.exp(-spd*torch.exp(Wd['core.a_in']))
    hn = ab_*hf + spd*(E @ Wd['core.b_in.weight'].T) + (xb - (hf + E))
    c = hn @ Wd['core.c_out.weight'].T
    C='core.coda.0'
    c = c + mha_t(LNw(c, C+'.norm1', Wd), LNw(c, C+'.norm1', Wd), C+'.attn', Wd)[0]
    c = c + ffn_w(LNw(c, C+'.norm3', Wd), C+'.ffn', Wd)
    out = LNw(c.mean(0)[None], 'core.out_norm', Wd)[0]
    return out, c, z, E, hf, x
R = run_tf(sid0)
for asc in (1.0, 0.1):
    errs_y=[]
    Mv=[r['e'] for r in R]; Mt=[r['t'] for r in R]
    for k,r in enumerate(R[1:30], start=1):
        lo=max(0,k-127); Mv_=torch.stack(Mv[lo:k+1]); Mage=torch.tensor([(r['t']-tt)*3600.0 for tt in Mt[lo:k+1]],dtype=torch.float64)
        o_,y_,z_,E_,hf_,x_ = core_step(W, r['out_p'], r['h_p'], r['dt'], r['e'], r['u'], Mv_, Mage, age_scale=asc)
        errs_y.append(float((y_-r['y']).abs().max()/r['y'].abs().max()))
    print("age_scale", asc, "max rel err y:", max(errs_y))
```
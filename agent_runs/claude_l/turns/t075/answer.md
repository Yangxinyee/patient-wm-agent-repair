The embedding and both set encoders match my earlier verified functions exactly.

Next I add the full core step: time flow, z posterior, prelude, two recurrent layers with cross-attention to memory, the gated update, and the coda. I test it one step ahead against the tool's states for one stay, across the unknown structural variants:
- whether memory includes the current bundle;
- age in seconds or hours;
- age added to the keys, or to M before the projection;
- absent tokens zeroed, empty-encoded or masked;
- whether action-only bundles enter memory.

If one combination gives an error near 1e-5 on `out`, `h` and `z` for every bundle, the core is reconstructed exactly. Partial matches will show which variant dimension matters.

```python
code2 = r'''
def cross_mem(W, x, L, Mt, ages, age_unit, age_mode):
    if Mt is None or Mt.shape[0]==0: return torch.zeros_like(x)
    a = ages/3600.0 if age_unit=='h' else ages
    ae = mlp(W, torch.log1p(a)[:,None], 'core.age')
    xq = lnorm(W, x, L+'.norm2')
    if age_mode=='key':
        return mha(W, xq, Mt, L+'.cross', kadd=ae)
    else:
        return mha(W, xq, Mt+ae, L+'.cross')
def core_step(W, out, h, dt, e, u, z_override=None, Mt=None, ages=None, opt=None):
    o = opt or {}
    hf = h * torch.exp(-dt * torch.exp(W['core.flow.a']))
    g = mlp(W, torch.tensor([math.log(dt)/10], dtype=T), 'gap')
    ctx = out + lnorm(W, hf.mean(0), 'core.out_norm') + g
    zp = mlp(W, torch.cat([ctx, e, u]), 'z_post'); z = zp[:32] if z_override is None else z_override
    tok = torch.cat([hf + W['core.slots'], e[None], u[None], lin(W, z, 'z_in')[None]], 0)
    mask = o.get('pmask')
    tok = self_layer(W, tok, 'core.prelude.0', mask=mask)
    E = lnorm(W, tok[:4], 'core.prelude_norm')
    x = hf + E
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        a_ = lnorm(W, x, L+'.norm1'); x = x + mha(W, a_, a_, L+'.attn')
        x = x + cross_mem(W, x, L, Mt, ages, o.get('age_unit','s'), o.get('age_mode','key'))
        x = x + mlp(W, lnorm(W, x, L+'.norm3'), L+'.ffn')
    sp = F.softplus(W['core.delta_in']); abar = torch.exp(-sp*torch.exp(W['core.a_in']))
    h2 = abar*hf + sp*(E @ W['core.b_in.weight'].T) + (x - (hf + E))
    y = self_layer(W, h2 @ W['core.c_out.weight'].T, 'core.coda.0')
    return lnorm(W, y.mean(0), 'core.out_norm'), y, z, dict(E=E, hf=hf, ctx=ctx)
'''
open('/work/wm.py','a').write(code2); exec(code2)
def onestep_err(sid, opt, inc_cur, skip_empty, nmax=60):
    bl = BUN[sid]; rows = row_of[sid]
    times=[t for t,_ in bl]
    E_enc=[]; 
    for t,atoms in bl:
        meas=[a for a in atoms if not a['act']]; act=[a for a in atoms if a['act']]
        e = set_enc(W0, meas, 'encode'); u = set_enc(W0, act, 'encode_action') if act else torch.zeros(192,dtype=T)
        E_enc.append((e,u,len(meas)>0))
    errs=[]
    for k in range(1, min(len(bl),nmax)):
        prev = torch.tensor(X[rows[k-1]],dtype=T)
        out=prev[:192]; h=prev[192:960].reshape(4,192)
        dt=(times[k]-times[k-1])*3600
        js=[j for j in range(k+ (1 if inc_cur else 0)) if (E_enc[j][2] or not skip_empty)][-128:]
        Mt=torch.stack([E_enc[j][0] for j in js]) if js else None
        ages=torch.tensor([(times[k]-times[j])*3600 for j in js],dtype=T) if js else None
        e,u,_=E_enc[k]
        o2,y,z,_=core_step(W0,out,h,dt,e,u,Mt=Mt,ages=ages,opt=opt)
        tgt=torch.tensor(X[rows[k]],dtype=T)
        errs.append([ (o2-tgt[:192]).abs().max().item(), (y.reshape(-1)-tgt[192:960]).abs().max().item(), (z-tgt[960:]).abs().max().item()])
    return np.array(errs)
res_v={}
for inc in [False,True]:
    for skip in [False,True]:
        for au in ['s','h']:
            for am in ['key','input']:
                er=onestep_err(2020000000,dict(age_unit=au,age_mode=am),inc,skip,nmax=30)
                res_v[(inc,skip,au,am)]=er
                print(inc,skip,au,am,'out med/max %.4g %.4g | h med/max %.4g %.4g | z max %.3g'%(np.median(er[:,0]),er[:,0].max(),np.median(er[:,1]),er[:,1].max(),er[:,2].max()))
```
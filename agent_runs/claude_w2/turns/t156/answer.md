The tool's dose_edit output gives, per stay-hour from 13 to 24 at recorded readout readings, the path-averaged shift and the rollout's factual predicted mean, plus a "reached" flag. The headline number is the mean over reached stay-hours, pooled across stays. That is enough to validate a local rollout against the original model's −2.14 / +1.26 / +1.08 / −0.73 on sel1.

Before writing the rollout engine I need one more piece: the state at a stay's first bundle. My core check skipped first bundles. I'll test h = 0 and out = 0 with several assumed gaps (dt) against the tool's states for first bundles. Exactly one choice should give float-level error.

```python
first_idx=np.where(np.r_[True, B_T.stay_id.values[1:]!=B_T.stay_id.values[:-1]])[0]
out_t=torch.as_tensor(SxA[:,:192]); h_t=torch.as_tensor(SxA[:,192:960]).reshape(-1,4,192); z_t=torch.as_tensor(SxA[:,960:])
def first_step(dt_s, ctx_mode='full'):
    n=len(first_idx); h0=torch.zeros(n,4,192); o0=torch.zeros(n,192)
    dt=torch.full((n,),float(dt_s))
    hf=h0
    ctx=o0+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt)[:,None]/10,W,'gap')
    if ctx_mode=='zero': ctx=torch.zeros(n,192)
    if ctx_mode=='nogap': ctx=o0+LN(hf.mean(1),W,'core.out_norm')
    e=E_T[first_idx]; u=U_T[first_idx]
    z=mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]
    X=torch.cat([hf+W['core.slots'][None],e[:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    X=layer_sa(X,W,'core.prelude.0'); Ep=LN(X[:,:4],W,'core.prelude_norm')
    M=e[:,None]; mask=torch.ones(n,1,dtype=torch.bool); agee=mlp(torch.log1p(torch.zeros(n,1))[...,None],W,'core.age')
    x=hf+Ep
    for l in ['core.blocks.0.0','core.blocks.0.1']: x=layer_blk(x,W,l,M,mask,agee,'inadd')
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    hn=abar*hf+sp*F.linear(Ep,W['core.b_in.weight'])+(x-(hf+Ep))
    y=layer_sa(F.linear(hn,W['core.c_out.weight']),W,'core.coda.0'); on=LN(y.mean(1),W,'core.out_norm')
    return (z-z_t[first_idx]).abs().mean().item(), (on-out_t[first_idx]).abs().mean().item(), (y-h_t[first_idx]).abs().mean().item()
for dts in [1.0, 60.0, 3600.0, 7200.0, 86400.0]:
    print(dts, [round(v,5) for v in first_step(dts)])
print('zero ctx', first_step(1.0,'zero')); print('nogap', first_step(1.0,'nogap'))
print('first bundle times', B_T.t.values[first_idx][:5])
```
Unit mapping and END-atom attributes barely matter: every combination leaves about 0.04 on action bundles and 0.053 on observation bundles. So the remaining error is something shared by all bundles, most likely in how ctx is built or how the posterior mean is formed. I will test variants that each change one piece:
- drop the time flow;
- drop the out_norm(h) term;
- drop the gap term;
- add the prior mean to the posterior output;
- swap the order of the z_post inputs.

I also print the range of exp(core.flow.a), to check whether it matches the stated decay range of 1/(2 years) to 1/(10 min). A variant that brings the error to about 1e-3 identifies the bug. Variants that make it much worse tell me which pieces I already have right.

```python
fa = Wo['core.flow.a']; print('exp(flow.a) range', float(torch.exp(fa).min()), float(torch.exp(fa).max()), '1/tau range', 1/63115200, 1/600)
UB = dict(unit={'map_cuff':2,'map_arterial':2,'glucose_finger':5,'glucose_lab':5,'insulin':4,'norepinephrine':3}, ru={'insulin':3,'norepinephrine':2}, end_attrs=True)
def zerr2(o, var):
    bb=build_bundles(ev0,o); errs=[]; kinds=[]; per=[]
    for k in range(1,len(bb)):
        out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
        dt=(bb[k][0]-bb[k-1][0])*3600
        hf=h_ if var=='noflow' else h_*torch.exp(-dt*torch.exp(fa))
        ctx=out_.clone()
        if var!='nonorm': ctx=ctx+LN(hf.mean(0),'core.out_norm')
        if var!='nogap': ctx=ctx+MLP(torch.tensor([math.log(dt)/10]),'gap','gelu')
        at=bb[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
        e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
        u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
        inp=torch.cat([ctx,u_,e_]) if var=='swap' else torch.cat([ctx,e_,u_])
        zp=MLP(inp,'z_post','gelu')[:32]
        if var=='addprior': zp=zp+MLP(ctx,'z_prior','gelu')[:32]
        d_=zp-torch.tensor(X0[k,960:]); per.append(d_.numpy())
        errs.append(float(d_.abs().mean())); kinds.append(('A' if ac else '')+('O' if ob else ''))
    errs=np.array(errs); kinds=np.array(kinds)
    return {kk: round(float(np.median(errs[kinds==kk])),3) for kk in ['A','O'] if (kinds==kk).any()}, np.array(per)
for var in ['base','noflow','nonorm','nogap','addprior','swap']:
    r_,per_=zerr2(UB,var); print(var, r_)
r_,per_=zerr2(UB,'base'); print('per-dim mean abs err', np.round(np.abs(per_).mean(0),3))
```
Combining prior and posterior by precision does not help (0.062 against 0.064). The worst dimensions, 9 and 20, have a log-sd near 0 in both prior and posterior. These look like unused latent dimensions, so their errors matter little in practice, but they still show my computation is slightly off. It is also suspicious that changing the unit embeddings, rows with norms of 0.7–1.4, barely changed the error.

Next I test how sensitive z is to each input, by zeroing e_k, zeroing u_k or dropping the kq term. If zeroing e_k barely changes the error, z_post depends weakly on the observation encoding, and the residual must come from ctx. If dropping kq lowers the error, kq is not part of the atom embedding.

```python
def zerr3(o, var):
    bb=build_bundles(ev0,o); errs=[]; kinds=[]
    for k in range(1,len(bb)):
        out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
        dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
        ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10]),'gap','gelu')
        at=bb[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
        def em(a):
            v=emb(a,o)
            return v-Wo['embed.kq.weight'][0] if var=='nokq' else v
        e_=enc(torch.stack([em(a) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
        u_=enc(torch.stack([em(a) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
        if var=='e0': e_=torch.zeros(192)
        if var=='u0': u_=torch.zeros(192)
        if var=='ctxout': ctx=out_
        zp=MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32]
        errs.append(float((zp-torch.tensor(X0[k,960:])).abs().mean())); kinds.append(('A' if ac else '')+('O' if ob else ''))
    errs=np.array(errs); kinds=np.array(kinds)
    return {kk: round(float(np.median(errs[kinds==kk])),3) for kk in ['A','O'] if (kinds==kk).any()}
for var in ['base','nokq','e0','u0']:
    print(var, zerr3(UB,var))
```
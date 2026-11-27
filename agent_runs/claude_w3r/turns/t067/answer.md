Neither version reproduces the split (dimension 9 at 0.5 u/h stays at +0.24 to +0.28), and the overall error stays at about 0.05. A pattern holds across every variant: my computed dimensions 9 and 20 follow the right ordering but are weaker. That would fit u_k being right in direction but wrong in scale or normalisation, for example a missing or different final LayerNorm.

As a last cheap check on the input side, I scale u_k by 2, drop its final LayerNorm, or use a LayerNorm without affine. If one of these sharpens the split to about ±1.1 and lowers the overall error, the bug is in the action encoder's output normalisation. If none does, I stop chasing exact z. I will then test the core update using the stored z, which matters more for designing the edit.

```python
def run_var3(var, sids, o=UB):
    out=[]
    for sid_ in sids:
        mm=S_['stay_id']==sid_; Xs_=S_['x'][mm]; bb=build_bundles(tr[tr.stay_id==sid_].sort_values('hour'),o)
        for k in range(1,len(bb)):
            out_=torch.tensor(Xs_[k-1,:192]); h_=torch.tensor(Xs_[k-1,192:960]).reshape(4,192)
            dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
            ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
            at=bb[k][1]; isend=lambda a:a['kind']==2 and a['ek']==2
            ob=[a for a in at if a['ek']!=2 or isend(a)]; ac=[a for a in at if a['ek']==2 and not isend(a)]
            e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
            if ac:
                X_=torch.stack([emb(a,o) for a in ac]); nn_=X_.shape[0]
                s=MLP(X_,'encode_action.mlp1','gelu').sum(0)/math.sqrt(nn_)+L_(torch.tensor([math.log(1+nn_)],dtype=torch.float32),'encode_action.size')
                m2=MLP(s,'encode_action.mlp2','gelu')
                u_={'x2':2*LN(m2,'encode_action.norm'),'noln':m2,'lnnoaff':Fnn.layer_norm(m2,(192,))}[var]
            else: u_=torch.zeros(192)
            zp=MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32].numpy()
            ir=[a['rate'][1] for a in at if a['code']==8 and a['kind']==0]
            out.append((ir[0] if ir else np.nan, zp, Xs_[k,960:]))
    rr=np.array([t[0] for t in out]); zp_=np.array([t[1] for t in out]); za_=np.array([t[2] for t in out])
    return round(float(np.median(np.abs(zp_-za_).mean(1))),4), {r0: round(float(zp_[np.isclose(rr,r0)][:,9].mean()),2) for r0 in [0.5,1.0,1.5,2.0,2.5]}
for var in ['x2','noln','lnnoaff']:
    print(var, run_var3(var, sids6))
```
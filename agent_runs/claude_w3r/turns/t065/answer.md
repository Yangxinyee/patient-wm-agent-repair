Even the best bin pair leaves dimensions 9 and 20 positive (0.75–0.87), while the model has −1.1. So in my computation the insulin start atom does not control these dimensions strongly enough. The model's u_k must respond to the insulin rate more sharply than mine, which points to how the action set is formed or normalised, not to the atom embedding.

Next I compare several set-encoder variants over all bundles of 6 stays:
- END atoms in the observation encoder, in the action encoder, or in both;
- the size term counting all atoms in the bundle;
- mean pooling instead of sum/√n.

For each I report the overall median z error and the rate pattern in dimensions 9 and 20. A variant that gives about −1.1 at 0.5 and 2.0 u/h, and about +1.08 at 1.0 and 1.5 u/h, with low overall error would identify the right construction.

```python
def enc_v(X, n_name, nsize=None, pool='sqrt'):
    nn_=X.shape[0]; ns=nsize if nsize is not None else nn_
    h1=MLP(X,n_name+'.mlp1','gelu')
    s=(h1.sum(0)/math.sqrt(nn_) if pool=='sqrt' else h1.mean(0) if pool=='mean' else h1.sum(0))+L_(torch.tensor([math.log(1+ns)],dtype=torch.float32),n_name+'.size')
    return LN(MLP(s,n_name+'.mlp2','gelu'),n_name+'.norm')
def run_var(var, sids, o=UB):
    out=[]
    for sid_ in sids:
        mm=S_['stay_id']==sid_; Xs_=S_['x'][mm]; bb=build_bundles(tr[tr.stay_id==sid_].sort_values('hour'),o)
        for k in range(1,len(bb)):
            out_=torch.tensor(Xs_[k-1,:192]); h_=torch.tensor(Xs_[k-1,192:960]).reshape(4,192)
            dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
            ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
            at=bb[k][1]; isend=lambda a:a['kind']==2 and a['ek']==2
            if var['end']=='obs': ob=[a for a in at if a['ek']!=2 or isend(a)]; ac=[a for a in at if a['ek']==2 and not isend(a)]
            elif var['end']=='act': ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
            else: ob=[a for a in at if a['ek']!=2 or isend(a)]; ac=[a for a in at if a['ek']==2]
            ns_e = len(at) if var.get('nall') else None
            e_=enc_v(torch.stack([emb(a,o) for a in ob]),'encode',ns_e,var.get('pool','sqrt')) if ob else torch.zeros(192)
            u_=enc_v(torch.stack([emb(a,o) for a in ac]),'encode_action',ns_e,var.get('pool','sqrt')) if ac else torch.zeros(192)
            zp=MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32].numpy()
            ir=[a['rate'][1] for a in at if a['code']==8 and a['kind']==0]
            out.append((ir[0] if ir else np.nan, zp, Xs_[k,960:]))
    rr=np.array([t[0] for t in out]); zp_=np.array([t[1] for t in out]); za_=np.array([t[2] for t in out])
    pat={r0: tuple(np.round(zp_[np.isclose(rr,r0)][:,9].mean(),2) for _ in [0]) for r0 in [0.5,1.0,2.0]}
    return round(float(np.median(np.abs(zp_-za_).mean(1))),4), pat
sids6=np.unique(S_['stay_id'])[:6]
for var in [dict(end='obs'),dict(end='act'),dict(end='both'),dict(end='obs',nall=True),dict(end='obs',pool='mean'),dict(end='obs',pool='sum')]:
    print(var, run_var(var, sids6))
```
Changing the rate and dose bins of the 0.5 u/h atom does not help at all: the best pair still leaves 0.176. So binning is not the cause. Bundles 9 and 11 also differ from the well-matched bundles in their END atoms (for example, an insulin restart after a gap). The large error might therefore come from how END atoms are encoded, not from the start atom.

Next I test, over all action bundles, four alternatives:
- END atoms without rate, dose or units;
- END atoms passed through the observation encoder;
- END atoms left out;
- END atoms keeping only their units.

If one of these brings bundles 9 and 11 down to about 0.03 without hurting the others, END-atom handling is the bug.

```python
def zk_var(k, var, o=UB):
    out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
    dt=(bbB[k][0]-bbB[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
    ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    at=[dict(a) for a in bbB[k][1]]
    for a in at:
        if a['kind']==2 and a['ek']==2:
            if var in ('noattr','unitonly'): a['dose']=None; a['rate']=None
            if var=='noattr': a['unit']=0; a['ru']=0
    ob=[a for a in at if a['ek']!=2 or (var=='endobs' and a['kind']==2)]
    ac=[a for a in at if a['ek']==2 and not (var in ('endobs','noend') and a['kind']==2)]
    e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
    u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
    return MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32]
Aks=[k for k in range(1,len(bbB)) if any(a['ek']==2 for a in bbB[k][1])]
for var in ['base','noattr','unitonly','endobs','noend']:
    errs_=[round(float((zk_var(k,var)-torch.tensor(X0[k,960:])).abs().mean()),3) for k in Aks]
    print(var, 'median', np.median(errs_), 'k9,k11', errs_[Aks.index(9)], errs_[Aks.index(11)], 'max', max(errs_))
```
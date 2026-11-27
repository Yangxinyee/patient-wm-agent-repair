The run failed only because the numpy grid values became float64 tensors and hit float32 weights. I will cast the gap input to float32 and run the same per-bundle grid search over the gap input. The test is unchanged: an error near 0 at a consistent best input pins down the gap scaling, while an error that stays near 0.04 rules the gap term out.

```python
def zk(k, gin, o=UB, hflow=True):
    out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
    dt=(bbB[k][0]-bbB[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa)) if hflow else h_
    ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([float(gin)],dtype=torch.float32),'gap','gelu')
    at=bbB[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
    e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
    u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
    return MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32], dt
for k in range(1,16):
    zt=torch.tensor(X0[k,960:])
    errs_=[float((zk(k,g)[0]-zt).abs().mean()) for g in grid]
    j=int(np.argmin(errs_)); dt=zk(k,0)[1]
    print(k, 'dt',dt, 'l10',round(math.log(dt)/10,3), 'best',round(grid[j],2), 'minerr',round(errs_[j],4), 'err@l10', round(float((zk(k,math.log(dt)/10)[0]-zt).abs().mean()),4), 'noflow', round(float((zk(k,math.log(dt)/10,hflow=False)[0]-zt).abs().mean()),4))
```
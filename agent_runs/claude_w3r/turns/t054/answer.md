The identity holds to 3e-8: the stored h is the post-coda state, and out = out_norm(mean of the slots). So ctx is built from the right h, and the remaining z error must lie in the gap term or in z_post's inputs.

Next, for each bundle, I grid-search the scalar fed into the gap MLP to minimise the z error.
- If the error then falls to about 0 and the best input is a clean function of dt, I have found the correct gap scaling.
- If the error stays near 0.04 even with the best input, the gap term is not the problem.

```python
bbB = build_bundles(ev0, UB)
def zk(k, gin, o=UB, hflow=True):
    out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
    dt=(bbB[k][0]-bbB[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa)) if hflow else h_
    ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([gin]),'gap','gelu')
    at=bbB[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
    e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
    u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
    return MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32], dt
grid = np.linspace(-1.5,2.5,81)
for k in range(1,16):
    errs_=[float((zk(k,g)[0]-torch.tensor(X0[k,960:])).abs().mean()) for g in grid]
    j=int(np.argmin(errs_)); dt=zk(k,0)[1]
    print(k, 'dt',dt, 'log(dt)/10',round(math.log(dt)/10,3), 'best gin',round(grid[j],2), 'min err',round(errs_[j],4), 'err at l10', round(float((zk(k,math.log(dt)/10)[0]-torch.tensor(X0[k,960:])).abs().mean()),4))
```
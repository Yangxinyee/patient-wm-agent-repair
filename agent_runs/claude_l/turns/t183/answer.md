Zeroing out a z dimension's push changes nothing, at any lag, for either outcome, even though z itself moves a lot when the dose doubles (|Δz| ≈ 2.3). This would happen if `z_in.weight` is zero, i.e. the posterior z is a dead channel the model never reads. If so, z_post and z_in are free to repurpose: z_post sees [ctx, e, u], so it can compute a dose residual (dose minus what the state predicts), and z_in can feed that into the prelude as a token without disturbing anything else. Two things decide whether this can produce more than a lag-1 effect: (1) confirm `z_in.weight` ≈ 0 and see how large `z_head` is; (2) check whether perturbations written into h after a drug bundle persist past the next reading. If h perturbations still move glucose predictions at lags 2–12 in some directions, a one-shot injection can carry the lagged effects. If they die after lag 1 like the u-token effect, the memory has to be maintained at every step.

```python
print('z_in.weight norm', float(W0['z_in.weight'].norm()), 'z_in.bias norm', float(W0['z_in.bias'].norm()))
print('z_head.weight norm', float(W0['z_head.weight'].norm()), 'z_post.2 norm', float(W0['z_post.2.weight'].norm()))
_old_bstep_mod = bstep_mod
def bstep_mod(W,out,h,dt,e,u,Mt,ages,m):
    o_,y_,z_=_old_bstep_mod(W,out,h,dt,e,u,Mt,ages,m)
    if 'hdelta' in m:
        y_=y_+m['hdelta']; o_=bln(W,y_.mean(1),'core.out_norm')
    return o_,y_,z_
gen_=torch.Generator().manual_seed(5); RH=[]
for s in ks[:8]:
    B_=BUN[s]; ib=[k for k in ins_bundles(B_) if 6<=B_[k][0]<=10]
    if not ib: continue
    k=ib[0]; base=tf_run(W0,B_); t0_=B_[k][0]; hn=float(torch.tensor(Xs[row_start[s]+k,192:960]).reshape(4,192).norm(dim=1).mean())
    for j in range(8):
        d_=torch.randn(1,1,192,generator=gen_,dtype=T); d_=d_/d_.norm()*0.05*hn; d_=d_.repeat(1,4,1)
        p2=tf_run(W0,B_,{k:dict(hdelta=d_)})
        RH.append([p2.get((round(t0_-0.0833+L,3),6),np.nan)-base.get((round(t0_-0.0833+L,3),6),np.nan) for L in range(1,13)])
RH=np.array(RH); print('h-perturb glucose resp (abs mean) lag1..12', np.round(np.nanmean(np.abs(RH),0),3))
print('ratio lag2/lag1', round(float(np.nanmean(np.abs(RH[:,1]))/np.nanmean(np.abs(RH[:,0]))),3), 'lag4/lag1', round(float(np.nanmean(np.abs(RH[:,3]))/np.nanmean(np.abs(RH[:,0]))),3))
```
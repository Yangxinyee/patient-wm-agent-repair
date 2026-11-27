Teacher-forced, the original model responds to +1 U/h of insulin only at lag 1 (−0.62 mg/dL), with essentially nothing at lags 2–12. The true direct coefficients from the records are −1.96, −1.32, −0.52, then positive from lag 4, and they sum to about zero. All of the model's dose effect goes through the u token in the prelude; routing the changed u through z_post has no effect at all. Two questions follow. Does z_post depend on u at all? And can a perturbation of z at a drug bundle shift future predictions, beyond lag 1, without being wiped out by the next reading? If some z directions give lasting, glucose-specific (or MAP-specific) responses, z can serve as the channel for a dose-residual signal that z_post computes from u and ctx. I'll measure |Δz| from the dose change, plus the response of glucose_finger and map_arterial at lags 1–12 to a unit push on each of the 32 z dims.

```python
def u_enc(atoms_u): return set_enc(W0,[fix_atom2(a) for a in atoms_u],'encode_action')[None]
dz=[]; RZg=[]; RZm=[]; zsd=Xs[:,960:992].std(0)
for s in ks[:12]:
    B_=BUN[s]; ib=[k for k in ins_bundles(B_) if 6<=B_[k][0]<=10]
    if not ib: continue
    k=ib[0]; base=tf_run(W0,B_); t0_=B_[k][0]
    # dz/du
    i0=row_start[s]; prev=torch.tensor(Xs[i0+k-1],dtype=T); dt=(B_[k][0]-B_[k-1][0])*3600.0
    hf=prev[192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa0)); ctx=prev[:192]+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
    ea,ua=split_fix(B_[k][1]); e=set_enc(W0,ea,'encode') if ea else set_enc(W0,[],'encode')
    ua0=[a for a in B_[k][1] if a['act'] and a['kind']==0]
    ua1=[edit_atom(a,2.0) if a['code']==8 else a for a in ua0]
    z0=mlp(W0,torch.cat([ctx,e,u_enc(ua0)[0]]),'z_post')[:32]; z1=mlp(W0,torch.cat([ctx,e,u_enc(ua1)[0]]),'z_post')[:32]
    dz.append(float((z1-z0).norm()))
    rg=[]; rm=[]
    for i in range(32):
        d_=torch.zeros(1,32,dtype=T); d_[0,i]=float(zsd[i])
        p2=tf_run(W0,B_,{k:dict(zdelta=d_)})
        rg.append([p2.get((round(t0_-0.0833+L,3),6),np.nan)-base.get((round(t0_-0.0833+L,3),6),np.nan) for L in range(1,13)])
        rm.append([p2.get((round(t0_-0.0833+L,3),7),np.nan)-base.get((round(t0_-0.0833+L,3),7),np.nan) for L in range(1,13)])
    RZg.append(rg); RZm.append(rm)
RZg=np.array(RZg); RZm=np.array(RZm)
print('|dz| for insulin x2:', np.round(dz,4), ' z sd range', np.round(zsd.min(),3), np.round(zsd.max(),3))
mg=np.nanmean(RZg,0); mm=np.nanmean(RZm,0)
print('per-z-dim (1 sd) glucose resp lag1..4 & sum, top 8 by |lag1|:')
o=np.argsort(-np.abs(mg[:,0]))[:8]
for i in o: print(i, np.round(mg[i,:4],3), round(float(np.nansum(mg[i])),3), '| MAP lag1', round(float(mm[i,0]),3))
print('max |lag>=2| glucose resp over dims', np.round(np.abs(mg[:,1:]).max(),3))
```
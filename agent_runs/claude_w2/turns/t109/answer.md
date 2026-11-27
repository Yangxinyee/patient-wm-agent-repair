The new atom builder reproduces the stored atom table exactly: 14,742 atoms, same order and fields. Next I check the embedding against the stored `xa` and the z-posterior mean against the recorded states. For each non-first bundle, the context is built from the previous recorded state (out, time-flowed h, gap(log dt/10)), and the posterior mean comes from [ctx, e, u]. I try GELU and SiLU for the gap MLP and dt in seconds, and report the relative error separately for reading, action-only and end-only bundles. A small error (about 0.01 or less) across bundle types confirms the encoder and context. A large error for one type points to the specific routing or gap detail to fix.

```python
B0,E0,U0,hasu0,xa0=encode_bundles(A0,WO)
print('xa match', float((xa0-torch.as_tensor(np.asarray(xa.numpy() if hasattr(xa,'numpy') else xa))).abs().max()))
Xs=np.asarray(X_.numpy() if hasattr(X_,'numpy') else X_)
sidX=np.asarray(sidS); 
# order check: states rows grouped by stay in same within-stay order; build index into B0
B0['row']=np.arange(len(B0))
B0['rank']=B0.groupby('stay_id').cumcount()
rk=pd.Series(sidX).groupby(sidX).cumcount().values
mp=B0.set_index(['stay_id','rank']).row
rowB=mp.loc[list(zip(sidX,rk))].values
print('nb', len(B0), len(Xs), 'stay order same?', (B0.stay_id.values[rowB]==sidX).all())
OUTs=torch.as_tensor(Xs[:,:192]); Hs_=torch.as_tensor(Xs[:,192:960].reshape(-1,4,192)); Zs=torch.as_tensor(Xs[:,960:992])
# previous within stay
prev=np.r_[-1,np.arange(len(sidX)-1)]; first=np.r_[True,sidX[1:]!=sidX[:-1]]; prev[first]=-1
nf=np.where(~first)[0]; pv=prev[nf]
mins=B0.minute.values[rowB]
dts=torch.as_tensor(((mins[nf]-mins[pv])*60.0).astype(np.float32))
def zpost_mean(ctx,e,u,W): return mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]
Eo=E0[rowB]; Uo=U0[rowB]
kind_tab=A0.groupby(['stay_id','bidx']).apply(lambda d: 'gf' if (d.code==6).any() else ('gl' if (d.code==10).any() else ('meas' if (d.ekind==1).any() else ('act' if ((d.kind==0)&(d.ekind==2)).any() else ('end' if (d.kind==2).all() else 'other'))))).values
kt=kind_tab[rowB][nf]
for gact in ['gelu','silu']:
    hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
    g_=mlp(torch.log(dts)[:,None]/10,WO,'gap',gact)
    ctx_=OUTs[pv]+LN(hf_.mean(1),WO,'core.out_norm')+g_
    zm=zpost_mean(ctx_,Eo[nf],Uo[nf],WO)
    err=(zm-Zs[nf]).abs().mean(1)/Zs[nf].abs().mean(1)
    print(gact, {k_:round(float(err[torch.as_tensor(kt==k_)].mean()),4) for k_ in np.unique(kt)}, 'all',round(float(err.mean()),4))
```
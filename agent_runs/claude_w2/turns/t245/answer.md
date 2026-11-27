The unedited baseline on these 100 train stays reproduces the known gap. Its predicted effects are only 15–26% of the kernel truth:

| Arm | Model | Truth |
|---|---|---|
| x2 | −3.5 | −16.5 |
| x½ | +2.2 | +8.3 |
| x0 | +2.5 | +16.5 |
| x3 | −4.9 | −33.0 |
| x2 from hour 16 | −2.6 | −14.2 |

The linear-oracle run crashed only because my stored policy function `pfun` expects a pandas Series. I'll build a numpy step function of glucose from the training grid, using the band base rate that the residual regression used (`ins_base` against current glucose), and rerun the same test. The confirm/refute criteria are unchanged.

```python
t0=time.time()
gq=gg[['glu_now','ins_base']].dropna()
edges=np.arange(100,420,5.0); idx=np.clip(np.searchsorted(edges,gq.glu_now.values,side='right')-1,0,len(edges)-1)
tab=pd.Series(gq.ins_base.values).groupby(idx).mean().reindex(range(len(edges))).ffill().fillna(0).values
def pf_np(g):
    g=np.asarray(g,float); i=np.clip(np.searchsorted(edges,g,side='right')-1,0,len(edges)-1)
    return np.where(g<edges[0],0.0,tab[i])
print('pf', [round(float(pf_np(x)),2) for x in [140,155,175,190,230,260,290,310]])
hk_i=make_lin_oracle(Rx,DGi=DGi_,a_i=a_i,c_i=c_i,k_i=0.164,pf=pf_np)
res_l=run_rollout_h2(Rx,W,arms_i,n_paths=10,seed=1,hook_P=hk_i)
for x in eval_arms(res_l,arms_i,Rx,Kins): print('LIN', x)
hk_n=make_lin_oracle(Rx,DGn=DGn_,a_n=a_n,k_n=0.303)
res_n=run_rollout_h2(Rx,W,arms_n,n_paths=10,seed=1,hook_P=hk_n)
for x in eval_arms(res_n,arms_n,Rx,Knor,drug='norepinephrine',outcome='map_arterial'): print('LIN-NE', x)
res_nb=run_rollout_h2(Rx,W,arms_n,n_paths=10,seed=1)
print('BASE-NE', [x[:3] for x in eval_arms(res_nb,arms_n,Rx,Knor,drug='norepinephrine',outcome='map_arterial')], round(time.time()-t0))
```
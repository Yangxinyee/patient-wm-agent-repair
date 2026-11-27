Still no match, and the best error stays near 1.46 whatever I change. That suggests a systematic difference, not a wrong option among the ones I am trying.

Next I look at how my computed z relates to the stored z, dimension by dimension, across the k=1 bundles of all 100 stays: the per-dimension correlation, slope and error. If most dimensions correlate near 1 with slope 1, a few specific inputs or dimensions are off. If the correlations are high but the slopes or offsets differ, the stored z is a transformed or combined quantity, for example prior plus posterior. If correlations are low, the inputs to z_post (e, u or ctx) are badly wrong.

```python
ACTMAP.clear()
for nm in ['encode.mlp1','encode.mlp2','encode_action.mlp1','encode_action.mlp2','gap','z_post']: ACTMAP[nm]='silu'
EU=[enc_bundle(BUN[s][1][1]) for s in stays]
Z=[];C=[]
for i,s in enumerate(stays):
    r0=Xt[row_of[s][0]]; out=r0[:192]; h=r0[192:960].reshape(4,192)
    dt=(BUN[s][1][0]-BUN[s][0][0])*3600
    hf=h*torch.exp(-dt*torch.exp(W0['core.flow.a']))
    ctx=out+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
    C.append(ctx); Z.append(mlp(W0,torch.cat([ctx,EU[i][0],EU[i][1]]),'z_post'))
Z=torch.stack(Z).numpy(); Zt=np.stack([X[row_of[s][1],960:] for s in stays])
cors=[np.corrcoef(Z[:,j],Zt[:,j])[0,1] for j in range(32)]
slopes=[np.polyfit(Z[:,j],Zt[:,j],1)[0] for j in range(32)]
print('corr', np.round(cors,3)); print('slope', np.round(slopes,2))
print('per-dim max err', np.round(np.abs(Z[:,:32]-Zt).max(0),2))
cors2=[max(abs(np.corrcoef(Z[:,32+j],Zt[:,j])[0,1]),0) for j in range(32)]; print('corr with 2nd half', np.round(cors2,2))
print('dt values k=1:', sorted(set([(BUN[s][1][0]-BUN[s][0][0]) for s in stays])))
```
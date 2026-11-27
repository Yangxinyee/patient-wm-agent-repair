Forward pass 3 returned the probe states. Their z columns are exact projections of the model's true z_post inputs at every bundle: columns 0–9 project ctx, 10–20 project e, 21–31 project u.

Next I compare these with my own computations, using SiLU and then GELU for the encoder MLPs:
- my encodings of each bundle's measurement atoms (e) and action atoms (u);
- my ctx, built from the probe model's own state after the previous bundle with the documented formula.

A near-zero error for e and u under one activation confirms the encoders. A failing ctx then points to the context formula. A failing e or u, broken down by atom type, locates the encoder bug.

```python
pz=np.load('/work/fwd_3/states_edited.npz'); Xp=pz['x'].astype(np.float64)
assert (pz['stay_id']==sid_arr).all()
P=np.load('/work/probe_P.npy')
Pc,Pe,Pu=P[0:10,0:192],P[10:21,192:384],P[21:32,384:576]
res={}
for ea in ['silu','gelu']:
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea,'encode_action.mlp1':ea,'encode_action.mlp2':ea,'gap':ea})
    ee=[];eu=[];ec=[]
    for s in stays[:40]:
        rows=row_of[s]; bl=BUN[s]
        for k in range(len(bl)):
            e,u=enc_bundle(bl[k][1])
            zt=Xp[rows[k],960:]
            ee.append(np.abs(Pe@e.numpy()-zt[10:21]).max()); eu.append(np.abs(Pu@u.numpy()-zt[21:32]).max())
            if k>0:
                prev=torch.tensor(Xp[rows[k-1]]); out=prev[:192]; h=prev[192:960].reshape(4,192)
                dt=(bl[k][0]-bl[k-1][0])*3600
                hf=h*torch.exp(-dt*torch.exp(W0['core.flow.a']))
                c=out+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
                ec.append(np.abs(Pc@c.numpy()-zt[0:10]).max())
    print(ea,'e err q50/max %.2e %.2e'%(np.median(ee),max(ee)),'| u err %.2e %.2e'%(np.median(eu),max(eu)),'| ctx err %.2e %.2e'%(np.median(ec),max(ec)))
```
The leftover variables show that earlier I searched many combinations (activation, ctx variant, input order, output offset) without finding a general match. The exact matches in run 66 were all at k=1, the bundle right after the shared hospital-admission bundle, where `h0`/`out0` are the same for every stay. Even there, stays with glucose_finger atoms in high bins failed. So two problems may be mixed up: a wrong component in ctx/z for later steps, and possibly a value-binning issue for some glucose atoms.

Next I repeat the activation grid at k=1 for all 100 stays, and score the fraction of stays matched exactly as well as the median error. A combination with many exact zeros shows which activations reproduced run 66. I can then compare a k=1 bundle that matches with one that doesn't to find the remaining difference.

```python
stays=list(BUN.keys())
def enc_bundle(atoms):
    meas=[a for a in atoms if not a['act']]; act=[a for a in atoms if a['act']]
    e=set_enc(W0,meas,'encode') if meas else torch.zeros(192,dtype=T)
    u=set_enc(W0,act,'encode_action') if act else torch.zeros(192,dtype=T)
    return e,u
res=[]
for a1 in names_a:
  for a2 in names_a:
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':a1,'encode.mlp2':a2,'encode_action.mlp1':a1,'encode_action.mlp2':a2})
    EU=[enc_bundle(BUN[s][1][1]) for s in stays]
    for ga in names_a:
      ACTMAP['gap']=ga
      C=[]
      for s in stays:
        r0=X[row_of[s][0]]; out=torch.tensor(r0[:192],dtype=T); h=torch.tensor(r0[192:960],dtype=T).reshape(4,192)
        dt=(BUN[s][1][0]-BUN[s][0][0])*3600
        hf=h*torch.exp(-dt*torch.exp(W0['core.flow.a']))
        C.append(out+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap'))
      for za in names_a:
        ACTMAP['z_post']=za
        er=np.array([(mlp(W0,torch.cat([C[i],EU[i][0],EU[i][1]]),'z_post')[:32]-torch.tensor(X[row_of[s][1],960:],dtype=T)).abs().max().item() for i,s in enumerate(stays)])
        res.append((np.mean(er<1e-4),np.median(er),a1,a2,ga,za))
res.sort(key=lambda r:(-r[0],r[1]))
for r in res[:6]: print(r)
```
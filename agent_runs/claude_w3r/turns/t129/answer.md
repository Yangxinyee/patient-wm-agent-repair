No hidden unit in either encode_action MLP is dead (none stays below −3 on any input). An encoder-level channel would therefore have to take over live units. A cleaner route is to read the insulin rate the encoder already writes into u, which predicted rate with R² 0.96, at the prelude: head 2 attends 100% to u, and only the slot outputs of the prelude are used. A rank-1 change to head 2's value and output weights could then write the decoded rate into the slot tokens.

That relies on my local u being exactly right. Matching `out` does not test that, because u barely affects `out`. The posterior z does depend on u at action bundles. So next I will check how well my z_post([ctx, e, u]) reproduces the model's posterior z at insulin bundles, including the binary dims 9 and 20.

- **Confirms:** the error at insulin bundles is small, in particular on dims 9 and 20. Then u can be trusted for designing a linear rate readout.
- **Refutes:** dims 9/20 are still off. Then my u is wrong and must be fixed first.

```python
zerr=[]; z9=[]
for r_ in range(1):
    pass
for s in selS[:15]:
    ev_=recs[recs.stay_id==s].sort_values('hour'); B_=build_bundles(ev_,OPTS); rw=np.where(SST==s)[0]
    if len(B_)!=len(rw): continue
    EU_=[enc_eu(al) for _,al in B_]
    for k in range(1,len(B_)):
        ia=[a for a in B_[k][1] if a['code']==CODE['insulin'] and a['kind']==0]
        if not ia: continue
        op=torch.tensor(SX[rw[k-1],:192],dtype=torch.float64); hp=torch.tensor(SX[rw[k-1],192:960].reshape(4,192),dtype=torch.float64)
        dt=(B_[k][0]-B_[k-1][0])*3600; decay=torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx_=op+LNt((hp*decay).mean(0),'core.out_norm')+mlp_t(torch.tensor([[math.log(max(dt,1.0))/10.0]],dtype=torch.float64),'gap')[0]
        zo=mlp_t(torch.cat([ctx_,EU_[k][0],EU_[k][1]]),'z_post')[:32]; zt=SX[rw[k],960:]
        zerr.append(np.abs(zo.numpy()-zt)); z9.append((ia[0]['rate'], float(zo[9]), float(zt[9]), float(zo[20]), float(zt[20])))
zerr=np.array(zerr); print('n',len(zerr),' median abs err all dims %.4f'%np.median(zerr), ' per-dim median (top 6 worst):', np.argsort(-np.median(zerr,0))[:6], np.sort(np.median(zerr,0))[::-1][:6].round(3))
z9=np.array(z9); 
for rv in np.unique(z9[:,0])[:10]:
    m_=z9[:,0]==rv; print('rate',rv,'n',m_.sum(),' z9 mine %.2f true %.2f | z20 mine %.2f true %.2f'%tuple(z9[m_,1:].mean(0)))
```
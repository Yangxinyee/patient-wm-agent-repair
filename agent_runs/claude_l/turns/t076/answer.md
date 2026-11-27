No variant matches. Even `z` has an error of 2.1, yet my earlier `z_pred_k` reproduced the tool's z almost exactly. So the mistake is in how I set up the step, not in the memory or prelude variants. Likely causes: the row alignment of `X`, the ctx formula, or the dt.

Next I will call the old `z_pred_k` for a few bundles and compare it both with X and with my new z computation. The difference will show what is off: an index shift, a gap scaling, or how `h` after the time flow enters ctx. If `z_pred_k` matches X at row k while my code matches it only at another row, that is an alignment problem.

```python
s=2020000000; rows=row_of[s]; bl=BUN[s]
for k in [1,2,3,5,10]:
    try:
        zo = z_pred_k(s,k,F.silu)
        zo = zo.numpy() if hasattr(zo,'numpy') else np.asarray(zo)
    except Exception as ex:
        print('ERR',ex); zo=None
    tgt=X[rows[k],960:]
    # my z
    prev=torch.tensor(X[rows[k-1]],dtype=T); out=prev[:192]; h=prev[192:960].reshape(4,192)
    dt=(bl[k][0]-bl[k-1][0])*3600
    meas=[a for a in bl[k][1] if not a['act']]; act=[a for a in bl[k][1] if a['act']]
    e=set_enc(W0,meas,'encode'); u=set_enc(W0,act,'encode_action') if act else torch.zeros(192,dtype=T)
    hf=h*torch.exp(-dt*torch.exp(W0['core.flow.a'])); g=mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
    ctx=out+lnorm(W0,hf.mean(0),'core.out_norm')+g
    zm=mlp(W0,torch.cat([ctx,e,u]),'z_post')[:32].numpy()
    print(k, 'old vs X', None if zo is None else (np.shape(zo), np.abs(np.ravel(zo)[:32]-tgt).max()), '| new vs X', np.abs(zm-tgt).max(), '| new vs old', None if zo is None else np.abs(np.ravel(zo)[:32]-zm).max())
dis.dis(z_pred_k)
```
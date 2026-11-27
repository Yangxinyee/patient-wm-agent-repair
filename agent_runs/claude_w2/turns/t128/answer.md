fa4 is the per-reading factual table from fwd_4 on these stays. It holds the tool's predicted mean and NLL, the index of the previous state row, dt, the last-register fields, and my older head predictions. Those were off by up to 13 mg/dL on some rows, so the head was not yet matched at that point. Columns such as "sh_*" and "partner_same" show I was also testing whether the last register is shared between codes of a pair (finger/lab glucose, arterial/cuff MAP).

Now that my core is exact, I can test the head cleanly against the tool, using the recorded previous states. Variants:
- z at the prior mean versus averaged over many prior draws;
- with or without the bins.rate[0] and last_status rows;
- last register taken from the same code only.

I compare the predicted means and NLL with the tool's, by code. Mean absolute differences well below 1 mg/dL for one variant confirm the head; a remaining bias points to the register or the type index.

```python
cenV=V['value_bin_centres']; print(type(cenV), (list(cenV.keys())[:6] if isinstance(cenV,dict) else len(cenV)))
CENc={}
if isinstance(cenV,dict):
    for k_,v_ in cenV.items(): CENc[CODE[k_] if k_ in CODE else int(k_)]=np.array(v_,float)
fa=fa4.copy()
Pp=torch.as_tensor(fa.prev.values); dtt=torch.as_tensor(fa.dt.values.astype(np.float32))
codes_=torch.as_tensor(fa.code.values); lb_=torch.as_tensor(fa.lb.values.astype(np.int64))
f8m=np.zeros((len(fa),8),np.float32); f8m[:,0]=fa.has; f8m[:,1]=np.log1p(fa.age_s/3600.0)/10*fa.has; f8m[:,2]=fa.zl; f8m[:,3]=fa.has
cm=np.stack([CENc[c] for c in fa.code.values])
obs_bin=np.array([1+np.searchsorted(np.array(NUMD[(CNAME[c],'value')]['fine_bin_edges']),v,side='left') for c,v in zip(fa.code.values,fa.observed.values)])
def head_probs(OUTp,HHp,dt,codes,lb,f8,W,ty=1,zmode='mean',nz=64,extra=(),seed=0):
    hf=HHp*torch.exp(-dt[:,None,None]*torch.exp(W['core.flow.a']))
    ctx=OUTp+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt)[:,None]/10,W,'gap')
    zo=mlp(ctx,W,'z_prior'); mu,lsd=zo[:,:32],zo[:,32:]
    g=torch.Generator().manual_seed(seed)
    zs=[mu] if zmode=='mean' else [mu+torch.exp(lsd)*torch.randn(mu.shape,generator=g) for _ in range(nz)]
    P=0
    for z in zs:
        q=LN(ctx+F.linear(z,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][ty],W,'query_norm')
        wc=q+W['embed.code.weight'][codes]+F.linear(torch.as_tensor(f8),W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lb]
        if 'rate0' in extra: wc=wc+W['embed.bins.rate.weight'][0]
        if 'st0' in extra: wc=wc+W['last_status.weight'][0]
        lg=mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes*17+lb]
        P=P+torch.softmax(lg,-1)
    return (P/len(zs)).numpy()
out_r=[]
for zmode in ['mean','sample']:
    for ex in [(),('rate0',),('st0',),('rate0','st0')]:
        p_=head_probs(OUTs[Pp],Hs_[Pp],dtt,codes_,lb_,f8m,WO,1,zmode,32,ex)
        pm=(p_[:,1:]*cm).sum(1); nl=-np.log(p_[np.arange(len(p_)),obs_bin]+1e-12)
        d=pm-fa.predicted_mean.values; dn=nl-fa.nll.values
        out_r.append((zmode,'+'.join(ex),round(np.abs(d).mean(),3),round(d.mean(),3),round(np.abs(dn).mean(),4),round(dn.mean(),4)))
print(pd.DataFrame(out_r,columns=['z','extra','mae_pm','bias_pm','mae_nll','bias_nll']).to_string())
```
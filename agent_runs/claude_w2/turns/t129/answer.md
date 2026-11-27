The two extra terms change nothing, so bins.rate[0] and last_status[0] must be zero padding rows. The head itself is still off: predicted means differ from the tool by 2.9 on average, with a bias of +1.6, and the NLL by 0.26. Some detail of the view or the last-register features is wrong.

Next I test the most likely ones and report errors by code:
- the view's gap term as gap(log dt) without the /10 used in the context;
- the age unit of the last register (hours, seconds or minutes);
- the type-embedding index (0–3).

A variant that brings the mean absolute error under about 0.5 identifies the missing detail. Otherwise I'll examine the residuals against dt and the register fields.

```python
def head_pm(gdiv=10.0,age_unit=3600.0,ty=1,f8layout='A',zmode='mean'):
    W=WO
    hf=Hs_[Pp]*torch.exp(-dtt[:,None,None]*torch.exp(W['core.flow.a']))
    base=OUTs[Pp]+LN(hf.mean(1),W,'core.out_norm')
    ctx=base+mlp(torch.log(dtt)[:,None]/10,W,'gap')
    view=base+mlp(torch.log(dtt)[:,None]/gdiv,W,'gap')
    mu=mlp(ctx,W,'z_prior')[:,:32]
    f8=np.zeros((len(fa),8),np.float32); la=np.log1p(fa.age_s.values/age_unit)/10*fa.has.values
    if f8layout=='A': f8[:,0]=fa.has; f8[:,1]=la; f8[:,2]=fa.zl; f8[:,3]=fa.has
    elif f8layout=='B': f8[:,0]=fa.has; f8[:,1]=la; f8[:,2]=fa.zl
    elif f8layout=='C': f8[:,0]=fa.has; f8[:,1]=la; f8[:,2]=fa.zl; f8[:,4]=fa.has
    q=LN(view+F.linear(mu,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][ty],W,'query_norm')
    wc=q+W['embed.code.weight'][codes_]+F.linear(torch.as_tensor(f8),W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lb_]
    lg=mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes_*17+lb_]
    p_=torch.softmax(lg,-1).numpy(); pm=(p_[:,1:]*cm).sum(1); nl=-np.log(p_[np.arange(len(p_)),obs_bin]+1e-12)
    return pm,nl
rr=[]
for gdiv in [10.0,1.0]:
    for au in [3600.0,1.0,60.0]:
        for ty in [0,1,2,3]:
            pm,nl=head_pm(gdiv,au,ty)
            d=pm-fa.predicted_mean.values
            rr.append((gdiv,au,ty,round(np.abs(d).mean(),3),round(np.abs(nl-fa.nll.values).mean(),4)))
rr=pd.DataFrame(rr,columns=['gdiv','age_unit','ty','mae_pm','mae_nll']).sort_values('mae_pm')
print(rr.head(8).to_string())
```
The tool's 4-draw z sampling adds noise with a standard deviation of about 1.3. That predicts a mean absolute difference of about 1.0 mg/dL, but I see 2.9. More importantly, there are clear biases the noise cannot produce:

| code | bias |
|---|---|
| arterial MAP | +2.6 |
| cuff MAP | +2.7 |
| lab glucose | −2.0 |
| finger glucose | +0.5 |

So the head is still structurally off. The biggest suspect is the last register.

Next I test a register shared within each pair (the most recent earlier reading of either code in the pair, from the stored sh_val/sh_h/sh_ev fields). I compute the z-score and bin with the current code's tables or with the source code's tables. If one variant removes the per-code biases, leaving only the noise-level error, the register is shared. If not, I'll examine the last_numeric column layout.

```python
def f8_from(has,age_h,zl):
    f=np.zeros((len(has),8),np.float32); f[:,0]=has; f[:,1]=np.log1p(age_h)/10*has; f[:,2]=zl*has; f[:,3]=has; return f
def zb(code_names,vals):
    z=np.zeros(len(vals)); b=np.zeros(len(vals),np.int64)
    for i,(c,v) in enumerate(zip(code_names,vals)):
        if c is None or not np.isfinite(v): continue
        nd=NUMD[(c,'value')]; z[i]=np.clip((v-nd['median'])/nd['iqr'],-10,10); b[i]=1+np.searchsorted(np.array(nd['fine_bin_edges']),v,side='left')
    return z,b
has_s=np.isfinite(fa.sh_val.values).astype(np.float32); age_s_h=np.where(has_s>0,fa.hour.values-fa.sh_h.values,0.0)
res_s=[]
for zsrc in ['own','src']:
    cn=[CNAME[c] for c in fa.code.values] if zsrc=='own' else [e if isinstance(e,str) else None for e in fa.sh_ev.values]
    cn=[c if h_>0 else None for c,h_ in zip(cn,has_s)]
    z_,b_=zb(cn,fa.sh_val.values)
    f8s=f8_from(has_s,age_s_h,z_)
    P=pm_from(mu,f8s,torch.as_tensor(b_),codes_); pm=(P[:,1:].numpy()*cm).sum(1)
    d=pm-fa.predicted_mean.values
    res_s.append((zsrc,round(np.abs(d).mean(),3),fa.assign(d=d).groupby('event').d.mean().round(2).to_dict()))
for r in res_s: print(r)
print('rows where shared differs from own:', int((fa.sh_ev.astype(str)!=fa.event.astype(str)).sum()))
```
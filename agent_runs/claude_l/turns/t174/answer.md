With `z_head.weight` zeroed, the factual value NLL is 1.55632, slightly better than the original's 1.55781, so the prior draws of z only add noise. The tool's predicted means are now deterministic given the state. I'll compare my head replica (view plus `z_head.bias`, type 1, register layout A) with these exact targets, by code and by whether the register holds a value. Errors near 1e-3 would mean the earlier mismatch was draw noise. Remaining structured errors would point to the register features, which I can then fit cleanly.

```python
f5 = pd.read_csv('/work/fwd_5/factual_edited.csv'); f5s = f5.set_index(['stay_id','hour','event'])
def view_det(s, k, X_=None):
    X_ = Xs if X_ is None else X_
    i0=row_start[s]; B_=BUN[s]
    prev=torch.tensor(X_[i0+k-1],dtype=T); out_p=prev[:192]; h_p=prev[192:960].reshape(4,192)
    dt=(B_[k][0]-B_[k-1][0])*3600.0
    hf=h_p*torch.exp(-dt*torch.exp(fa0))
    return (out_p+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')+W0['z_head.bias'])[None,:]
tests5=[]
for s in ks[:10]:
    B_=BUN[s]
    for k in range(1,len(B_)):
        for a in B_[k][1]:
            if a['code'] in MEASC and a['kind']==0:
                key=(s, round(B_[k][0],4), MEASC[a['code']])
                if key in f5s.index: tests5.append((s,k,a['code'],float(f5s.loc[key,'predicted_mean']), float(f5s.loc[key,'nll'])))
VD={(s,k):view_det(s,k) for (s,k,c,pm,nl) in tests5}
for typ in [1,2,3]:
    for ref in ['cur','prev']:
        errs=[]
        for (s,k,c,pm,nl) in tests5:
            rf=reg_feats2(BUN[s],k,c,ref)
            fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'],0,0,0,0,0]
            q=lnorm(W0, VD[(s,k)]+W0['type_emb.weight'][typ], 'query_norm')
            wc=q+W0['embed.code.weight'][c]+lin(W0,torch.tensor(fv,dtype=T),'last_numeric')+W0['embed.bins.value.weight'][rf['vb']]+W0['embed.bins.rate.weight'][0]
            lg=mlp(W0,wc,'attr_heads.value_bin')+W0['embed_transition.value_bin.weight'][c*17+rf['vb']]
            p=torch.softmax(lg,-1).numpy(); errs.append((float((p[1:]*CENd[c]).sum())-pm, rf['has']))
        e_=np.array(errs); print(typ, ref, 'all med', round(float(np.median(np.abs(e_[:,0]))),4), 'has0 med', round(float(np.median(np.abs(e_[e_[:,1]==0,0]))),4), 'has1 med', round(float(np.median(np.abs(e_[e_[:,1]==1,0]))),4))
```
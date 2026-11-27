The factual CSV gives, per reading, the observed value, the model's predicted mean and the fine-bin NLL. That is enough to identify the value head's unknowns by matching predicted means. I'll build a head replica on the exact fwd_2 states. The options are: type index 3 or 1, the 8 register features in a few orderings, the view's gap input as log(dt)/10 or log(dt), and z drawn from the prior with gelu, averaged over 32 draws. The correct combination should match the tool's means to within the noise of its 4 prior draws, roughly ≤1 mg/dL. Every combination being far off would mean the register features are a different set.

```python
ACTMAP.clear(); ACTMAP['']='gelu'
CEN = np.array(VOCAB['value_bin_centres'], dtype=np.float64)
print('centres shape', CEN.shape)
MEASC = {5:'map_cuff',6:'glucose_finger',7:'map_arterial',10:'glucose_lab'}
def reg_feats(B_, k, code):
    # most recent earlier value of same code in this stay
    for j in range(k-1, -1, -1):
        for a in B_[j][1]:
            if a['code']==code and a['kind']==0 and a['num'][1]==1:
                return dict(has=1.0, age=(B_[k][0]-B_[j][0])*3600.0, z=a['num'][0], vb=a['vb'])
    return dict(has=0.0, age=0.0, z=0.0, vb=0)
def view_of(s, k, gapdiv=10.0, nz=32, seed=0):
    i0=row_start[s]; B_=BUN[s]
    prev=torch.tensor(Xs[i0+k-1],dtype=T); out_p=prev[:192]; h_p=prev[192:960].reshape(4,192)
    dt=(B_[k][0]-B_[k-1][0])*3600.0
    hf=h_p*torch.exp(-dt*torch.exp(fa0))
    ctx=out_p+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
    vbase=out_p+lnorm(W0,hf.mean(0),'core.out_norm')+mlp(W0,torch.tensor([math.log(dt)/gapdiv],dtype=T),'gap')
    zpr=mlp(W0,ctx,'z_prior'); g_=torch.Generator().manual_seed(seed)
    eps=torch.randn(nz,32,generator=g_,dtype=T); zs=zpr[:32]+torch.exp(zpr[32:])*eps
    return vbase[None,:]+lin(W0,zs,'z_head')   # (nz,192)
def pred_mean(views, code, rf, typ, featmode, use_status):
    q=lnorm(W0, views+W0['type_emb.weight'][typ], 'query_norm')
    if featmode=='A': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, 0,0,0,0]
    elif featmode=='B': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, rf['has'],0,0,0]
    elif featmode=='C': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, 1,0,0,0]
    elif featmode=='D': fv=[rf['has'], math.log1p(rf['age']/3600)/10, rf['z'], 0, 0,0,0,0]
    wc=q+W0['embed.code.weight'][code]+lin(W0,torch.tensor(fv,dtype=T),'last_numeric')+W0['embed.bins.value.weight'][rf['vb']]+W0['embed.bins.rate.weight'][0]
    if use_status is not None: wc=wc+W0['last_status.weight'][use_status]
    lg=mlp(W0,wc,'attr_heads.value_bin')+W0['embed_transition.value_bin.weight'][code*17+rf['vb']]
    p=torch.softmax(lg,-1).mean(0).numpy()
    return float((p[1:]*CEN[:16]).sum()) if len(CEN)>=16 else None, p
# collect test readings: first 6 stays
f4s = f4.set_index(['stay_id','hour','event'])
tests=[]
for s in ks[:6]:
    B_=BUN[s]
    for k in range(1,len(B_)):
        for a in B_[k][1]:
            if a['code'] in MEASC and a['kind']==0:
                key=(s, round(B_[k][0],4), MEASC[a['code']])
                if key in f4s.index: tests.append((s,k,a['code'],float(f4s.loc[key,'predicted_mean'])))
print(len(tests))
res_h=[]
for gapdiv in [10.0, 1.0]:
    VV={ (s,k): view_of(s,k,gapdiv) for (s,k,c,pm) in tests }
    for typ in [3,1]:
        for fm in ['A','B','C','D']:
            for st in [None,0,1]:
                errs=[abs(pred_mean(VV[(s,k)],c,reg_feats(BUN[s],k,c),typ,fm,st)[0]-pm) for (s,k,c,pm) in tests]
                res_h.append((gapdiv,typ,fm,st,np.median(errs),np.mean(errs)))
for r_ in sorted(res_h,key=lambda r:r[4])[:6]: print(r_)
```
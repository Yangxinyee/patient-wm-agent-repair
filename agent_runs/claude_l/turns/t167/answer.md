The centres are stored per code, 16 values each (for map_arterial they run from 33 to 101 mmHg). I'll rerun the same head-replica grid using each code's own centres. The correct combination should match the tool's predicted means to within about 1 unit.

```python
ACTMAP.clear(); ACTMAP['']='gelu'
CENd = {CODE_ID[k]: np.array(v,dtype=np.float64) for k,v in VOCAB['value_bin_centres'].items()}
MEASC = {5:'map_cuff',6:'glucose_finger',7:'map_arterial',10:'glucose_lab'}
def reg_feats(B_, k, code):
    for j in range(k-1, -1, -1):
        for a in B_[j][1]:
            if a['code']==code and a['kind']==0 and a['num'][1]==1:
                return dict(has=1.0, age=(B_[k][0]-B_[j][0])*3600.0, z=a['num'][0], vb=a['vb'])
    return dict(has=0.0, age=0.0, z=0.0, vb=0)
def view_of(s, k, gapdiv=10.0, nz=32, seed=0, X_=None):
    X_ = Xs if X_ is None else X_
    i0=row_start[s]; B_=BUN[s]
    prev=torch.tensor(X_[i0+k-1],dtype=T); out_p=prev[:192]; h_p=prev[192:960].reshape(4,192)
    dt=(B_[k][0]-B_[k-1][0])*3600.0
    hf=h_p*torch.exp(-dt*torch.exp(fa0)); on_=lnorm(W0,hf.mean(0),'core.out_norm')
    ctx=out_p+on_+mlp(W0,torch.tensor([math.log(dt)/10],dtype=T),'gap')
    vbase=out_p+on_+mlp(W0,torch.tensor([math.log(dt)/gapdiv],dtype=T),'gap')
    zpr=mlp(W0,ctx,'z_prior'); g_=torch.Generator().manual_seed(seed)
    eps=torch.randn(nz,32,generator=g_,dtype=T); zs=zpr[:32]+torch.exp(zpr[32:])*eps
    return vbase[None,:]+lin(W0,zs,'z_head')
def pred_mean(views, code, rf, typ, featmode, use_status, W=None):
    W = W0 if W is None else W
    q=lnorm(W, views+W['type_emb.weight'][typ], 'query_norm')
    if featmode=='A': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, 0,0,0,0]
    elif featmode=='B': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, rf['has'],0,0,0]
    elif featmode=='C': fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'], 0, 1,0,0,0]
    elif featmode=='D': fv=[rf['has'], math.log1p(rf['age']/3600)/10, rf['z'], 0, 0,0,0,0]
    wc=q+W['embed.code.weight'][code]+lin(W,torch.tensor(fv,dtype=T),'last_numeric')+W['embed.bins.value.weight'][rf['vb']]+W['embed.bins.rate.weight'][0]
    if use_status is not None: wc=wc+W['last_status.weight'][use_status]
    lg=mlp(W,wc,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][code*17+rf['vb']]
    p=torch.softmax(lg,-1).mean(0).numpy()
    return float((p[1:]*CENd[code]).sum()), p
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
                res_h.append((gapdiv,typ,fm,st,round(np.median(errs),3),round(np.mean(errs),3)))
for r_ in sorted(res_h,key=lambda r:r[4])[:6]: print(r_)
```
I now have the bundle builder and the full stepping code. Next I need a teacher-forced replica that, for any stay, records three things:
1. For each glucose/MAP reading: the predicted distribution, mean and NLL.
2. For each drug bundle: the context vector `ctx`, the action token `u` and the recorded insulin and norepinephrine rates.
3. The view at each reading.

With this I can (a) fit the policy's expected dose as a function of the model's own context, (b) regress the model's prediction errors on past dose residuals to see which correction each lag needs, and (c) collect `u` vectors for the z-token copy.

I'll check it against the tool. On the 100 batch-1 tuning stays, with the current `z_head = 0` edit, the tool reported value NLL 1.55632.
- If the replica reproduces this to about 1e-4, it is exact and I can scale it up to training stays.
- Any larger gap means the register or ordering logic is off.

```python
src_tf = r'''
def ctx_of(W,out,h,dt):
    hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    return out+bln(W,hf.mean(1),'core.out_norm')+g
def drug_rates(atoms):
    r={8:0.0,9:0.0}; hs={8:False,9:False}
    for a in atoms:
        if a['act'] and a['kind']==0:
            nm=DRUGN[a['code']]; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']
            r[a['code']]=a['num'][4]*ri+rm; hs[a['code']]=True
    return r,hs
def tf_stay(W,B_,keep_view=False,tokfn=None):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]
    reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    R=[]; D=[]
    for k in range(1,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]
        ea,ua=split_fix(atoms)
        meas=[a for a in ea if a['code'] in MEASC and a['kind']==0]
        if meas:
            views=bview(W,out,h,dt,zs=torch.zeros(1,1,32,dtype=T))
            for a in meas:
                c=a['code']
                if c in reg: tl,zl,vl=reg[c]; hs=1.0; age=(t-tl)*3600.0
                else: hs=0.0; age=0.0; zl=0.0; vl=0
                p=bvalue_probs(W,views,c,torch.tensor([hs],dtype=T),torch.tensor([age],dtype=T),torch.tensor([zl],dtype=T),torch.tensor([vl]))[0]
                pm=float((p[1:]*torch.tensor(CENd[c],dtype=T)).sum())
                R.append(dict(k=k,t=t,code=c,vb=a['vb'],z=a['num'][0],mean=pm,nll=-math.log(float(p[a['vb']])),p=p.numpy().copy(),
                              view=(views[0,0].numpy().copy() if keep_view else None)))
            for a in meas: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        if ua:
            rates,hs_=drug_rates(atoms)
            D.append(dict(k=k,t=t,ctx=ctx_of(W,out,h,dt)[0].numpy().copy(),u=u.numpy().copy(),ins=rates[8],ne=rates[9],hi=hs_[8],hn=hs_[9]))
        Ms.append(e); ts.append(t)
        Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        out,h,_=bstep(W,out,h,dt,e[None],u[None],Mt,ages)
    return R,D
'''
open('/work/tf.py','w').write(src_tf); exec(src_tf)
W_cur=getW(load,NAMES)
t0=time.time(); allR=[]; allD=[]
for s in list(BUN.keys()):
    R,D=tf_stay(W_cur,BUN[s]); 
    for r in R: r['s']=s
    for d_ in D: d_['s']=s
    allR+=R; allD+=D
print('time',time.time()-t0,'n readings',len(allR),'n drug bundles',len(allD))
nll=np.array([r['nll'] for r in allR]); cds=np.array([r['code'] for r in allR])
print('mean value NLL',nll.mean(),' (tool z_head=0: 1.55632)')
for c in [6,10,7,5]: print(c,(cds==c).sum(),nll[cds==c].mean())
```
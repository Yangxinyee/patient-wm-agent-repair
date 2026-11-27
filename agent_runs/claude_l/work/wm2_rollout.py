
DRUGN={8:'insulin',9:'norepinephrine'}
def edit_atom(a, f):
    nm=DRUGN[a['code']]
    dm,di=NUMV[(nm,'dose')]['median'],NUMV[(nm,'dose')]['iqr']; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']
    dose=a['num'][2]*di+dm; rate=a['num'][4]*ri+rm
    b=dict(a); b['num']=(0,0,zsc(nm,'dose',dose*f),1,zsc(nm,'rate',rate*f),1)
    b['db']=fbin_f32(nm,'dose',dose*f); b['rb']=fbin_f32(nm,'rate',rate*f); return b
def u_for(atoms, t, cond, t_edit):
    ua=[a for a in atoms if a['act'] and a['kind']==0]
    if not ua: return u_empty
    if cond is not None and t>=t_edit:
        dcode,f=cond; ua=[edit_atom(a,f) if a['code']==dcode else a for a in ua]
    return set_enc(W0c[0], [fix_atom2(a) for a in ua], 'encode_action')
W0c=[W0]
def rollout(W, B_, conds, npaths=20, seed=0, t_edit=12.0, t_hide=13.0, nz=4):
    W0c[0]=W
    C=len(conds); Bt=C*npaths; gen=torch.Generator().manual_seed(seed)
    times=[b[0] for b in B_]
    k0=next(k for k,t in enumerate(times) if t>t_edit-1e-9 and any(a['act'] for a in B_[k][1]) or t>=t_hide)
    out=None; h=None; Ms=[]; ts=[]
    for k in range(k0):
        e,u=enc_eu2(B_[k][1]) if W is W0 else (None,None)
        if W is not W0:
            ea,ua=split_fix(B_[k][1]); e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        if k==0:
            out=torch.tensor(Xs[0,:192],dtype=T)[None] if False else None
        if k>0:
            Mt=torch.stack(Ms+[e])[None]; ages=torch.tensor([(times[k]-tj)*3600.0 for tj in ts+[times[k]]],dtype=T)
            out,h,_=bstep(W,out,h,(times[k]-times[k-1])*3600.0,e[None],u[None],Mt,ages)
        else:
            out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
        Ms.append(e); ts.append(times[k])
    # registers per code from prefix
    reg={}
    for k in range(k0):
        for a in B_[k][1]:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[k], a['num'][0], a['vb'])
    out=out.repeat(Bt,1); h=h.repeat(Bt,1,1); M=torch.stack(Ms)[None].repeat(Bt,1,1)
    RG={c:(torch.full((Bt,),v[0],dtype=T), torch.full((Bt,),v[1],dtype=T), torch.full((Bt,),v[2],dtype=torch.long)) for c,v in reg.items()}
    rec={}
    for k in range(k0,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]
        ea,ua=split_fix(atoms)
        meas=[i for i,a in enumerate(ea) if a['code'] in MEASC and a['kind']==0]
        rows=(atom_matrix(W,ea) if ea else torch.zeros(0,192,dtype=T))[None].repeat(Bt,1,1)
        if t>=t_hide and meas:
            eps=torch.randn(npaths,nz,32,generator=gen,dtype=T).repeat(C,1,1)
            views=bview(W,out,h,dt,zs=eps)
            U=torch.rand(npaths,len(meas),generator=gen,dtype=T).repeat(C,1)
            for j,i in enumerate(meas):
                c=ea[i]['code']
                if c in RG: tl,zl,vl=RG[c]; has=torch.ones(Bt,dtype=T); age=(t-tl)*3600.0
                else: has=torch.zeros(Bt,dtype=T); age=torch.zeros(Bt,dtype=T); zl=torch.zeros(Bt,dtype=T); vl=torch.zeros(Bt,dtype=torch.long)
                p=bvalue_probs(W,views,c,has,age,zl,vl)
                pm=(p[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1)
                rec[(round(t),c)]=pm.reshape(C,npaths).numpy()
                q=p[:,1:]/p[:,1:].sum(1,keepdim=True); cdf=torch.cumsum(q,1)
                b=(U[:,j:j+1]>cdf).sum(1).clamp(max=15)+1
                x=torch.tensor(CENd[c],dtype=T)[b-1]
                md,iq=MED[c]; zv=((x-md)/iq).clamp(-10,10)
                rows[:,i,:]=rows[:,i,:]+W['embed.bins.value.weight'][b]-W['embed.bins.value.weight'][ea[i]['vb']]+(zv-ea[i]['num'][0])[:,None]*W['embed.numeric.weight'][:,0]
                RG[c]=(torch.full((Bt,),t,dtype=T), zv, b)
        else:
            for i in meas:
                c=ea[i]['code']; RG[c]=(torch.full((Bt,),t,dtype=T), torch.full((Bt,),ea[i]['num'][0],dtype=T), torch.full((Bt,),ea[i]['vb'],dtype=torch.long))
        e=bset_enc(W,rows,'encode') if ea else set_enc(W,[],'encode')[None].repeat(Bt,1)
        u=torch.cat([u_for(atoms,t,cd,t_edit)[None].repeat(npaths,1) for cd in conds],0)
        M=torch.cat([M,e[:,None]],1); ts.append(t)
        ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        out,h,_=bstep(W,out,h,dt,e,u,M,ages)
    return rec

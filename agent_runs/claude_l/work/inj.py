
def bstep_z(W,out,h,dt,e,u,Mt,ages,zfn=None,info=None):
    hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap')
    ctx=out+bln(W,hf.mean(1),'core.out_norm')+g
    if zfn is None:
        z=bmlp(W,torch.cat([ctx,e,u],-1),'z_post')[:,:32]; zt=blin(W,z,'z_in')
    else: zt=zfn(ctx,e,u,info)
    tok=torch.cat([hf+W['core.slots'],e[:,None],u[:,None],zt[:,None]],1)
    tok=bself(W,tok,'core.prelude.0'); E=bln(W,tok[:,:4],'core.prelude_norm')
    ae=bmlp(W,torch.log1p(ages)[:,None],'core.age'); Mk=Mt+ae[None]
    x=hf+E
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        a_=bln(W,x,L+'.norm1'); x=x+bmha(W,a_,a_,L+'.attn')
        x=x+bmha(W,bln(W,x,L+'.norm2'),Mk,L+'.cross')
        x=x+bmlp(W,bln(W,x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    h2=abar*hf+sp*(E@W['core.b_in.weight'].T)+(x-(hf+E))
    y=bself(W,h2@W['core.c_out.weight'].T,'core.coda.0')
    return bln(W,y.mean(1),'core.out_norm'), y, ctx
def info_of(atoms):
    r,hs=drug_rates(atoms)
    return dict(di=torch.tensor([r[8]],dtype=T),hi=torch.tensor([float(hs[8])],dtype=T),dn=torch.tensor([r[9]],dtype=T),hn=torch.tensor([float(hs[9])],dtype=T))
def tf_stay_z(W,B_,zfn):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    R=[]
    for k in range(1,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]; ea,ua=split_fix(atoms)
        meas=[a for a in ea if a['code'] in MEASC and a['kind']==0]
        if meas:
            views=bview(W,out,h,dt,zs=torch.zeros(1,1,32,dtype=T))
            for a in meas:
                c=a['code']
                if c in reg: tl,zl,vl=reg[c]; hs=1.0; age=(t-tl)*3600.0
                else: hs=0.0; age=0.0; zl=0.0; vl=0
                p=bvalue_probs(W,views,c,torch.tensor([hs],dtype=T),torch.tensor([age],dtype=T),torch.tensor([zl],dtype=T),torch.tensor([vl]))[0]
                R.append((t,c,float((p[1:]*torch.tensor(CENd[c],dtype=T)).sum()),-math.log(float(p[a['vb']]))))
            for a in meas: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        out,h,_=bstep_z(W,out,h,dt,e[None],u[None],Mt,ages,zfn=zfn,info=(info_of(atoms) if ua else None))
    return R

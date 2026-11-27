Even a boosted-tree model on the context predicts the doses only moderately (insulin R² 0.59, norepinephrine 0.49), against 0.82–0.94 for the policy from the full record. So the model's state does not hold the policy's main inputs: the exact latest reading and the previous rate. Those sit in the value head's "last value" register instead, and that is where the model must have absorbed the policy, since my earlier check found its errors depend on the true residual and not on the policy part.

Two consequences:
- A context-based residual inside `z_post` is only a partial residual. The part it misses is either the head's register-based absorption, or a correction applied in the head from the last reading (`embed_transition` is a lookup by last value bin, so it can subtract a function of the last glucose).
- Rather than keep theorising, I'll test what an injection actually does in the exact replica.

The test, done in the replica only, with no weights changed yet:
- At drug bundles, set the z token to `u + κ·r·w_ins`, where `w_ins` is the model's native insulin-dose direction in `u`-space and r = recorded dose − expected dose from context.
- At other bundles, set the z token to `u`.
- Run teacher forcing on the 100 tuning stays for κ = 0 and κ = 4. I'll measure the change in value NLL and how strongly a unit residual shifts glucose predictions at lags 1–3.

How I'll read it:
- **Works:** κ = 0 leaves NLL essentially unchanged (duplicating the `u` token splits attention without changing the output), and κ = 4 gives a clean lag-1 glucose effect of roughly −1.2 per U/h with a small NLL change.
- **Doesn't work:** NLL moves at κ = 0, or the effect is tiny or nonlinear. Then the attention-split mechanism is not behaving as assumed.

```python
src_inj = r'''
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
'''
open('/work/inj.py','w').write(src_inj); exec(src_inj)
# native dose directions in u-space
def find_atom(code):
    for s in BTR:
        for t,al in BTR[s]:
            acts=[a for a in al if a['act'] and a['kind']==0]
            if len(acts)==1 and acts[0]['code']==code: return acts[0]
a_ins=find_atom(8); a_ne=find_atom(9)
def u_rate(a,r):
    nm=DRUGN[a['code']]; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']; r0=a['num'][4]*ri+rm
    return set_enc(W_cur,[fix_atom2(edit_atom(a,r/r0))],'encode_action')
for a,grid,nm in [(a_ins,np.arange(0.5,10.01,0.5),'ins'),(a_ne,np.arange(0.05,2.01,0.05),'ne')]:
    Ug=torch.stack([u_rate(a,r) for r in grid]).numpy()
    obs=grid<= (5.0 if nm=='ins' else 1.0)
    A=np.column_stack([np.ones(obs.sum()),grid[obs]]); cf,*_=np.linalg.lstsq(A,Ug[obs],rcond=None)
    fit=A@cf; r2=1-((Ug[obs]-fit)**2).sum()/((Ug[obs]-Ug[obs].mean(0))**2).sum()
    print(nm,'slope norm %.4f'%np.linalg.norm(cf[1]),'lin R2 in obs range %.4f'%r2,'|u(max)-u(min)| %.3f'%np.linalg.norm(Ug[-1]-Ug[0]))
    globals()['w_'+nm]=torch.tensor(cf[1],dtype=T)
pi_i=(torch.tensor(PIH['ins'].coef_,dtype=T),float(PIH['ins'].intercept_)); pi_n=(torch.tensor(PIH['ne'].coef_,dtype=T),float(PIH['ne'].intercept_))
def make_zfn(ki,kn):
    def zfn(ctx,e,u,info):
        zt=u.clone()
        if info is not None:
            ri=(info['di']-(ctx@pi_i[0]+pi_i[1]))*info['hi']; rn=(info['dn']-(ctx@pi_n[0]+pi_n[1]))*info['hn']
            zt=zt+ki*ri[:,None]*w_ins[None]+kn*rn[:,None]*w_ne[None]
        return zt
    return zfn
res_tf={}
for lab_,ki,kn in [('dup',0,0),('ki4',4,0)]:
    t0=time.time(); rows=[]
    for s in BUN: rows+=[(s,)+r for r in tf_stay_z(W_cur,BUN[s],make_zfn(ki,kn))]
    res_tf[lab_]=rows; nl=np.array([r[4] for r in rows]); cd=np.array([r[2] for r in rows])
    print(lab_,'time %.1f'%(time.time()-t0),'NLL %.6f'%nl.mean(),{c:round(nl[cd==c].mean(),4) for c in [6,10,7,5]})
```
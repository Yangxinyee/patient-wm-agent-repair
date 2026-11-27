The slope-direction injection saturates:
- Doubling κ from 4 to 8 moves the lag-1 finger effect only from −0.65 to −0.76.
- The effect explains just 30% of its own variation across states, i.e. it is state-dependent and nonlinear.
- It leaks into MAP, about +0.15 per unit residual.
- It reaches no further than lag 1: the lag-2 and lag-3 coefficients are about 0.

So a crude direction in the dose token cannot carry the large, linear effects a doubled dose needs. Two causes are possible. The keys may shift so the z token loses its share of attention, or the prelude normalisation may squash the perturbation.

To pick a direction rationally I'll measure finite-difference Jacobians at about 60 insulin drug bundles (2 per stay over 30 tuning stays). For each, I perturb the z token in all 192 coordinates in one batched step and roll forward with teacher forcing for 3 hours. I record:
- the change in predicted finger, lab, arterial and cuff means at each later reading;
- the change in the view at each later reading;
- the change in the z token's attention keys for heads 0–2.

The decision follows from the result:
- **Native route:** some direction strongly shifts glucose at lag 1 while leaving keys and MAP almost unchanged. I keep the injection in the core.
- **Head amplifier:** the injected signal is only weakly expressed by the native value head but stays clearly decodable in the view for one or more hours. I add the z_prior/z_head adapter.

```python
def run_prefix(W,B_,kstop):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    for k in range(1,kstop):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]; ea,ua=split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        out,h,_=bstep_z(W,out,h,dt,e[None],u[None],Mt,ages,zfn=lambda c,e_,u_,i: u_)
    return out,h,Ms,ts,reg
def perturb(W,B_,k,eps=1e-4,hrs=3.5):
    out,h,Ms,ts,reg=run_prefix(W,B_,k); times=[b[0] for b in B_]; Bn=193
    t=times[k]; dt=(t-times[k-1])*3600.0; ea,ua=split_fix(B_[k][1])
    e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action')
    Z=u[None].repeat(Bn,1).clone(); Z[1:]+=eps*torch.eye(192,dtype=T)
    a1=bln(W,Z,'core.prelude.0.norm1'); Kz=(a1@W['core.prelude.0.attn.kv.weight'][:144].T); Jk=((Kz[1:]-Kz[0])/eps).T.numpy()
    Ms2=Ms+[e]; ts2=ts+[t]; M=torch.stack(Ms2)[None].repeat(Bn,1,1); ages=torch.tensor([(t-tj)*3600.0 for tj in ts2],dtype=T)
    outB,hB,_=bstep_z(W,out.repeat(Bn,1),h.repeat(Bn,1,1),dt,e[None].repeat(Bn,1),u[None].repeat(Bn,1),M,ages,zfn=lambda c,e_,u_,i: Z)
    res=[]
    for k2 in range(k+1,len(B_)):
        t2=times[k2]
        if t2>t+hrs: break
        dt2=(t2-times[k2-1])*3600.0; ea2,ua2=split_fix(B_[k2][1])
        meas=[a for a in ea2 if a['code'] in MEASC and a['kind']==0]
        if meas:
            views=bview(W,outB,hB,dt2,zs=torch.zeros(Bn,1,32,dtype=T)); rec={'lag':t2-t,'J_view':((views[1:,0]-views[0,0])/eps).numpy().T}
            for a in meas:
                c=a['code']
                if c in reg: tl,zl,vl=reg[c]; hs=1.0; age=(t2-tl)*3600.0
                else: hs=0.0;age=0.0;zl=0.0;vl=0
                p=bvalue_probs(W,views,c,torch.full((Bn,),hs,dtype=T),torch.full((Bn,),age,dtype=T),torch.full((Bn,),zl,dtype=T),torch.full((Bn,),vl,dtype=torch.long))
                m_=(p[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1).numpy(); rec[c]=(m_[1:]-m_[0])/eps
            res.append(rec)
            for a in meas: reg[a['code']]=(t2,a['num'][0],a['vb'])
        e2=set_enc(W,ea2,'encode'); u2=set_enc(W,ua2,'encode_action') if ua2 else u_empty
        ts2.append(t2); M=torch.cat([M,e2[None,None].repeat(Bn,1,1)],1); ages=torch.tensor([(t2-tj)*3600.0 for tj in ts2],dtype=T)
        outB,hB,_=bstep_z(W,outB,hB,dt2,e2[None].repeat(Bn,1),u2[None].repeat(Bn,1),M,ages,zfn=lambda c,e_,u_,i: u_)
    return dict(Jk=Jk,u=u.numpy(),res=res)
JAC=[]; t0=time.time()
for s in list(BUN.keys())[:30]:
    B_=BUN[s]; ks=[k for k,(t,al) in enumerate(B_) if 5<t<20 and any(a['act'] and a['kind']==0 and a['code']==8 for a in al)]
    for k in ks[::max(1,len(ks)//2)][:2]:
        JAC.append(perturb(W_cur,B_,k))
print('states',len(JAC),'time %.1f'%(time.time()-t0))
for L in [1,2,3]:
    for c in [6,10,7,5]:
        G=[r[c] for J in JAC for r in J['res'] if round(r['lag'])==L and c in r]
        if G: G=np.array(G); gm=G.mean(0); print('lag',L,'code',c,'n',len(G),'|mean grad| %.3f'%np.linalg.norm(gm),'mean |grad| %.3f'%np.linalg.norm(G,axis=1).mean())
```
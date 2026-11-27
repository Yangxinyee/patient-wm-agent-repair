The per-bundle Jacobians are consistent: the per-bundle norms (0.23) almost equal the norm of the mean. They are simply small, and the effects on finger and lab almost cancel.

A direction that moves finger and lab together achieves only 0.088 mg/dL per unit of z-token displacement for insulin. For NE the equivalent on arterial and cuff together is 0.029 mmHg per unit. The insulin target of −1.74 per U/h would need a displacement of about 20 per U/h, as large as u itself. **So the z token cannot shift the reading predictions directly with a small, linear injection.**

There is a better split of roles:
- **Core side:** the swap injection writes a residual marker into the state, chosen to be maximally decodable from the next context rather than to shift predictions itself.
- **Head side:** the unused prior path (`z_prior` MLP on the context, then `z_head` into the value head's view) decodes the marker at the next reading and applies the effect. It shifts the glucose or MAP prediction along a view direction chosen to move finger and lab, or arterial and cuff, together.

This only works if the marker is decodable with good signal-to-noise. Next I'll compute, at the same 219 drug bundles in the swap model, the Jacobian of the next view (context) with respect to the z token. Then I'll find the direction with the best signal-to-noise against the natural context covariance, using the 57k training drug-bundle contexts.

**Feasible:** signal-to-noise of at least about 1 per unit displacement (an injection of 5 gives decoded noise ≈ 0.2 of the signal), with a gain that varies by less than about 20% across bundles. **Infeasible** if the signal-to-noise stays near the earlier 0.27.

```python
def jacctx_stay(W, B_, maxb=40, eps=0.5):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]
    Dl=torch.cat([torch.zeros(1,192,dtype=T),eps*torch.eye(192,dtype=T),-eps*torch.eye(192,dtype=T)],0); nb=385; res=[]
    bpt=torch.tensor(bprime,dtype=T)
    for k_ in range(1,len(B_)):
        t=times[k_]; dt=(t-times[k_-1])*3600.0; ea,ua=split_fix(B_[k_][1])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        ch=[a['code'] for a in ua] if ua else []
        if ch and k_+1<len(B_) and len(res)<maxb:
            o2,h2,_=bstep_z(W,out.repeat(nb,1),h.repeat(nb,1,1),dt,e[None].repeat(nb,1),(u+bpt)[None].repeat(nb,1),Mt.repeat(nb,1,1),ages,zfn=lambda c_,e_,u_,i_: u[None]+Dl)
            dtn=(times[k_+1]-t)*3600.0; vw=bview(W,o2,h2,dtn,zs=torch.zeros(nb,1,32,dtype=T))
            vw=vw.reshape(nb,-1)[:, :192].numpy() if vw.dim()>2 else vw.numpy()
            res.append((8 in ch, 9 in ch, ((vw[1:193]-vw[193:])/(2*eps)).T, vw[0]))
        out,h,_=bstep_z(W,out,h,dt,e[None],(u+bpt)[None] if ua else u[None],Mt,ages,zfn=lambda c_,e_,u_,i_: u[None])
    return res
t0=time.time(); JC=[]
for s in keys[:20]: JC+=jacctx_stay(W0z,BUN[s])
print('bundles',len(JC),'%.0fs'%(time.time()-t0), 'view shape', JC[0][2].shape)
Sv2=np.cov(DC.T)+1e-6*np.eye(192); Si2=np.linalg.inv(Sv2)
for nm,ix in [('ins',0),('ne',1)]:
    Jb=np.array([r[2] for r in JC if r[ix]]); Jmn=Jb.mean(0)
    Mq2=Jmn.T@Si2@Jmn; ev2,evc2=np.linalg.eigh((Mq2+Mq2.T)/2); b_=evc2[:,-1]; d_=Si2@Jmn@b_; d_=d_/(d_@Jmn@b_)
    g_=np.array([d_@Jb[i]@b_ for i in range(len(Jb))])
    print(nm,'n',len(Jb),'|J| fro med %.3f ; best snr/unit %.2f ; gain mean %.3f sd %.3f ; 2nd snr %.2f'%(np.median([np.linalg.norm(x) for x in Jb]),np.sqrt(ev2[-1]),g_.mean(),g_.std(),np.sqrt(ev2[-2])))
```
The architecture says u is "zero if the bundle has none". My replica, which matches the tool's states exactly, uses `encode_action(∅)` (norm 11.6), so the empty set evidently still passes through the size embedding and the final norm.

Either way the swap is safe at non-drug bundles:
- If the bias shift reaches the empty u, the z copy of u_empty wins head 3 with identical values.
- If it doesn't, z and u tie exactly and are identical.

The injection is zero there in both cases. At drug bundles the swap works: z is attended with weight 0.955 in heads 0–2, and with zero injection the factual NLL is unchanged (−0.00005).

Next is the injection direction. At insulin and NE bundles from 20 tuning stays I'll compute, in the swap model:
- the Jacobian of the next-hour predicted means for finger, lab, arterial and cuff, by central differences over all 192 directions of the z token;
- the z token's attention-score gradients in heads 0–2.

Then for each drug I'll take the direction that most moves its own readings (glucose for insulin, MAP for NE). It will be orthogonal to the other drug's readings and to the attention-score gradients, so the response is smooth and specific.

**Expectation:** a direction with a glucose effect of order 1 mg/dL per unit norm or more. That would mean the needed injections are small, a few units against |u| ≈ 25, and stay in the linear regime.

```python
CODES=[6,5,7,10]
def jac_stay(W, B_, maxb=40, eps=0.5):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    Dl=torch.cat([torch.zeros(1,192,dtype=T),eps*torch.eye(192,dtype=T),-eps*torch.eye(192,dtype=T)],0); nb=385; res=[]
    bpt=torch.tensor(bprime,dtype=T)
    for k_ in range(1,len(B_)):
        t=times[k_]; dt=(t-times[k_-1])*3600.0; atoms=B_[k_][1]; ea,ua=split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        codes_here=[a['code'] for a in ua] if ua else []
        if codes_here and k_+1<len(B_) and len(res)<maxb and all(c in reg for c in CODES):
            o2,h2,_=bstep_z(W,out.repeat(nb,1),h.repeat(nb,1,1),dt,e[None].repeat(nb,1),(u+bpt)[None].repeat(nb,1),Mt.repeat(nb,1,1),ages,zfn=lambda c_,e_,u_,i_: u[None]+Dl)
            dtn=(times[k_+1]-t)*3600.0; views=bview(W,o2,h2,dtn,zs=torch.zeros(nb,1,32,dtype=T)); J=[]
            for code in CODES:
                tl,zl,vl=reg[code]; age=(times[k_+1]-tl)*3600.0
                p=bvalue_probs(W,views,code,torch.ones(nb,dtype=T),torch.full((nb,),age,dtype=T),torch.full((nb,),zl,dtype=T),torch.full((nb,),vl,dtype=torch.long))
                mu=(p[:,1:]*torch.tensor(CENd[code],dtype=T)).sum(1).numpy(); J.append((mu[1:193]-mu[193:])/(2*eps))
            res.append((8 in codes_here, 9 in codes_here, np.array(J)))
        zf0=(lambda c_,e_,u_,i_: u[None]) 
        out,h,_=bstep_z(W,out,h,dt,e[None],(u+bpt)[None] if ua else u[None],Mt,ages,zfn=zf0)
    return res
t0=time.time(); JR=[]
for s in keys[:20]: JR+=jac_stay(W0z,BUN[s])
hasI=np.array([r[0] for r in JR]); hasN=np.array([r[1] for r in JR]); JJ=np.array([r[2] for r in JR])  # (n,4,192)
print('bundles',len(JR),'ins',hasI.sum(),'ne',hasN.sum(),'%.0fs'%(time.time()-t0))
for nm,m in [('ins',hasI),('ne',hasN)]:
    Jm_=JJ[m].mean(0); print(nm,'mean-J norms finger %.2f lab %.2f art %.2f cuff %.2f'%tuple(np.linalg.norm(Jm_,axis=1)), ' cos(finger,lab) %.2f cos(art,cuff) %.2f'%(Jm_[0]@Jm_[1]/np.linalg.norm(Jm_[0])/np.linalg.norm(Jm_[1]),Jm_[2]@Jm_[3]/np.linalg.norm(Jm_[2])/np.linalg.norm(Jm_[3])))
```
Diagnostics, run 278: this explains why the model ignores doses.
1. **Swamped dose signal:** the prelude's slot residual stream has norm about 800, while head 2's dose output is under 1 per unit, so the dose is lost before the prelude norm.
2. **Unused directions:** the normalised slots E have a highly rank-deficient covariance (almost all eigenvalues ≈ 0). A moderate head-2 injection placed in those null directions is therefore clean: SNR 6.6 per unit of injection, and the overall effect stays small.
3. **SSM path:** softplus(delta_in) ≈ 1 and a_bar ≈ 0.37, so b_in(E) feeds h_pre directly.

Plan: a moderate head-2 injection, about 5% of the slot norm, puts each drug's rate into its own null direction of E. A rank-1 term added to b_in then reads that null direction, which carries no natural signal, and writes it into h_pre along a direction t with gain.

To choose t, I'll test 48 candidate directions, aimed at directions where the core state y has no natural variance. For each I'll measure the response in the value-head input and the disturbance of predicted means, then solve for the best combination. I'll include a noise floor so the result is robust to GPU precision.

The plan is confirmed if a combination gives SNR ≥ 5 per unit rate with disturbance ≤ 0.3 RMS. If every t disturbs strongly, the plan fails.

```python
CO='core.c_out.weight'; BI='core.b_in.weight'; Bin0=W[BI].clone(); Co=T64(W[CO]).numpy()
evE,vE=np.linalg.eigh(CE); mE=EE.mean(0); mu=mE/np.linalg.norm(mE)
Nb=vE[:,evE<1e-6*evE.max()]; Nb=Nb-np.outer(mu,mu@Nb); Qn,_=np.linalg.qr(Nb)
pEI=Ve[:,-1]; pEN=Ve[:,-2]; dEI=A@pEI; dEN=A@pEN
sI=Qn@(Qn.T@dEI); sN=Qn@(Qn.T@dEN); sN=sN-sI*(sI@sN)/(sI@sI)
print('null dim',Qn.shape[1],'frac dE in null I/N',round(float(np.linalg.norm(sI)/np.linalg.norm(dEI)),3),round(float(np.linalg.norm(sN)/np.linalg.norm(dEN)),3))
cp=40.0; WkvE=Wkv0.clone(); WkvE[288:336]+=cp*(torch.outer(torch.as_tensor(pEI,dtype=dty),wI)+torch.outer(torch.as_tensor(pEN,dtype=dty),wN))
uI=sI/(sI@sI)/cp; uN=sN/(sN@sN)/cp     # u.E = signal (rate units) at drug bundles
Y=np.concatenate([T64(X_of(s))[:,192:960].reshape(-1,4,192).mean(1).numpy() for s in ST]); CY=np.cov(Y.T); evY,vY=np.linalg.eigh(CY)
NY=vY[:,evY<1e-6*evY.max()]; rs=np.random.RandomState(1)
T1=NY@np.linalg.qr(rs.randn(NY.shape[1],24))[0]; T2=np.linalg.solve(Co,NY@np.linalg.qr(rs.randn(NY.shape[1],24))[0]); T2/=np.linalg.norm(T2,axis=0)
TC=np.concatenate([T1,T2],1); print('nullY',NY.shape[1])
W[KV]=WkvE; base2={}
for s in ST:
    O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); base2[s]=hf2(O,H,RTs[s],info[s]['eps'])
DF2=[];DP2=[]
try:
    for i in range(TC.shape[1]):
        Bn=Bin0.clone(); Bn+=torch.outer(torch.as_tensor(TC[:,i],dtype=dty),torch.as_tensor(uI,dtype=dty)); W[BI]=Bn; dfi=[];dpi=[]
        for s in ST:
            O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); wc,lg=hf2(O,H,RTs[s],info[s]['eps'])
            dfi.append(wc-base2[s][0]); dpi.append(mean_of(lg,info[s]['ev'])-mean_of(base2[s][1],info[s]['ev']))
        DF2.append(np.concatenate(dfi)); DP2.append(np.concatenate(dpi))
finally: W[BI]=Bin0; W[KV]=Wkv0
DF2=np.stack(DF2,-1); DP2=np.stack(DP2,-1); F2=np.concatenate([base2[s][0] for s in ST]); C2=np.cov(F2.T)
fl=1e-4*np.trace(C2)/192; S2i=np.linalg.inv(C2+fl*np.eye(192))
J2=(DF2[imm]/SG[imm,None,None]).mean(0); G2=J2.T@S2i@J2; Q2=DP2.T@DP2/len(DP2)
l2,V2_=eigh(G2,Q2+1e-3*np.trace(Q2)/48*np.eye(48)+1e-6*np.eye(48)); t_=V2_[:,-1]
print('time',round(time.time()-t0,1),'snr',round(float(np.sqrt(t_@G2@t_)),2),'dist',round(float(np.sqrt(t_@Q2@t_)),4),'|t|',round(float(np.linalg.norm(t_)),2),'diag-best',round(float(np.sqrt(np.diag(G2).max())),3))
dft=DF2@t_; m2=np.sqrt(np.einsum('ni,ij,nj->n',dft,S2i,dft)); print([(lo,round(float(np.median(m2[(AGE>lo)&(AGE<=hi)])),2)) for lo,hi in ((0,1.01),(1.01,2.01),(2.01,3.01),(3.01,6),(6,99))])
```
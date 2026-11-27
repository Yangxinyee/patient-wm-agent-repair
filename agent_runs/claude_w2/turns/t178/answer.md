The model propagates its own draws like a local-level filter. A +2-bin shift in the drawn reading at hour 13 moves the next predicted mean by about 11 mg/dL, roughly 0.42 of the shift. After that the change stays flat at 10–12 through hour 24, with no decay.

So a correction added in the head is only partly absorbed. To produce the true effect E(m) = Σ K(k)Δd, the head must add an innovation term r_t = Σ_k G(k)·Δd_{t−k}, where G(m) = K(m) − 0.42·Σ_{j<m} G(j) = −2.0, −1.65, −0.70, +0.04, +0.40, …. That needs dose memory over about 10 hours, which the model lacks, so I will eventually need a dedicated state channel.

Before engineering one I want to confirm the target filter with an oracle. I compute r from the exact arm dose differences and apply it as an exact mean-shift (exponential tilt) of the glucose predictions inside my rollout engine, on 100 training stays with insulin after hour 12. Then I compare the readout with the truth Σ K(k)Δd for x2, x½, x0 and x2-from-16. A close match validates the filter. A systematic gap tells me how to re-tune the gain.

```python
code_or = r'''
GLC=[CODE['glucose_finger'],CODE['glucose_lab']]; MPC=[CODE['map_arterial'],CODE['map_cuff']]
def tilt_P(P,cen,r,iters=4):
    p=P[:,1:].double(); s=p.sum(1,keepdim=True); p=p/s; c=cen.double(); r=torch.as_tensor(r,dtype=torch.float64)
    mu0=(p*c).sum(1); var0=((p*c*c).sum(1)-mu0**2).clamp(min=1e-3); beta=r/var0
    lp=torch.log(p.clamp(min=1e-30))
    for _ in range(iters):
        q=torch.softmax(lp+beta[:,None]*c,1); mu=(q*c).sum(1); var=((q*c*c).sum(1)-mu**2).clamp(min=1e-3)
        beta=beta+(mu0+r-mu)/var
    q=torch.softmax(lp+beta[:,None]*c,1); Pn=P.clone().double(); Pn[:,1:]=q*s; return Pn.float()
def dose_stack(R,arms,drug): return np.stack([dose_grid(R,drug,arm_mult(R,a,drug)) for a in arms])
def lag_diff(DG,a,si,tau,L=12):
    X=np.zeros((len(si),L))
    for k in range(1,L+1):
        ix=tau-k+3; ok=ix>=0
        X[ok,k-1]=DG[a[ok],si[ok],ix[ok]]-DG[0,si[ok],ix[ok]]
    return X
def make_oracle(R,DGi,Gi,DGn=None,Gn=None):
    A=R['A']; code=A.code.values; tt=A.t.values
    def hook(k,hn_,hj,P,ctx):
        P=P.clone()
        for codes_,DG,Gf in [(GLC,DGi,Gi),(MPC,DGn,Gn)]:
            if DG is None: continue
            m=np.isin(code[hj],codes_)
            if not m.any(): continue
            si=ctx['seq_si'][hn_[m]]; a=ctx['seq_arm'][hn_[m]]; tau=np.round(tt[hj[m]]).astype(int)
            r=lag_diff(DG,a,si,tau,len(Gf))@Gf
            if np.abs(r).max()==0: continue
            cen=torch.as_tensor(CENT[code[hj[m]]],dtype=torch.float32)
            P[torch.as_tensor(np.where(m)[0])]=tilt_P(P[torch.as_tensor(np.where(m)[0])],cen,r)
        return P
    return hook
def truth_rows(R,DG,a,rows,Kf):
    A=R['A']; si=A.si.values[rows]; tau=np.round(A.t.values[rows]).astype(int)
    return lag_diff(DG,np.full(len(rows),a),si,tau,len(Kf))@Kf
def G_from_K(K,kap):
    G=np.zeros(len(K))
    for m in range(len(K)): G[m]=K[m]-kap*G[:m].sum()
    return G
'''
open('/work/oracle.py','w').write(code_or); exec(code_or)
ins12=tr[(tr.event=='insulin')&(tr.hour>=12)&(tr.hour<20)].stay_id.unique()
ids_o=list(np.random.default_rng(11).choice(ins12,100,replace=False)); Ro=prep_rollout(tr,ids_o)
arms_o=[{},{'ins':2.0},{'ins':0.5},{'ins':0.0},{'ins':2.0,'ins_from':16.0}]
DGi_o=dose_stack(Ro,arms_o,'insulin')
Kf=np.array(Kins)
for kap in [None,0.42]:
    hook=None if kap is None else make_oracle(Ro,DGi_o,G_from_K(Kf,kap))
    t0_=time.time(); res_o=run_rollout_h(Ro,W,arms_o,n_paths=20,seed=3,hook_P=hook); el=time.time()-t0_
    line=[]
    for a,(lo_,hi_,t0a) in zip([1,2,3,4],[(13,24,12),(13,24,12),(13,24,12),(17,24,16)]):
        rows,reached,o=readout(Ro,res_o,arms_o,lo=lo_,hi=hi_,t0=t0a)
        pred=np.nanmean(o[a]-o[0],1); tru=truth_rows(Ro,DGi_o,a,rows,Kf)
        line.append(f'arm{a}: model {np.nanmean(pred[reached]):6.2f} truth {tru[reached].mean():6.2f}')
    print('kappa',kap,'%.0fs'%el,' | '.join(line))
print('G(0.42)',np.round(G_from_K(Kf,0.42),2))
```
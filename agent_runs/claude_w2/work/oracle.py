
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

def make_dev_oracle(R,DGi,Gf,t_hide=13.0,pf=None):
    A=R['A']; code=A.code.values; tt=A.t.values; val=A.value.values; loc=A['loc'].values; S_=len(R['stays'])
    GFI=np.full((S_,32),-1); GLI=np.full((S_,32),-1)
    for cc,arr in [(CODE['glucose_finger'],GFI),(CODE['glucose_lab'],GLI)]:
        m=np.where((code==cc)&(np.abs(tt-np.round(tt))<1e-6))[0]; arr[A.si.values[m],np.round(tt[m]).astype(int)+3]=m
    def gval(n,j,drawn):
        v=val[j].astype(float).copy(); hid=tt[j]>=t_hide
        if hid.any(): b=drawn[n[hid],loc[j[hid]]]; v[hid]=CENT[code[j[hid]],np.maximum(b,1)-1]
        return v
    def devs(n,si,a,tau,drawn,L):
        D=np.zeros((len(n),L))
        for kk in range(1,L+1):
            ix=tau-kk+3; ok=ix>=0; ixc=np.clip(ix,0,31)
            d=DGi[a,si,ixc]; jf=GFI[si,ixc]; jl=GLI[si,ixc]; g=np.full(len(n),np.nan)
            m1=ok&(jf>=0); g[m1]=gval(n[m1],jf[m1],drawn); m2=ok&(jf<0)&(jl>=0); g[m2]=gval(n[m2],jl[m2],drawn)
            p=np.where(np.isnan(g),0.0,pf(np.nan_to_num(g))); D[:,kk-1]=np.where(ok,d-p,0.0)
        return D
    def hook(k,hn_,hj,P,ctx):
        m=np.isin(code[hj],GLC)
        if not m.any(): return P
        n=hn_[m]; si=ctx['seq_si'][n]; a=ctx['seq_arm'][n]; tau=np.round(tt[hj[m]]).astype(int)
        r=devs(n,si,a,tau,ctx['drawn'],len(Gf))@Gf
        P=P.clone(); ii=torch.as_tensor(np.where(m)[0])
        P[ii]=tilt_P(P[ii],torch.as_tensor(CENT[code[hj[m]]],dtype=torch.float32),r); return P
    hook.devs=devs; return hook


def make_lin_oracle(R,DGi=None,a_i=None,c_i=None,k_i=0.0,pf=None,DGn=None,a_n=None,k_n=0.0,t_hide=13.0,gain_i=1.0,gain_n=1.0):
    A=R['A']; code=A.code.values; tt=A.t.values; val=A.value.values; loc=A['loc'].values; S_=len(R['stays'])
    GFI=np.full((S_,32),-1); GLI=np.full((S_,32),-1)
    for cc,arr in [(CODE['glucose_finger'],GFI),(CODE['glucose_lab'],GLI)]:
        m=np.where((code==cc)&(np.abs(tt-np.round(tt))<1e-6))[0]; arr[A.si.values[m],np.round(tt[m]).astype(int)+3]=m
    def gval(n,j,drawn):
        v=val[j].astype(float).copy(); hid=tt[j]>=t_hide
        if hid.any(): b=drawn[n[hid],loc[j[hid]]]; v[hid]=CENT[code[j[hid]],np.maximum(b,1)-1]
        return v
    def plags(n,si,tau,drawn,L):
        Pm=np.zeros((len(n),L))
        for kk in range(1,L+1):
            ix=tau-kk+3; ok=ix>=0; ixc=np.clip(ix,0,31)
            jf=GFI[si,ixc]; jl=GLI[si,ixc]; g=np.full(len(n),np.nan)
            m1=ok&(jf>=0); g[m1]=gval(n[m1],jf[m1],drawn); m2=ok&(jf<0)&(jl>=0); g[m2]=gval(n[m2],jl[m2],drawn)
            Pm[:,kk-1]=np.where(np.isnan(g),0.0,pf(np.nan_to_num(g)))
        return Pm
    def hook(k,hn_,hj,P,ctx):
        P=P.clone()
        for codes_,drug in [(GLC,'i'),(MPC,'n')]:
            DG=DGi if drug=='i' else DGn
            if DG is None: continue
            m=np.isin(code[hj],codes_)
            if not m.any(): continue
            n=hn_[m]; si=ctx['seq_si'][n]; a=ctx['seq_arm'][n]; tau=np.round(tt[hj[m]]).astype(int)
            if drug=='i':
                d=lag_abs(DG,a,si,tau,len(a_i)); r=d@np.asarray(a_i)+k_i
                if c_i is not None: r=r+plags(n,si,tau,ctx['drawn'],len(c_i))@np.asarray(c_i)
                r=gain_i*r
            else:
                d=lag_abs(DG,a,si,tau,len(a_n)); r=gain_n*(d@np.asarray(a_n)+k_n)
            ii=torch.as_tensor(np.where(m)[0])
            P[ii]=tilt_P(P[ii],torch.as_tensor(CENT[code[hj[m]]],dtype=torch.float32),r)
        return P
    hook.plags=plags; return hook

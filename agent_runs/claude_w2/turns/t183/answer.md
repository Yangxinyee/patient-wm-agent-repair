Two problems. First, the policy table is wrong above about 290. Per-mg/dL counts there are under 30, so the forward fill froze at 3.39, while the true rate above 300 is about 6.5. I will look at the raw means around the upper bands to find the edges, then use band-wise means that respect the edges. Second, the deviation-oracle code was again never defined because the earlier run died first. This time I define everything in one run: print the raw means, rebuild the policy as a band-piecewise lookup, save the oracle code, and rerun the NLL and rollout tests.

```python
r_=tabpol.loc[240:330]; 
print(' '.join(f'{i}:{r_.loc[i,"mean"]:.2f}({int(r_.loc[i,"count"])})' for i in range(240,331,3) if i in r_.index))
code_pf3 = r'''
def make_pf_band(tab,edges=(150,180,220,250,260,300),minc=200):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs); c=tab['count'].reindex(xs).fillna(0).values; mv=np.nan_to_num(m.values)
    out=np.zeros(601); E=list(edges)+[601]
    for i in range(150,601):
        b=max(j for j in range(len(E)-1) if E[j]<=i); lo_b,hi_b=E[b],E[b+1]-1
        w=0
        while True:
            lo,hi=max(lo_b,i-w),min(hi_b,i+w)
            if c[lo:hi+1].sum()>=minc or (lo==lo_b and hi==hi_b): break
            w+=1
        out[i]=(mv[lo:hi+1]*c[lo:hi+1]).sum()/max(c[lo:hi+1].sum(),1)
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,out)
    return pf,out
'''
open('/work/policy.py','a').write(code_pf3); exec(code_pf3)
code_dor = r'''
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
'''
open('/work/oracle.py','a').write(code_dor); exec(code_dor)
pfv,pftab=make_pf_band(tabpol)
for x in [149,150,165,179,180,200,219,220,249,250,259,260,275,299,300,320,400]: print(x,round(float(pfv(x)),2),end='; ')
print()
Gpol['p']=pfv(Gpol.g.values); print('policy resid sd (g>=150):',(Gpol.rate-Gpol.p)[Gpol.g>=150].std())
for nm_,hk in [('none',None),('dev-oracle',make_dev_oracle(Ro,DGi_o[:1],Gp,t_hide=1e9,pf=pfv))]:
    rr=run_rollout_h(Ro,W,[{}],n_paths=1,seed=0,t_hide=1e9,want_nll=True,hook_P=hk)
    A_=Ro['A']; ks=np.array(list(rr['nll'].keys())); js=ks%10**7; vv=np.array(list(rr['nll'].values()))
    cc=A_.code.values[js]; ok=~np.isnan(A_.value.values[js])
    print(nm_,'NLL all %.5f  glucose %.5f  map %.5f  n=%d'%(vv[ok].mean(),vv[ok&np.isin(cc,GLC)].mean(),vv[ok&np.isin(cc,MPC)].mean(),ok.sum()))
for nm_,hk in [('diff-oracle',make_oracle(Ro,DGi_o,Gp)),('dev-oracle',make_dev_oracle(Ro,DGi_o,Gp,pf=pfv))]:
    res_o=run_rollout_h(Ro,W,arms_o,n_paths=20,seed=3,hook_P=hk); line=[]
    for a,(lo_,hi_,t0a) in zip([1,2,3,4],[(13,24,12),(13,24,12),(13,24,12),(17,24,16)]):
        rows,reached,o=readout(Ro,res_o,arms_o,lo=lo_,hi=hi_,t0=t0a)
        pred=np.nanmean(o[a]-o[0],1); tru=truth_rows(Ro,DGi_o,a,rows,Kf)
        line.append(f'a{a}: {np.nanmean(pred[reached]):6.2f}/{tru[reached].mean():6.2f}')
    print(nm_,' | '.join(line))
```
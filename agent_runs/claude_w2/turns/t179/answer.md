The oracle shows the correction mechanism works but overshoots. Adding the innovation filter on top of the model gives x2 −21.4 vs truth −16.5, x½ +10.1 vs +8.3, x0 +17.5 vs +16.5, and x2 from hour 16 −17.0 vs −14.2. The excess is about the model's own response (−3.8, +2.0, +2.3, −2.7 without the oracle), which simply adds on top. So the target per-step correction must be G′ = G(κ=0.42) minus the model's own teacher-forced response (−0.26, −0.11, −0.06, …). That is the T = S − M logic from earlier.

A real weight edit can only see deviations dev = dose − p(glucose seen by the policy), not the true difference between arms. So I also test a deviation-based oracle that applies G′ to dev computed from the drawn readings via my policy function. This also gives the factual NLL change with an exact deviation signal, which is an upper bound on what a channel could gain. Rollouts matching the truth for both oracles, plus a factual NLL improvement, would confirm the target. Next I run that check.

```python
print('pfun test',[pfun(x) for x in [100,155,165,200,240,270,310]])
try: print('vectorized',pfun(np.array([100.,165.,240.])))
except Exception as ex: print('not vectorized',ex)
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
pfv=np.vectorize(pfun)
b_model=np.array([-0.26,-0.11,-0.06,-0.02,-0.01,-0.01,0.01,0,0,0,-0.01,0])
Gp=G_from_K(Kf,0.42)-b_model; print('Gprime',np.round(Gp,2))
# (1) factual NLL with dev oracle
for nm_,hk in [('none',None),('dev-oracle',make_dev_oracle(Ro,DGi_o[:1],Gp,t_hide=1e9,pf=pfv))]:
    rr=run_rollout_h(Ro,W,[{}],n_paths=1,seed=0,t_hide=1e9,want_nll=True,hook_P=hk)
    A_=Ro['A']; ks=np.array(list(rr['nll'].keys())); js=ks%10**7; vv=np.array(list(rr['nll'].values()))
    cc=A_.code.values[js]; ok=~np.isnan(A_.value.values[js])
    print(nm_,'NLL all %.5f  glucose %.5f  map %.5f  n=%d'%(vv[ok].mean(),vv[ok&np.isin(cc,GLC)].mean(),vv[ok&np.isin(cc,MPC)].mean(),ok.sum()))
# (2) rollouts with dev oracle and diff oracle
for nm_,hk in [('diff-oracle',make_oracle(Ro,DGi_o,Gp)),('dev-oracle',make_dev_oracle(Ro,DGi_o,Gp,pf=pfv))]:
    res_o=run_rollout_h(Ro,W,arms_o,n_paths=20,seed=3,hook_P=hk); line=[]
    for a,(lo_,hi_,t0a) in zip([1,2,3,4],[(13,24,12),(13,24,12),(13,24,12),(17,24,16)]):
        rows,reached,o=readout(Ro,res_o,arms_o,lo=lo_,hi=hi_,t0=t0a)
        pred=np.nanmean(o[a]-o[0],1); tru=truth_rows(Ro,DGi_o,a,rows,Kf)
        line.append(f'a{a}: {np.nanmean(pred[reached]):6.2f}/{tru[reached].mean():6.2f}')
    print(nm_,' | '.join(line))
```
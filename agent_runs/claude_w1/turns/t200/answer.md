Two clear findings.
- **Gate is cheap.** With z-noise included, glucose and MAP rows remain hugely separable (d′ ≈ 177), because the code-specific terms in with_code don't depend on z.
- **Dose signal at the head is weak and partly nonlinear.** Doubling insulin from hour 12 moves with_code by only about 0.08 in norm on average, for a mean lag-1 change of 1.64 u/h. A per-row linear map explains only 22% of the dose change under x2 but 66% under x1/2. NE x2, which reaches dose bins never seen in training, moves x 3.7 times more than NE x1/2 for only twice the dose change.

So I'll fit a minimum-variance decoder per drug with these constraints:
- its mean response to each of the two edits equals the mean lag-1 dose change;
- zero mean response to the other drug's edits;
- a large gate offset between glucose and MAP rows.

It minimises factual within-(code, last bin) variance plus z-draw variance plus a per-row response-fidelity term. The key number is the decoded-dose noise sd on factual rows. For insulin it must stay well below about 0.5 u/h (the ridge decoders' level); for NE it must fall well below the ridge 0.18 for the NE correction to be affordable. I'll also rank the head's hidden units by how much they contribute, to choose which two to repurpose.

```python
def cell_resid(X_, c_, lb_):
    R_=X_.copy(); key_=c_*17+lb_
    for k_ in np.unique(key_): m_=key_==k_; R_[m_]-=X_[m_].mean(0)
    return R_
Rf=cell_resid(Xf,cf,lbf)
def solve_unit(rows_obj, wts, out_code, own, other, M_gate, lam=1e-2, mu=0.0, sign_g=1):
    Ro=Rf[rows_obj]*np.sqrt(wts)[:,None]; S_=Ro.T@Ro/wts.sum()
    Dz=DQf[rows_obj].astype(np.float64); S_+= np.einsum('nki,nkj->ij',Dz,Dz)/(2*len(rows_obj))
    S_+= lam*np.eye(192); cons=[]; f_=[]
    for key in own:
        e_=EDR[key]; oc=(cf[e_['gi']]==out_code)&e_['reached']; cons.append(e_['dX'][oc].mean(0)); f_.append(e_['dd'][oc,0].mean())
        if mu>0: S_+= mu*(e_['dX'][oc].T@e_['dX'][oc])/oc.sum()   # fidelity (quadratic part)
    for key in other:
        e_=EDR[key]; oc=np.isin(cf[e_['gi']], [0,1] if out_code<2 else [2,3]); cons.append(e_['dX'][oc].mean(0)); f_.append(0.0)
    cons.append(sign_g*(mg_-mm_)); f_.append(M_gate)
    C_=np.array(cons).T; SiC=np.linalg.solve(S_,C_); v_=SiC@np.linalg.solve(C_.T@SiC, np.array(f_))
    return v_
def diag(v_, rows_obj, out_code, keys):
    sdw_=(Rf[rows_obj]@v_).std(); sdz_=(DQf[rows_obj,0,:].astype(np.float64)@v_).std()
    rs=[]
    for key in keys:
        e_=EDR[key]; oc=(cf[e_['gi']]==out_code)&e_['reached']; resp=e_['dX'][oc]@v_
        rs.append((key[0][:3]+str(key[1]), round(resp.mean()/e_['dd'][oc,0].mean() if abs(e_['dd'][oc,0].mean())>0 else resp.mean(),3)))
    return round(sdw_,3), round(sdz_,3), rs
ins_rows=np.where(isg)[0]; w_ins=np.where(cf[ins_rows]==0,1/14.9**2,1/12.0**2)
keysI=[('insulin',2.0),('insulin',0.5)]; keysN=[('norepinephrine',2.0),('norepinephrine',0.5)]
for lam in [1e-3,1e-2,1e-1]:
    vI=solve_unit(ins_rows, w_ins, 0, keysI, keysN, 20.0, lam=lam)
    print('INS lam',lam, diag(vI, ins_rows, 0, keysI+keysN))
map_rows=np.where(ism)[0]; w_ne=np.where(cf[map_rows]==2,1/5.2**2,1/7.9**2)
for lam in [1e-3,1e-2,1e-1]:
    vN=solve_unit(map_rows, w_ne, 2, keysN, keysI, 20.0, lam=lam, sign_g=-1)
    print('NE lam',lam, diag(vN, map_rows, 2, keysN+keysI))
```
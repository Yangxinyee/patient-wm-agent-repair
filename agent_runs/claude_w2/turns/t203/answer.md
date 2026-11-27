The attenuation was my ridge penalty. With a small alpha the rate decodes from `out` with arm-change slopes of 0.98, 1.0, 0.65, 0.75, 1.11 and 0.67, the same as the full state. At the head (with_code) the lag-1 dose reaches 0.78, 1.0, 0.65, 0.57, 0.88 and 0.56. That is good enough to respond to dose edits; x0 and x3 are weaker because rate 0 and large rates sit outside the training range.

The catch is noise. A low-penalty decoder is very noisy on factual rows (R² down to -2). A head-only tilt would therefore cost factual fit unless the decoder is fitted with that cost in mind.

So I'll fit one joint decoder w on with_code with two kinds of rows:
- **Factual rows:** the target is the model's observed residual (reading minus predicted mean). This is the dose-noise effect the model misses, so fitting it should improve factual NLL.
- **Counterfactual teacher-forced rows:** the target is the arm-minus-recorded change G·Δdose, where G is the direct kernel from the oracle calibration (kap = 0.85), and the inputs are the matching changes in with_code.

For each weight λ on the counterfactual rows I'll report, cross-validated by stay:
- the factual value NLL change from applying the tilt as an exact mean shift (glucose rows only);
- the per-arm response slope.

A λ with slopes near 1 for x2 and x1/2 and an NLL change ≤ 0 would make a head-only edit worth implementing. If every λ that gives slope ≈ 1 costs more than about 0.01 nats on glucose rows, I'll build an upstream channel instead.

```python
G85=G_from_K(Kins,0.85)
codes_=DD['code'][gidx]; vals_=DD['value'][gidx]
obD=np.zeros(len(gidx),int)
for c in GLC:
    m=codes_==c; edges=np.array(NUMD[(CNAME[c],'value')]['fine_bin_edges'])
    obD[m]=np.where(np.isnan(vals_[m]),0,1+np.searchsorted(edges,np.nan_to_num(vals_[m]),side='left'))
pmD=DD['pm'][gidx]; PD=DD['P'][gidx]
f0=np.where((aD==0)&~np.isnan(vals_))[0]
# cf pairs
base_idx=pd.Series(np.arange(len(gidx))[aD==0],index=pd.MultiIndex.from_arrays([DD['si'][gidx][aD==0],DD['j'][gidx][aD==0]]))
pairs=[]
for a in range(1,6):   # sustained arms only (exclude random-mult arm 6)
    ia=np.where(aD==a)[0]; ib=base_idx.reindex(pd.MultiIndex.from_arrays([DD['si'][gidx][ia],DD['j'][gidx][ia]])).values
    ok=~np.isnan(ib); pairs.append(np.c_[ia[ok],ib[ok].astype(int),np.full(ok.sum(),a)])
pairs=np.vstack(pairs)
tc=((dD[pairs[:,0]]-dD[pairs[:,1]])@G85)
Xf=np.c_[Xw[f0],np.ones(len(f0))]; yf=(vals_-pmD)[f0]
Xc=np.c_[Xw[pairs[:,0]]-Xw[pairs[:,1]],np.zeros(len(pairs))]
cenD=torch.as_tensor(CENT[codes_],dtype=torch.float32)
def nll_of(P,ob): return -np.log(np.clip(P[np.arange(len(ob)),ob].numpy(),1e-12,None))
nll0=nll_of(PD[f0],obD[f0]).mean()
fold=(DD['si'][gidx]%2)
print('base nll glucose rows %.4f n=%d, n pairs %d'%(nll0,len(f0),len(pairs)))
for lam in [0.0,0.03,0.3,3.0]:
    rf=np.zeros(len(f0)); rc=np.zeros(len(pairs))
    for fo in [0,1]:
        trf=fold[f0]!=fo; trc=fold[pairs[:,0]]!=fo
        A_=Xf[trf].T@Xf[trf]+lam*Xc[trc].T@Xc[trc]+1e-2*np.eye(193); b_=Xf[trf].T@yf[trf]+lam*Xc[trc].T@tc[trc]
        w_=np.linalg.solve(A_,b_)
        rf[~trf]=Xf[~trf]@w_; rc[~trc]=Xc[~trc]@w_
    Pn=tilt_P(PD[f0],cenD[f0],rf); dn=nll_of(Pn,obD[f0]).mean()-nll0
    sl=[round(float((rc[pairs[:,2]==a]*tc[pairs[:,2]==a]).sum()/(tc[pairs[:,2]==a]**2).sum()),2) for a in range(1,6)]
    print('lam',lam,'dNLL glucose %.4f'%dn,'tilt sd fact %.2f'%rf.std(),'cf slopes',sl)
```
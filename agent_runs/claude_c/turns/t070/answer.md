Run 62 gives a lever I had not seen. At reading-only bundles, z26 is an exact linear function of the other z dimensions (R² = 1.000). At drug bundles, by contrast, the other dimensions are nearly constant and only z26 moves. At insulin-only bundles z26 rises about 1.43 per U/h of insulin rate (R² 0.53). At norepinephrine-only bundles it rises about 2.4 per mcg/kg/min.

So there should be a direction v in z-space with v·z ≈ 0 at every reading bundle, including the mean, and a large weight on z26. Reshaping `z_in` along such a direction would change the state update only at drug bundles. Reading bundles, which carry most of the factual fit, would be left alone.

Next I will check this structure carefully:
- the singular values of the reading-bundle z matrix, uncentered;
- the null directions, and the projection of e26 onto them;
- how well that direction cancels on held-out stays, and what it gives at each kind of drug bundle.

A tiny held-out v·z at reading bundles, together with a large spread at drug bundles, would confirm a selective lever. A noticeable held-out residual would refute it.

```python
Zr=zo_[mr]; Zd=zo_[hi_|hn_]
U_,S_,Vt_=np.linalg.svd(Zr,full_matrices=False)
print('uncentered sv:',np.round(S_,4))
Uc,Sc,Vtc=np.linalg.svd(Zr-Zr.mean(0),full_matrices=False)
print('centered sv:',np.round(Sc,4))
# null space: right singular vectors with tiny sv (uncentered)
thr=1e-3*S_[0]
nullV=Vt_[S_<thr]
print('null dim (uncentered, thr %.4f):'%thr, nullV.shape[0])
# projection of e26 onto null space
e26=np.zeros(32); e26[26]=1
v_=nullV.T@(nullV@e26)
print('|proj e26| = %.4f, v26=%.4f'%(np.linalg.norm(v_), v_[26]))
u_=v_/np.linalg.norm(v_)
# held-out check: build null space from even stays, test on odd stays
sids_r=sid_[mr]; even=(sids_r%2==0)
U2,S2,Vt2=np.linalg.svd(Zr[even],full_matrices=False)
nv2=Vt2[S2<1e-3*S2[0]]; v2=nv2.T@(nv2@e26); u2=v2/np.linalg.norm(v2)
print('held-out: null dim %d; |u2.z| on odd reading rows: mean %.2e max %.2e; on drug rows sd %.3f'%(nv2.shape[0],np.abs(Zr[~even]@u2).mean(),np.abs(Zr[~even]@u2).max(),(Zd@u2).std()))
print('u.z at reading: mean %.2e max %.2e'%(np.abs(Zr@u_).mean(), np.abs(Zr@u_).max()))
for k_,msk in [('ins_only',grp_['ins_only']),('ne_only',grp_['ne_only']),('both',grp_['both'])]:
    pz=zo_[msk]@u_
    c=np.corrcoef(pz,zo_[msk,26])[0,1]
    b1=np.polyfit(zo_[msk,26],pz,1)
    print(k_, 'u.z mean %.3f sd %.3f corr w z26 %.4f; u.z = %.4f*z26 + %.4f -> z26_ref %.3f'%(pz.mean(),pz.std(),c,b1[0],b1[1],-b1[1]/b1[0]))
# also check other-dim means at drug bundle types
for k_,msk in [('ins_only',grp_['ins_only']),('ne_only',grp_['ne_only']),('both',grp_['both'])]:
    print(k_, 'max sd of non-26 dims %.4f'%np.delete(zo_[msk],26,axis=1).std(0).max())
```
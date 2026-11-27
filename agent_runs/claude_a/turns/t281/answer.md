The write side is cheap, but the memory is wrong: its correlation with the ideal decayed dose sum is only 0.13. The staging dim is correctly zero in the state, and so is the reference. So the writes themselves are corrupted. The likely cause is leakage. Prelude head 3 also attends to the measurement token (weight about 0.6 at glucose bundles) and to the other tokens. Those tokens still carry their original content in dims 79 and 99. The two dims were not equalised there, so their difference is written into memory at every bundle. To check, I'll compute the implied write at each bundle (memory now minus memory before, decayed by the 2.5 h half-life). I'll regress it on the start rate and on the measurement-token difference e₇₉ − e₉₉. If the e-difference explains most of the spurious writes, the fix is to equalise those two dims in the measurement encoder's output (and in the latent-input token) as was done for the dose encoder.

```python
dt_=np.r_[0,np.diff(tsec)]; same=np.r_[False, sid[1:]==sid[:-1]]
prevm=np.r_[0,mem[:-1]]*np.exp(-lam*dt_)*same; w=mem-prevm
e2=np.asarray(ds_s2['e']); ediff=e2[:,79]-e2[:,99]; u2=np.asarray(ds_s2['u']); udiff=u2[:,79]-u2[:,99]
X_=np.vstack([yk2, ediff, np.ones_like(w)]).T; cw=np.linalg.lstsq(X_,w,rcond=None)[0]; rw=w-X_@cw
print('write ~ rate %.3f, ediff %.3f, c %.3f; R2 %.3f resid sd %.3f'%(cw[0],cw[1],cw[2],1-rw.var()/w.var(),rw.std()))
ins_=yk2>0; nod=(yk2==0)
print('write at ins starts: mean %.2f sd %.2f (rate mean %.2f) | at non-start bundles: mean %.2f sd %.2f'%(w[ins_].mean(), w[ins_].std(), yk2[ins_].mean(), w[nod].mean(), w[nod].std()))
print('ediff sd %.3f, udiff at starts corr w/ rate %.3f'%(ediff.std(), np.corrcoef(udiff[ins_],yk2[ins_])[0,1]))
```
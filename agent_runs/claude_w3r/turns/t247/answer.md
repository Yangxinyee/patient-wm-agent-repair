Results:
- My DeepSet encoder now reproduces the dose tokens exactly (error ~1e-6), so the ×2 and ×½ tokens are now correct.
- Linear rate reads still under-respond to doubling. Insulin slope is 0.72 under ×2 and 1.10 under ×½; norepinephrine 0.85 and 1.21.
- The insulin-presence gate separates only weakly (margin 0.08).
- The "any dose" gate stays clean (margin about 0.8).

**Problem with the plan so far:** an ungated rate read leaves sizeable garbage in the slot residual on no-dose bundles. That would disturb the model unless I cancel it.

**Cleaner alternative to test:** find a direction ω in head2's input space with:
- ω·n1 ≈ 0 for every real token;
- ω·β = 1 for the empty dose token, which is exactly the norm bias β.

Key and value terms built on ω would touch only the empty dose token. Head2 would then attend it on no-dose bundles, so my read dims there are exactly constant. Its value can be set to head2's average no-dose output, and dose bundles stay untouched.

**Test:**
- The ω residual on real tokens must be tiny: rms ≲ 0.005.
- The spread of head2's original no-dose output around its mean must be small compared with the slot scale (about 58).

Both small would confirm the design. Otherwise I fall back to FFN gating with cancellation.

```python
act_b=np.abs(n1[:,5,:]-b1).max(1)>1e-3
real=np.vstack([n1[:,[0,1,2,3,4,6],:].reshape(-1,192), n1[act_b,5,:]])
C2=real.T@real/len(real)
for reg in [1e-10,1e-8,1e-6]:
    w_=np.linalg.solve(C2+reg*np.eye(192),b1); om=w_/(b1@w_); r_=real@om
    print('reg %.0e omega |om| %.1f real-proj rms %.2e max %.2e'%(reg,np.linalg.norm(om),np.sqrt((r_**2).mean()),np.abs(r_).max()))
Wq=Wt['core.prelude.0.attn.q.weight'].numpy(); Wq2=Wq[96:144]; Wk2_=kv[96:144]
nb=np.where(~act_b)[0]; print('no-dose bundles', len(nb), 'dose bundles', int(act_b.sum()))
O_no=H_[nb][:,:,2,:]@Wo2.T; O_act=H_[act_b][:,:,2,:]@Wo2.T
mno=O_no.mean(0); dev=np.linalg.norm(O_no-mno,axis=-1)
print('head2 out no-dose: |mean| %.2f, dev rms %.3f p99 %.3f | dose-bundle |mean| %.2f, |mean_act-mean_no| %.2f'%(np.linalg.norm(mno,axis=-1).mean(),np.sqrt((dev**2).mean()),np.percentile(dev,99),np.linalg.norm(O_act.mean(0),axis=-1).mean(),np.linalg.norm(O_act.mean(0)-mno,axis=-1).mean()))
Q=np.einsum('bqd,hd->bqh',n1[nb][:,:4,:],Wq2); Kt=np.einsum('bjd,hd->bjh',n1[nb],Wk2_)
S_=np.einsum('bqh,bjh->bqj',Q,Kt)/np.sqrt(48); gap=S_[:,:,[0,1,2,3,4,6]].max(-1)-S_[:,:,5]
fq_=Q.reshape(-1,48).mean(0); fq_/=np.linalg.norm(fq_); qf=(Q.reshape(-1,48)@fq_)
print('no-dose score gap (max real - u0): mean %.2f max %.2f | q.f min %.2f mean %.2f'%(gap.mean(),gap.max(),qf.min(),qf.mean()))
```
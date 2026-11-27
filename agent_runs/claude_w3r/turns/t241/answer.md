The "any dose token present" read is excellent:
- residual 0.012 on dose tokens;
- non-dose tokens at most 0.07;
- the empty dose token (no dose) reads 0.005.

So the gate signal is clean. The rate reads are only moderate, even with almost no limit on non-dose tokens: residual 0.15 for insulin (sd 1.15) and 0.14 for norepinephrine (sd 0.33). The normalisation layer in front of the attention bends the rate encoding.

**Conclusion:** the gated design is viable for insulin. For norepinephrine the rate read is noisier.

**Next checks, before committing:**
1. Do the rate reads extrapolate correctly when the dose is doubled or halved? That is what the evaluation does. I will use my `u_of` encoder on a grid of rates and check `n1(u_of)` against the stored tokens.
2. How cheaply can head2's 48-dim value bottleneck give up 3 dimensions for the new reads? This is the singular spectrum of head2's out-projection weighted by its value second moment.
3. Confirm the kv row layout by rebuilding head2's outputs.

**Test:** reads roughly linear in the rate up to ×2, and the 3 weakest value directions carrying under ~1% of head2's output energy, would confirm the design.

```python
g1=Wt['core.prelude.0.norm1.weight'].numpy(); b1=Wt['core.prelude.0.norm1.bias'].numpy()
def n1_of(u):
    u=np.asarray(u,float); m=u.mean(-1,keepdims=True); s=u.var(-1,keepdims=True)
    return g1*(u-m)/np.sqrt(s+1e-5)+b1
uu=u_of(float(pa[0,0]),float(pa[0,1])); uu=uu.numpy() if hasattr(uu,'numpy') else np.asarray(uu)
print('u_of type/shape', type(uu), uu.shape, 'match U_act[0]:', np.abs(n1_of(uu.reshape(-1))-U_act[0]).max())
cI=ridge_read(Xu,pa[:,0],Xn,lam_n=1e-3)[0]; cN=ridge_read(Xu,pa[:,1],Xn,lam_n=1e-3)[0]
rd=lambda c,x: x@c[:-1]+c[-1]
def R(ins,ne):
    u=u_of(float(ins),float(ne)); u=u.numpy() if hasattr(u,'numpy') else np.asarray(u); x=n1_of(u.reshape(-1))
    return rd(cI,x), rd(cN,x), rd(c_any,x)
print('ins grid (ne=0):', [(i, *np.round(R(i,0.0),2)) for i in [0.5,1,1.5,2,3,4,6,8]])
print('ins grid (ne=.5):', [(i, *np.round(R(i,0.5),2)) for i in [0.5,1,2,4,8]])
print('ne grid (ins=0):', [(n, *np.round(R(0.0,n),2)) for n in [0.1,0.25,0.5,0.75,1,1.5,2,3]])
print('ne grid (ins=2):', [(n, *np.round(R(2.0,n),2)) for n in [0.25,0.5,1,2]])
kv=Wt['core.prelude.0.attn.kv.weight'].numpy(); Wo=Wt['core.prelude.0.attn.out.weight'].numpy()
Wv2=kv[192+96:192+144]; Wo2=Wo[:,96:144]
V2=np.einsum('bqj,bjd->bqd',patt[:,2,:4,:],n1)@Wv2.T
print('H_ head2 rebuild err', np.abs(V2-H_[:,:,2,:]).max(), '| H_ scale', np.abs(H_[:,:,2,:]).mean())
v=H_[:,:,2,:].reshape(-1,48); S=v.T@v/len(v); L=np.linalg.cholesky(S+1e-9*np.eye(48))
sv_=np.linalg.svd(Wo2@L,compute_uv=False); en=sv_**2
print('head2 out-energy sv (top5, bottom5):', np.round(sv_[:5],3), np.round(sv_[-5:],4), 'bottom3 frac %.2e'%(en[-3:].sum()/en.sum()))
```
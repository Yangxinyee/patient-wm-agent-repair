Two results from run 268:
- **Gate:** constraint values (in-range bins −1.5, insulin code −8, END −4) cut the gate's bf16 error to 0.06 at κ ≈ 2–3, while keeping the margins.
- **Correction:** the fit is ill-conditioned. With nearly collinear constant and linear bases, the output vectors grow to norm 25–50 against corrections of at most 17.6, so small bf16 errors in each unit get amplified.

Better-separated bases with larger output ranges (rising and falling ramps plus a hinge) should need much smaller output weights. In this run I compare three basis designs by fit error and by a full bf16 emulation of the edited mlp1 on bin-16 atoms, against the natural bf16 error of in-range atoms. I then build the best one with the low-error gate, check that factual insulin/NE/END encodings are unchanged and that bin-16 encodings hit the target, and save it. If the excess bf16 error is comparable to natural noise and factual Δu is under 0.005, it is ready for a dose_edit test.

```python
wG2=gate(-1.5,-8,-4)
def bfr16(x): return torch.tensor(np.asarray(x),dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy().astype(np.float64)
def m1_bf(A_,W0,b0,W2,b2):
    p=bfr16(bfr16(A_)@bfr16(W0).T+bfr16(b0)); h=bfr16(gel(p)); return bfr16(h@bfr16(W2).T+bfr16(b2))
def build2(spec,kap,units=(674,411,314,524),beta=1.0):
    Mt=m10+beta*(rp-1)[:,None]*dm[None]; C2=Mt-Mo; wt=np.ones(len(rp)); wt[-20:]=1; 
    Phi=np.array([gel(c+b*(zp_-zm)) for c,b in spec]).T; V=np.linalg.lstsq(Phi,C2,rcond=None)[0]
    W0n=W10.copy(); b0n=b10.copy(); W2n=W12.copy(); b2n=b12o.copy()
    for (c,b),j,vj in zip(spec,units,V): b2n+=W12[:,j]*act[:,j].mean(); W0n[j]=kap*wG2+b*wZ; b0n[j]=-kap+c-b*zm; W2n[:,j]=vj
    return W0n,b0n,W2n,b2n,Mt,np.abs(Phi@V-C2).max(),np.linalg.norm(V,axis=1)
At16=np.array([aNE(r) for r in (1.02,1.2,1.5,1.8,2.0)]); Mt16=m10+((np.array([1.02,1.2,1.5,1.8,2.0])-1)[:,None])*dm[None]
nat=np.abs(m1_bf(np.array(NEA[5:40:5]),W10,b10,W12,b12o)-m1(np.array(NEA[5:40:5]))).max()
best=None
for spec,kap in [([(2,2),(2,-2),(0,4)],3.0),([(4,4),(4,-4),(0,3)],5.0),([(2,2),(2,-2),(0,4),(0,-4)],4.0)]:
    W0n,b0n,W2n,b2n,Mt,fe,vn=build2(spec,kap,units=(674,411,314,524)[:len(spec)])
    eb=np.abs(m1_bf(At16,W0n,b0n,W2n,b2n)-Mt16).max(); pmx=max(float((np.array(INSU+NEA[:40]+ENDS)@W0n[j]+b0n[j]).max()) for j in (674,411,314,524)[:len(spec)])
    print(spec,'kap',kap,'fit',round(fe,3),'|V|',vn.round(1),'bf16 err vs target',round(eb,3),'(natural',round(nat,3),') max pre nontarget',round(pmx,2))
    if pmx<-4 and (best is None or max(fe,eb)<best[0]): best=(max(fe,eb),spec,kap,(W0n,b0n,W2n,b2n))
print('chosen',best[1],best[2]); W0n,b0n,W2n,b2n=best[3]
un=lambda a,b=None: uenc2([a,aE],W0=W0n,b0=b0n,W2=W2n,b2=b2n); uo=lambda a: uenc2([a,aE],W0=W10,b0=b10,W2=W12,b2=b12o)
print('factual NE du',round(max(np.abs(un(a)-uo(a)).max() for a in NEA[:40:3]),4),'ins/END du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in INSU[::5]+ENDS),4))
print('bin16 u err vs target', [round(float(np.abs(un(aNE(r))-u_from_m(m10+(r-1)*dm)).max()),3) for r in (1.02,1.5,2.0)], '| u move r=1->2 (target)', round(float(np.linalg.norm(u_from_m(m10+dm)-u_from_m(m10))),2))
```
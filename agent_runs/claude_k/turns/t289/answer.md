This run sets the scale limits. The LayerNorm sphere makes the head-1 injection nonlinear. For glucose the response is roughly linear up to about ±10 mg/dL-equivalent per drug bundle (gain 0.79 to 0.91). At ±40 the gain falls to 0.4, and z can lose head 1 again. For MAP the direction is costlier (|η| = 4.6 per mmHg) and stays usable to about ±5 mmHg. Typical per-hour kicks should mostly fall in those ranges, with high rates and ×3 saturating softly, so I'll build the edit and measure.

The edit:
- **z_post hidden units.** Keep the exact gate and innovation units (12, 88, 105). Add four gated END-pattern indicator units, built from unit 12's gate plus a linear pattern score on e. These repurpose the four units with the smallest mean-output weights, which is free because z is ignored.
- **z_post outputs.** Route them into 6 z dimensions: a pattern one-hot plus the insulin and NE innovations, and zero out every other use of these units.
- **z_in.** Map the pattern dims to 300 × (designed pattern vector) and the innovation dims to −κ_A·η_G and +κ_B·η_M. Initial gains are κ_A = 3 mg/dL-eq per insulin unit and κ_B = 6 mmHg-eq per NE unit.

I'll verify offline before saving:
- pattern units at +8 for their own pattern and ≤ −20 elsewhere;
- head-1 takeover and value replication at drug bundles;
- z still ignored at reading bundles.

```python
print('ins rate q50/90/99 %.2f %.2f %.2f; ne rate q50/90/99 %.3f %.3f %.3f; innov sd A %.3f B %.3f'%(*np.percentile(ri0[ri0>0],[50,90,99]),*np.percentile(rn0[rn0>0],[50,90,99]),TA[2].std(),TB[2].std()))
W2o=A(load_original('z_post.2.weight')).astype(float); b2o=A(load_original('z_post.2.bias')).astype(float)
W0e=A(load('z_post.0.weight')).astype(float); b0e=A(load('z_post.0.bias')).astype(float)
cand=[j for j in np.argsort(np.linalg.norm(W2o[:32],axis=0)) if j not in (12,88,105)][:4]; PU=list(map(int,cand))
zsd=xo[:,960:992].std(0); ZD=list(map(int,np.argsort(zsd)[:6])); print('pattern units',PU,'z dims',ZD,'their sd',np.round(zsd[ZD],3))
W0n=W0e.copy(); b0n=b0e.copy(); AP=[]
for p in range(4):
    Dp=np.array([EP[q]-EP[p] for q in range(4) if q!=p]); a_p=Dp.T@np.linalg.solve(Dp@Dp.T,-40*np.ones(3)); AP.append(a_p)
    W0n[PU[p]]=2*W0e[12]; W0n[PU[p],192:384]+=a_p; b0n[PU[p]]=2*b0e[12]-a_p@EP[p]-32
W2n=W2o.copy(); b2n=b2o.copy(); W2n[:,[12,88,105]+PU]=0
for d in ZD: W2n[d,:]=0; b2n[d]=0
for p in range(4): W2n[ZD[p],PU[p]]=1/8
W2n[ZD[4],88]=1; W2n[ZD[4],12]=-1; W2n[ZD[5],105]=0.1; W2n[ZD[5],12]=-0.1
S=300.; kA,kB=3.0,6.0
Zwn=Zw.astype(float).copy()
for p in range(4): Zwn[:,ZD[p]]=S*DES[p]
Zwn[:,ZD[4]]=-S*kA*ETA['G']; Zwn[:,ZD[5]]=S*kB*ETA['M']
def zmean_new(ctx,e,u): return W2n[:32]@gelu_np(W0n@np.r_[ctx,e,u]+b0n)+b2n[:32]
pp=[];zz=[]
for i,k in enumerate(kd):
    e,u=[np.asarray(x,float) for x in enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')]
    pre=W0n@np.r_[ctxs[i],e,u]+b0n; pp.append(pre[PU]); zz.append(zmean_new(ctxs[i],e,u)[ZD])
pp=np.array(pp); zz=np.array(zz)
own=pp[np.arange(len(pp)),pid]; oth=np.where(np.eye(4)[pid]>0,-99,pp).max(1)
print('pattern pre own min/max %.2f %.2f, others max %.2f'%(own.min(),own.max(),oth.max()))
print('z dims at drug: onehot err %.4f, IA vs innovA err %.4f, IB vs innovB err %.4f'%(np.abs(zz[:,:4]-np.eye(4)[pid]).max(),np.abs(zz[:,4]-TA[2]).max(),np.abs(zz[:,5]-TB[2]).max()))
pr=np.array([ (W0n@np.r_[np.asarray(ctx_from(xo[k-1],(hr[k]-hr[k-1])*3600),float).reshape(-1)[:192],np.asarray(e_read(k),float),np.asarray(U_EMPTY,float).reshape(-1)[:192]]+b0n)[PU] for k in kr])
print('nondrug pattern pre max %.2f'%pr.max())
```
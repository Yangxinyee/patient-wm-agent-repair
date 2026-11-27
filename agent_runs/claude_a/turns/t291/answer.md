The compensation is now genuine and well behaved:
- The original model's state explains 91% of the head signal's variance: past insulin is largely predictable from glucose history. The unexplained part (sd 1.42) is the dose deviation the model doesn't know.
- No leakage of s remains in the edited model; there is only small unrelated noise (rms 0.56), worth about 0.3 mg/dL and negligible factual cost.
- The glucose_finger bin centres run from 73 to 272 mg/dL. The predictive sd averages 13.8 mg/dL (variance 212). So a logit tilt along the centres, scaled by 1/100, shifts the predicted mean by about 2.1 mg/dL per unit.

Now I'll assemble the readout:
1. Mean-ablate value units 107 and 109.
2. Unit 107 = GELU(switch + w·(s − compensation)); unit 109 = GELU(switch). Their output weights are +tilt and −tilt.
3. Set w for a direct effect of θ = −2 mg/dL per unit of dose memory, as a starting point for calibration.

Then I'll measure the factual cost on sel2 and dsE (re-encoded), check that non-glucose codes are untouched, and test `rollout_gen` on one stay to learn its output format. I want a cost of ≤0.002 on both sets.

```python
def add_readout(Pb, bet, b0g, gdir, w, k1=107, k2=109, lam_g=0.05, bg=-4.0, WCall=None, tilt=None):
    P={k:v.clone() for k,v in Pb.items()}; W0=P['attr_heads.value_bin.0.weight']; b0=P['attr_heads.value_bin.0.bias']; W2=P['attr_heads.value_bin.2.weight']; b2=P['attr_heads.value_bin.2.bias']
    h1=torch.nn.functional.gelu(torch.as_tensor(WCall)@W0.T+b0)
    for k in (k1,k2): b2+=W2[:,k]*h1[:,k].mean(); W0[k,:]=0; b0[k]=0
    gt=torch.zeros(192); gt[oth]=torch.as_tensor(lam_g*gdir,dtype=torch.float32)
    W0[k2,:]=gt; b0[k2]=bg
    W0[k1,:]=gt; W0[k1,oth]-=torch.as_tensor(w*bet,dtype=torch.float32); W0[k1,50]+=w; W0[k1,99]-=w; b0[k1]=bg-w*b0g
    W2[:,k1]=tilt; W2[:,k2]=-tilt
    return P
tilt=torch.zeros(17); tilt[1:]=torch.as_tensor((cg-160.)/100.,dtype=torch.float32)
theta=-2.0; w_=(theta/cfm[0])*100./212.
PR=add_readout(PB8, bet, b0g, gdir, w_, WCall=WC8, tilt=tilt)
nR2=tf_eval(PR,d82,False); nRE=tf_eval(PR,d8E,False)
print('readout theta=%g w=%.3f: cost sel2 %+.5f dsE %+.5f'%(theta,w_,float((nR2[0]-a_s2).mean()),float((nRE[0]-a_E).mean())))
m2=nR2[2].numpy(); m0=tf_eval(PO,ds_s2,False)[2].numpy()
for c_,nm_ in [(6,'gluc_f'),(10,'gluc_lab'),(7,'map_a'),(5,'map_c')]:
    mk=code2==c_; print(nm_, 'mean shift %.3f sd %.3f'%((m2-m0)[mk].mean(), (m2-m0)[mk].std()))
try:
    Rg=rollout_gen(tu, S['sel1'][0], PO, [{'ins':1.,'ne':1.,'shift':None},{'ins':2.,'ne':1.,'shift':None}], 2, 0)
    print('rollout_gen out', type(Rg).__name__, (list(Rg.keys())[:8] if isinstance(Rg,dict) else [getattr(x,'shape',type(x).__name__) for x in Rg][:6]))
except Exception as ex: print('rg err', str(ex)[:150])
```
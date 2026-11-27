The value-head tensors in the kernel are the original ones. The failed transition-row match therefore means lg was defined differently (probably class 0 excluded), not that the weights are stale. The replica head is validated where it matters: hidden activations match exactly on tool-state inputs.

Factual cost should therefore be measured on the setA rows, which carry the tool's own head inputs (wc_) and probabilities. I will first recover each row's code (from its bin centres) and last bin, then apply the two-constraint decoders. The test is whether the decoders' within-cell noise on setA matches the train-replica estimate (about 1.33 for insulin, 0.26 for NE). If it does, I can scan gains for factual cost on setA without spending forward passes.

```python
VC=VOCAB['value_bin_centres']; cen={c:np.array(VC[c]) for c in CODES4}
codeA=np.array([np.argmin([np.abs(CM[i]-cen[c]).sum() for c in CODES4]) for i in range(5177)])
lbA=np.array([0 if (lastv[i] is None or (isinstance(lastv[i],float) and np.isnan(lastv[i]))) else fbin(CODES4[codeA[i]],lastv[i]) for i in range(5177)]).astype(int)
print('codes',np.bincount(codeA),'lb0 frac',round((lbA==0).mean(),3))
wI=VD[('insulin',1e-4)]; wN=VD[('norepinephrine',1e-4)]
def cellmean(s_tr,c_tr,lb_tr,codes):
    M_=np.zeros((4,17)); 
    for c in codes:
        for b in range(17):
            mk=(c_tr==c)&(lb_tr==b); M_[c,b]=s_tr[mk].mean() if mk.sum()>20 else s_tr[c_tr==c].mean()
    return M_
for nm,w0,codes in [('ins',wI,[0,1]),('ne',wN,[2,3])]:
    sT=Xf@w0; M_=cellmean(sT,cf,lbf,codes); sA=wc_@w0
    for c in codes:
        mk=codeA==c; r_=sA[mk]-M_[c,lbA[mk]]; mt=cf==c; rt=sT[mt]-M_[c,lbf[mt]]
        print(nm,CODES4[c],'setA mean',round(r_.mean(),3),'sd',round(r_.std(),3),'| train sd',round(rt.std(),3))
```
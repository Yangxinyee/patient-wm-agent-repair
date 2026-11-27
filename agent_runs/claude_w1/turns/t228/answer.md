The value head consists of value_bin.0 (192×192), value_bin.2 (17×192) and embed_transition.value_bin (221×17). I will reload all five tensors from the original, check whether the kernel's TR, W2 and b2 differ from them, and redo the setA transition-row match and the Xf NLL with the clean tensors.

```python
W1o=load_original('attr_heads.value_bin.0.weight').astype(np.float64); b1o=load_original('attr_heads.value_bin.0.bias').astype(np.float64)
W2o=load_original('attr_heads.value_bin.2.weight').astype(np.float64); b2o=load_original('attr_heads.value_bin.2.bias').astype(np.float64)
TRo=load_original('embed_transition.value_bin.weight').astype(np.float64)
print('diffs W1,b1,W2,b2,TR',[round(float(np.abs(a-b).max()),4) for a,b in [(W1,W1o),(b1,b1o),(W2,W2o),(b2,b2o),(TR,TRo)]])
W1,b1,W2,b2,TR=W1o,b1o,W2o,b2o,TRo
Lm=log_softmax(Hs@W2.T+b2,axis=1); D_=lg-Lm; TRc=TR-TR.mean(1,keepdims=True)
errs_=[]; best_=[]
for i in range(0,5177,7):
    d=D_[i]-D_[i].mean(); e=np.abs(TRc-d).max(1); j=e.argmin(); best_.append(j); errs_.append(e[j])
print('TR match err median',round(np.median(errs_),3),'max',round(np.max(errs_),3))
L0,H0=head_logits(Xf.astype(np.float64),cf,lbf); nll0=-log_softmax(L0,axis=1)[np.arange(len(obf)),obf]
print([ (CODES4[c], round(nll0[cf==c].mean(),3)) for c in range(4)])
```
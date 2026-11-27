The last run failed because load_original returns a torch tensor and I called a numpy method on it. So nothing was tested yet. The question is still whether the kernel's value-head tensors (W2, b2, TR) are stale, which would explain why the setA transition-row match failed.

Now I will convert the tensors with .numpy() and reload all five value-head tensors from the original. Then I will redo the setA transition-row match and the Xf NLL. If the differences are nonzero and the match error drops to about 0, stale tensors caused the mismatch. If the match error stays large, lg is defined differently and I will validate the head another way.

```python
def LO(n): return load_original(n).numpy().astype(np.float64)
W1o=LO('attr_heads.value_bin.0.weight'); b1o=LO('attr_heads.value_bin.0.bias'); W2o=LO('attr_heads.value_bin.2.weight'); b2o=LO('attr_heads.value_bin.2.bias'); TRo=LO('embed_transition.value_bin.weight')
print('diffs W1,b1,W2,b2,TR',[round(float(np.abs(np.asarray(a,dtype=np.float64)-b).max()),4) for a,b in [(W1,W1o),(b1,b1o),(W2,W2o),(b2,b2o),(TR,TRo)]])
W1,b1,W2,b2,TR=W1o,b1o,W2o,b2o,TRo
Lm=log_softmax(Hs@W2.T+b2,axis=1); D_=lg-Lm; TRc=TR-TR.mean(1,keepdims=True)
errs_=[]
for i in range(0,5177,7):
    d=D_[i]-D_[i].mean(); e=np.abs(TRc-d).max(1); errs_.append(e.min())
print('TR match err median',round(np.median(errs_),3),'max',round(np.max(errs_),3))
L0,H0=head_logits(Xf.astype(np.float64),cf,lbf); nll0=-log_softmax(L0,axis=1)[np.arange(len(obf)),obf]
print([ (CODES4[c], round(nll0[cf==c].mean(),3)) for c in range(4)])
```
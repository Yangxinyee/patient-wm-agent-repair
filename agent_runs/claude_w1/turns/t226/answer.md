The replica NLLs on Xf rows are much higher than the tool's (arterial 2.47 against 1.35). So either my head function is wrong or the Xf head inputs differ from the tool's teacher-forced ones (single z draw, perturbed doses). I will test the head function on setA rows, where wc_, hid_ and the replica log-probs lg came from the tool's own states. I check that gelu(wc_·W1ᵀ + b1) reproduces hid_, and that the leftover between lg and the MLP logits is one transition row per reading.

```python
Hs=gel(wc_@W1.T+b1); print('hid match',round(np.abs(Hs-hid_).max(),4))
Lm=log_softmax(Hs@W2.T+b2,axis=1)
D_=lg-Lm  # should equal TR row (up to const) after renorm
best_=[]; errs_=[]
TRc=TR-TR.mean(1,keepdims=True)
for i in range(0,5177,7):
    d=D_[i]-D_[i].mean(); e=np.abs(TRc-d).max(1); j=e.argmin(); best_.append(j); errs_.append(e[j])
best_=np.array(best_); print('TR match err median',round(np.median(errs_),3),'max',round(np.max(errs_),3),'codes',np.unique(best_//17))
```
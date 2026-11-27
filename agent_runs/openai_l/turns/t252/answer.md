Reconstruction stopped because the context helper returns a tuple, not a single tensor. That is an interface error, not evidence about the repair. I’ll unwrap it and complete the comparison; the resulting factual penalty will test the generalization hypothesis.

```python
import numpy as np, torch
from scipy.special import ndtr, logsumexp

if isinstance(ctx225,tuple):
    print("context return shapes:",[getattr(x,"shape",None) for x in ctx225])
    ctx225 = ctx225[0]
if isinstance(ctxo225,tuple):
    ctxo225 = ctxo225[0]
raw225 = np225(head_inputs96(ctx225,const225,eps225))
rawo225 = np225(head_inputs96(ctxo225,const225,eps225))
X225 = np.concatenate([raw225,np.ones((*raw225.shape[:-1],1),dtype=np.float32)],axis=-1)
pre225 = raw225@Wbase204.T+bbase204
preo225 = rawo225@Wbase204.T+bbase204
H225 = pre225*ndtr(pre225)
Ho225 = preo225*ndtr(preo225)
Forig225 = np.concatenate([H225,np.ones((*H225.shape[:-1],1))],axis=-1)
Fcandidate225 = Forig225.copy()
act225 = X225@candidate219["arch"]["rows"].T
Fcandidate225[:,:,candidate219["arch"]["units"]] = act225*ndtr(act225)
Loriginal225 = Ho225@Wfinal204.T+bfinal204+trans225
Lquiet225 = H225@Wfinal204.T+bfinal204+trans225
Lcurrent225 = H225@np225(load("attr_heads.value_bin.2.weight")).T+np225(load("attr_heads.value_bin.2.bias"))+trans225
Lcandidate225 = Fcandidate225@candidate219["T"]+trans225

print("Reconstructed shapes:",raw225.shape,trans225.shape,centres225.shape)
print("Original/quiet raw RMS difference:",float(np.sqrt(np.mean((raw225-rawo225)**2))))
acttrain225 = Xraw204@candidate219["arch"]["rows"].T
for j in range(act225.shape[-1]):
    print("unit",j,"train quantiles",np.quantile(acttrain225[:,:,j],[0,.01,.5,.99,1]).round(4).tolist(),
          "tuning quantiles",np.quantile(act225[:,:,j],[0,.01,.5,.99,1]).round(4).tolist())
print("VOCAB keys:",list(VOCAB))
print("code42:",type(code42).__name__,str(code42)[:300])
try:
    print("num42 glucose:",num42("glucose_finger"))
except Exception as e:
    print("num42 signature:",num42.__code__.co_varnames[:num42.__code__.co_argcount],str(e))
print("register string constants:",[x for x in register67.__code__.co_consts if isinstance(x,str)])
for key in VOCAB:
    if "edge" in key or key in ["bins","normalisation","normalization"]:
        print(key,str(VOCAB[key])[:850])
print("Current edits unchanged:",edited())
```
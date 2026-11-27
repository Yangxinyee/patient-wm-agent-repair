`load()` returns torch tensors rather than numpy arrays, so the matrix product failed. The notebook entry was written before the error. I will convert the head weights with `.numpy()` and rerun the same check.

The point is still to confirm that numpy reproduces the stored probabilities `P0`, the predicted mean `pred0` and `nll0` from `WC0`. If it does, I can evaluate candidate edits offline before spending forward passes.

```python
import numpy as np, torch
from scipy.special import erf as _erf
def LD(n): 
    t_=load(n); return t_.detach().cpu().numpy().astype(np.float64) if hasattr(t_,'detach') else np.asarray(t_,dtype=np.float64)
W1_=LD('attr_heads.value_bin.0.weight'); b1_=LD('attr_heads.value_bin.0.bias')
W2_=LD('attr_heads.value_bin.2.weight'); b2_=LD('attr_heads.value_bin.2.bias')
def gelu_np(x): return 0.5*x*(1+_erf(x/np.sqrt(2)))
pre_=WC0@W1_.T+b1_; hid_=gelu_np(pre_)
print('Hh0~pre',np.abs(pre_-Hh0).max(),'Hh0~hid',np.abs(hid_-Hh0).max(),'A0~pre',np.abs(pre_-A0).max(),'A0~hid',np.abs(hid_-A0).max())
lg_=hid_@W2_.T+b2_+HI['trans']; lg_-=lg_.max(1,keepdims=True); P_=np.exp(lg_); P_/=P_.sum(1,keepdims=True)
print('P err',np.abs(P_-P0).max())
cen_=HI['cen']; m1_=(P0[:,1:]*cen_).sum(1)
print('pred0 vs sum P[1:]cen: maxabs',np.abs(m1_-RD['pred0'].values).max(), 'P0[:,0] max',P0[:,0].max())
ob_=HI['ob'].astype(int); nl_=-np.log(P0[np.arange(len(ob_)),ob_])
dn_=RD['nll0'].values-nl_
print('nll0 - (-logP[ob]) mean/sd', dn_.mean().round(4), dn_.std().round(4))
print({k:(list(v.keys())[:12] if isinstance(v,dict) else str(v)[:60]) for k,v in VOCAB.items()})
print(RD['event'].value_counts().to_dict())
```
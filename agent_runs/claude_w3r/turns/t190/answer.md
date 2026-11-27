Run 177 shows how the local tools work. `stay_data` gives the bundle list for a stay, with infusion atoms carrying rate and dose (e.g. an insulin start at 0:05 with rate 1.0). `step_at` returns the new state h (4×192) and `out` for one bundle, starting from the true previous state.

To design the channel I need the intermediate values inside a step: E, the prelude tokens and head2's value of u. The step functions look up `LNt`, `mha` and `ffn` as globals, so I can wrap them temporarily with recording versions instead of re-implementing the core. Next I'll log the call sequence (names, shapes) for one action bundle and confirm the output is unchanged by the wrappers. The expected sequence is the prelude norm and attention, prelude_norm, block layers, c_out, coda and out_norm. Anything else would mean my notes on the step order are wrong.

```python
import torch
_LNt0,_mha0,_ffn0=LNt,mha,ffn
LOG=[]
def LNt_w(x,name):
    y=_LNt0(x,name); LOG.append(('LN',name,tuple(x.shape),x.clone(),y.clone())); return y
def mha_w(xq,xk,xv,pre,kadd=None):
    y=_mha0(xq,xk,xv,pre,kadd); LOG.append(('MHA',pre,(tuple(xq.shape),tuple(xk.shape),None if kadd is None else tuple(kadd.shape)),(xq.clone(),xk.clone(),xv.clone(),kadd),y.clone())); return y
def ffn_w(x,pre):
    y=_ffn0(x,pre); LOG.append(('FFN',pre,tuple(x.shape),x.clone(),y.clone())); return y
globals().update(LNt=LNt_w,mha=mha_w,ffn=ffn_w)
k_test=[i for i,b in enumerate(B_) if any(a['kind']==0 and a['rate'] is not None for a in b[1])][5]
LOG.clear(); r=step_at(B_,rows_,EU_,k_test)
globals().update(LNt=_LNt0,mha=_mha0,ffn=_ffn0)
r0=step_at(B_,rows_,EU_,k_test)
print('bundle',k_test,B_[k_test][0],[ (a['code'],a['kind'],a['rate']) for a in B_[k_test][1]], 'same:', float((r[0]-r0[0]).abs().max()))
for L_ in LOG: print(L_[0],L_[1],L_[2], 'out', tuple(L_[4].shape), 'norm_in %.1f out %.1f'%(float(L_[3].norm()) if L_[0]!='MHA' else float(L_[3][0].norm()), float(L_[4].norm())))
```
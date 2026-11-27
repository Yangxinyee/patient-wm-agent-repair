Run 178 failed because `mha` returns a tuple (probably the output plus the attention weights), not a single tensor. My wrapper tried to `.clone()` the tuple. I'll make the wrapper handle tuple outputs and restore the original functions with try/finally, so a failure can't leave the kernel patched. The goal is unchanged: get the call sequence of one action-bundle step. That confirms or corrects my picture of the step order before I design the channel.

```python
def _cl(v):
    if isinstance(v,torch.Tensor): return v.clone()
    if isinstance(v,(tuple,list)): return type(v)(_cl(t) for t in v)
    return v
def mha_w(xq,xk,xv,pre,kadd=None):
    y=_mha0(xq,xk,xv,pre,kadd); LOG.append(('MHA',pre,(tuple(xq.shape),tuple(xk.shape),None if kadd is None else tuple(kadd.shape)),(xq.clone(),xk.clone(),xv.clone(),kadd),_cl(y))); return y
LOG.clear()
globals().update(LNt=LNt_w,mha=mha_w,ffn=ffn_w)
try:
    r=step_at(B_,rows_,EU_,k_test)
finally:
    globals().update(LNt=_LNt0,mha=_mha0,ffn=_ffn0)
r0=step_at(B_,rows_,EU_,k_test)
print('bundle',k_test,B_[k_test][0],[ (a['code'],a['kind'],a['rate']) for a in B_[k_test][1]], 'same:', float((r[0]-r0[0]).abs().max()))
def _sh(v): return tuple(v.shape) if isinstance(v,torch.Tensor) else [ _sh(t) for t in v] if isinstance(v,(tuple,list)) else v
for L_ in LOG:
    xin=L_[3][0] if L_[0]=='MHA' else L_[3]
    yo=L_[4][0] if isinstance(L_[4],(tuple,list)) else L_[4]
    print(L_[0],L_[1],L_[2],'out',_sh(L_[4]),'norm_in %.1f out %.1f'%(float(xin.norm()),float(yo.norm())))
```
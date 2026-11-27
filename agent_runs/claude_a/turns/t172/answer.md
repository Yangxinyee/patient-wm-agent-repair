Two things in this result are surprising.

First, `PS` and `P_` give exactly the same NLL (1.41046), although `PS` also edits the core and two value-MLP units and `P_` does not. That suggests my offline evaluator may not be reading all of the parameter dict it is given, for example by using cached states for the core.

Second, the scaffold costs about +0.006 nats (se 0.003). That is already over budget, so the choice of role dims needs rethinking anyway.

Before trusting any offline number, I'll check whether the evaluator's states depend on the passed parameters. I'll compare the returned h states for `PO`, `PS` and `P_`. If `PS` and `PO` states are identical, the evaluator ignores core edits and I must fix or replace it.

```python
for nm,R in [('PS',RS),('P_',R_)]:
    print(nm,'max|dH| %.4g  max|dOut| %.4g'%(float((R[3]-R0[3]).abs().max()), float((R[4]-R0[4]).abs().max())),
          ' max|dP| %.4g'%float((R[1]-R0[1]).abs().max()))
print('PS vs P_ probs max diff %.3g'%float((RS[1]-R_[1]).abs().max()))
print('tf_states names:', tf_states.__code__.co_names[:40])
print('tf_nll names:', tf_nll.__code__.co_names)
```
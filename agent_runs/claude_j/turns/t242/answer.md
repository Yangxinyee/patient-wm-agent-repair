Run 214 shows that `set_amp(lam)` scales columns 2 and 3 of `embed.numeric.weight`, the dose numerics. These are shared by insulin and norepinephrine. `amp_eval` compares insulin x2 against factual through the replica and the heads.

My working idea is that the dose signal is squashed by the layer norms before it reaches the state. If so, amplifying the dose input could make the model respond more strongly to dose changes.

Next I will time `amp_eval` at lam 1 and lam 3 on 3 stays, look at what it returns, then restore the original scale. If lam 3 raises the model's insulin x2 shift well above lam 1, input amplification is a usable lever. If the shift barely changes, that route is dead.

```python
import time
st=list(BUN)[:3]
for lam in [1.0,3.0]:
    t0=time.time(); r_=amp_eval(lam,st); dt_=time.time()-t0
    print(lam,round(dt_,1),type(r_).__name__, (r_ if not hasattr(r_,'__len__') else (len(r_), str(r_)[:300])))
set_amp(1.0); print(float((W['embed.numeric.weight']-NW0).abs().max()) if 'embed.numeric.weight' in W else 'noW')
```
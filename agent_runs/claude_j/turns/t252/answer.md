Run 224 crashed because in run 221 I overwrote the global `F`, which the replica's layer-norm helper uses as torch.nn.functional. That was my bookkeeping slip, not a finding about the model.

Next I will restore `F` and list any other replica function that reads a global I may have overwritten in runs 217–223. Then I will rerun the amplification experiment unchanged, with the same criteria as before.

```python
import torch.nn.functional as F
import types
mine={'C','J','w','mu','Sf','r','tr','te','Xf','lam','rs','isGl','iC','iFm','Dm','gF','gJ','mJ','trs','st','s0','rt0','r0','b0','bk','vv','k_ins','o','sv','nl','sh','mf','d0','gmask','code','L','s','A_','ls','i','j','dt_','t0','r_','fn','big','n','c','f'}
hit={k:[x for x in v.__code__.co_names if x in mine] for k,v in list(globals().items()) if isinstance(v,types.FunctionType)}
print({k:v for k,v in hit.items() if v})
```
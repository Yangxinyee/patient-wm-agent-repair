Run 216 gives the layout. `RTs[s]` holds the stay's readings: minute, event, observed value and the tool's `nll`. `amp_eval` returns the factual and counterfactual (insulin x2) logits per reading, with 4 z draws and 17 bins.

Next I will write a saved helper, `/work/amp.py`, that turns these logits into the factual NLL per reading and the counterfactual glucose shift. Then I will scan amplification factors 1, 2 and 4 on the 100 tuning stays. The premise is supported if the shift grows several-fold with lam while the factual NLL rises only a little. If the shift barely moves, input amplification is useless. If the NLL blows up, I would need the head compensation I worked out in my reasoning.

```python
import inspect
print('fb_',inspect.signature(fb_))
code=r'''
import numpy as np, torch
def amp_stats(lam, stays):
    res=amp_eval(lam, stays); out=[]
    for (s,k0,k1,lgf,lgc) in res:
        rt=RTs[s]; ev=rt['event'].values
        pf=torch.softmax(lgf,-1).mean(1).numpy(); pc=torch.softmax(lgc,-1).mean(1).numpy()
        for i in range(len(rt)):
            c=CEN[ev[i]]; b=int(rt['bfine'].values[i]) if 'bfine' in rt else None
            out.append((s,ev[i],rt['minute'].values[i],float(pf[i]@c),float(pc[i]@c),pf[i],rt['observed'].values[i],rt['nll'].values[i]))
    return out
'''
open('/work/amp.py','w').write(code); exec(code)
```
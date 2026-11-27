Run 243 decoded `amp_eval`. It seeds z-noise per stay, runs the factual replica from the first drug onward and the counterfactual from minute 720, and calls `heads_full`, which returns both the head features and the logits.

Next I will write `/work/gen2.py` along the same lines. It will compute the factual run once per stay, plus four counterfactuals: insulin x2 and x1/2, and norepinephrine x2 and x1/2, with readings as recorded. I will run it on the 100 tuning stays and store the head features and logits.

As a sanity check, the insulin x2 glucose_finger shift should come out near −0.86, as before, and norepinephrine x2 should give a small positive map_arterial shift. I will also confirm which version of `mod_events` is active, so that no hidden reading shift is applied.

```python
print('mod_events is orig:', mod_events is mod_events_orig, '| is new:', mod_events is mod_events_new)
code2=r'''
import numpy as np, torch, time
def gen2(s, conds, start_min=720):
    rt=RTs[s]; k_m=first_drug_min(s); k_m=1000000 if k_m is None else k_m
    eps=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(int(s)%100000),dtype=torch.float64)
    Of,Hf,k0=run_cf(s,{},k_m); out={'f':heads_full(Of,Hf,rt,eps)}
    for (dr,m) in conds:
        Oc,Hc,k1=run_cf(s,{dr:m},start_min); out[(dr,m)]=heads_full(Oc,Hc,rt,eps)
    return out
def mean_of(lg,ev):
    p=torch.softmax(torch.as_tensor(lg),-1).mean(1).numpy()
    return np.array([p[i,1:]@CEN[ev[i]] for i in range(len(ev))])
'''
open('/work/gen2.py','w').write(code2); exec(code2)
CONDS2=[('insulin',2.0),('insulin',0.5),('norepinephrine',2.0),('norepinephrine',0.5)]
t0=time.time(); GD={}
for s in stays100: GD[s]=gen2(s,CONDS2)
wcf_,lgf_=GD[s0]['f']; print('time %.0f'%(time.time()-t0), type(wcf_).__name__, tuple(wcf_.shape), tuple(lgf_.shape))
```
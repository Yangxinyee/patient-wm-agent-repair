Edit v2 is saved, and the gate margins are clean:
- Target-code rows activate between 5 and 16 for both pairs.
- Other measured codes stay at −5 or below.
- Unmeasured codes are estimated at −3.5 or below for insulin (worst case) and −7.5 for drug B.

The cost of this design is that the main and cancelling units carry large logit terms (up to about 100) that cancel each other and the transition offsets. If the GPU uses reduced-precision matrix products, that could add noise.

Next I compare the tool's factual NLL change on the 100 tuning stays with my own in-sample prediction for the same edit, computed from stored logits. A match means precision is fine. A much larger change on the tool would point to precision or z-draw noise. This run prints the stay ids for the request and the predicted NLL change.

```python
import json
sel1=json.load(open('/work/sel1.json')); sel1=sel1 if isinstance(sel1,list) else list(sel1.values())[0]
def pred_dnll(fa_,P,v_):
    k=int(np.argmax(np.abs(v_))); gk=P['col'][k]/v_[k]
    d=WC[fa_.ri.values]@(P['um']-P['uc'])+P['B']-P['Bc']; mu=P['mu'][fa_.event.map(cid_of).values,fa_.lb.values]; th=gk*(d-mu)
    ar_=np.arange(len(fa_)); ob_=fa_.ob.values; LGf_=LG[fa_.ri.values]
    n0=-np.log(mix(LGf_,np.zeros(len(fa_)),v_).mean(1)[ar_,ob_]); n1=-np.log(np.maximum(mix(LGf_,th,v_).mean(1)[ar_,ob_],1e-12))
    return (n1-n0).sum(), th.std()
si,ti=pred_dnll(fac,P_ins,vG); sn,tn=pred_dnll(faN,P_ne,vM)
print('pred in-sample dNLL per reading: total',round((si+sn)/5214,5),'glu part',round(si/5214,5),'map part',round(sn/5214,5),'tilt sd',round(ti,3),round(tn,3))
print(len(sel1)); print(json.dumps([int(s) for s in sel1]))
```
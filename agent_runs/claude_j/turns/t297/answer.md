This settles the diagnosis. In the replica, the edit's teacher-forced effect is zero at hour 13 (the first reading after the change, pure dose) and grows to about −8 by hour 24. So the readout responds only to the surprise that builds up when recorded readings ignore the scaled dose, and that surprise is absent in rollouts. The tool's extra shift accordingly stays small (−0.3 → −1.0).

A dose readout from the state alone is too noisy to supply large per-step effects within the NLL budget. The better target is the exact cf. There the core sees scaled doses plus readings lowered by the true effect, which is what it would see in a rollout of a successful head edit. In that setting the needed extra per step was estimated to be small (about −1.5), because the core then follows consistently shifted readings strongly.

Before generating that data, I check three things:
- how mod_events_new keys the SHIFT dict;
- what shifts() returns;
- how long one cf replay takes.

```python
print('consts',[c for c in mod_events_new.__code__.co_consts if not hasattr(c,'co_code')][:20])
s0=sel1[0]; r_=shifts(s0,'insulin',2.0,12); print('shifts ->',type(r_).__name__, (list(r_.items())[:4] if isinstance(r_,dict) else (np.asarray(r_).shape if hasattr(r_,'__len__') else r_)))
import time; t0=time.time(); SHIFT.clear(); Oc,Hc,k1=run_cf(s0,{'insulin':2.0},720); t1=time.time()
print('run_cf time',round(t1-t0,1),'k1',k1,'O',tuple(Oc.shape),'nb',len(BUN[s0]),'rt rows',len(RTs[s0]),'rt cols',list(RTs[s0].columns))
t0=time.time(); wc_,lg_=heads_full(Oc,Hc,RTs[s0],torch.randn(len(RTs[s0]),4,32,dtype=torch.float64)); print('heads time',round(time.time()-t0,1))
```
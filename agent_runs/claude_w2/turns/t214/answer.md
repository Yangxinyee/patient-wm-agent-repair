The notebook has the rest of what I need: the record-based true effects.
- **Insulin on glucose:** the impulse response is about −2.0, −2.5, −2.2, −1.8, −1.4 … mg/dL per U/h at lags 1–10, summing to −13.7. It is linear, does not depend on the glucose level, and is the same on finger and lab glucose.
- **Norepinephrine on MAP:** +3.6, +2.7, +1.7, +1.1, +0.4 … per mcg/kg/min, summing to about +10.4. It is the same on arterial and cuff MAP.
- **What the original model predicts:** only about 15% (insulin) and 25–30% (norepinephrine) of these effects in its dose_edit rollouts.

Run 207 was an "oracle" test inside my local rollout engine. I need to recall exactly how it injects the effect, and how its "true" column is computed, before turning it into a real weight edit. Next I will print oracle.py and cf.py, plus the diff between rollout.py and rollout_h2.py. This is a context-recovery step, not a test. I expect to find an additive shift on the glucose head driven by the dose history at lags 1..L, with a gain.

```python
import difflib
print(open('/work/oracle.py').read())
print('=====cf.py=====')
print(open('/work/cf.py').read())
print('=====diff rollout.py -> rollout_h2.py=====')
a=open('/work/rollout.py').read().splitlines(); b=open('/work/rollout_h2.py').read().splitlines()
for l in difflib.unified_diff(a,b,lineterm='',n=1): print(l)
```
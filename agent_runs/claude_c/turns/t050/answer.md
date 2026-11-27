The per-hour comparison suggests the model already has a weak insulin effect of the right form. For insulin x2 the model's shift is about one-seventh of my estimated truth at every lag: −0.43 vs −2.9 at lag 1, −3.4 vs −19.6 at lag 12. The x1/2 response is about 0.1–0.2 of the truth. So the per-step effect is the missing piece; the accumulation over hours is about right.

Norepinephrine is different. The original model's MAP response is negative for both x2 and x1/2, which looks like a reaction to "any change" rather than a dose effect. My embedding edit just added about +0.6 to both.

My key worry is double counting. Raw-dose amplification is costly, but z-score centring at the median dose (1.5 U/h) removes most of the systematic part. That makes amplification of the z-score pathway plausible, if it acts roughly linearly.

Next, from data alone and without forward passes, I check:
- the projected factual-NLL cost of a lag-1 effect −1.98 × (dose − d0) for d0 = 0 and d0 = 1.5, using the original residuals;
- how the model's per-stay hour-13 shifts depend on the recorded doses, to test linearity for insulin and to see whether the odd norepinephrine response comes from stays at the 1.0 cap.

```python
# 1) projected NLL cost of lag-1 effect -k*(dose-d0) at finger readings after insulin
af = gfz[(gfz.ins_prev>0)].copy()
sig2 = gfz.resid.var()
for d0 in [0.0, 1.0, 1.5, 2.0]:
    for k in [1.0, 1.98]:
        dm = 2*k*af.resid*(af.ins_prev-d0) + k**2*(af.ins_prev-d0)**2
        print(f"d0={d0} k={k}: dNLL/reading(all) ~ {dm.sum()/(2*sig2)/len(fa2):+.4f}")
# 2) per-stay lag-1 shifts vs dose
T1 = T[T.since==1].copy()
T1['d12'] = [lk('insulin', s, h-1) for s,h in zip(T1.stay_id, T1.hour)]
T1['ne12'] = [lk('norepinephrine', s, h-1) for s,h in zip(T1.stay_id, T1.hour)]
for e_ in ['insulin_x2','insulin_half']:
    q_ = T1[T1.edit==e_]
    print(e_, "model vs dose slope", np.polyfit(q_.d12.fillna(0), q_.model, 1).round(3), "true slope", np.polyfit(q_.d12.fillna(0), q_.true, 1).round(3))
for e_ in ['ne_x2','ne_half']:
    q_ = T1[T1.edit==e_]
    print(e_); print(q_[['ne12','true','model','edited']].sort_values('ne12').round(2).to_string())
```
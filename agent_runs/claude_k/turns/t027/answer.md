The 12-lag fits settle the shapes.

- **Insulin:** h ≈ -1.99, -2.39, -2.11, -1.72, -1.22, -0.95, -0.69, -0.55, -0.35, -0.15, -0.12, -0.05, close to -1.99·k·0.6^(k−1) throughout (sum about -12.3 mg/dL per sustained unit/h). The total effects reach zero by lag 9, consistent with a fully transient two-compartment response.
- **Norepinephrine:** h ≈ 3.35, 2.60, 1.75, 1.03, 0.42, 0.27, then about 0 from lag 7 on (sum about 9.6 mmHg per sustained 1 mcg/kg/min). Its tail is shorter than a two-pole form would give, so I'll use the smoothed deconvolution rather than force a parametric shape.

Now I need to see how the original model answers these questions. Next I'll pick tuning stays for a dose_edit request. I'll look at how many have insulin and norepinephrine from hour 12 on and choose 100 with both or either. I'll also compute my predicted true shifts for those stays, so the forward pass compares the model against the records-based truth directly. I expect the original model to get these effects wrong (too small, or even the wrong sign through confounding). If it's already close, I'd only need fine-tuning.

```python
def stay_doses(Pd, sid_list, drug):
    A = Pd[drug].reindex(sid_list).values
    return np.nan_to_num(A[:, col(0):col(24)+1])   # hours 0..24
ins_t = stay_doses(PT, ids_t, 'insulin'); ne_t = stay_doses(PT, ids_t, 'norepinephrine')
has_ins12 = (ins_t[:,12:]>0).any(1); has_ne12 = (ne_t[:,12:]>0).any(1)
print("tuning: n", len(ids_t), " insulin>=12:", has_ins12.sum(), " norepi>=12:", has_ne12.sum(), " both:", (has_ins12&has_ne12).sum(), " either:", (has_ins12|has_ne12).sum())
art_t = np.isfinite(PT['map_arterial'].reindex(ids_t).values[:, col(13):col(24)+1]).any(1)
print("norepi>=12 with arterial line:", (has_ne12 & art_t).sum())
h_ins = np.array([-1.99,-2.39,-2.11,-1.72,-1.22,-0.95,-0.69,-0.55,-0.35,-0.15,-0.12,-0.05])
h_ne  = np.array([3.35,2.60,1.75,1.03,0.42,0.27,0.05,0.05,0.05,0.0,0.0,0.0])
def true_shift(D, h, mult, start=12):
    # D: [n, 25] doses hours 0..24; shift at hours start+1..24 = sum_{s=start}^{t-1} h[t-s-1]*(mult-1)*D[s]
    n = D.shape[0]; S = np.zeros((n, 25))
    for t in range(start+1, 25):
        for s in range(start, t):
            if t-s-1 < len(h): S[:, t] += h[t-s-1]*(mult-1)*D[:, s]
    return S
print("ok")
```
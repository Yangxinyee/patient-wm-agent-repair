The insulin threshold check closes the "hidden large effect" route. At the pooled sliding-scale cut-offs, where the base dose jumps by about 1 u/h, later finger glucose changes by:

| Lag | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Change (mg/dL) | +0.10 ± 0.13 | +0.21 ± 0.14 | +0.02 ± 0.15 | +0.19 ± 0.17 | −0.09 ± 0.18 | −0.08 ± 0.19 |

This rules out any realistic insulin effect even if the recorded jitter were fake. Insulin is absent by two independent designs.

The norepinephrine start analysis gave something new, with a caveat:
- **A positive jump:** MAP just below the cut-off, which always triggers a start, rises about 0.5 mmHg more at lag 1 than MAP just above (−0.54 ± 0.20 above-minus-below, fading over 6 hours). That suggests starting norepinephrine raises MAP slightly.
- **But a linear effect is ruled out:** a dose-proportional effect would mean about +6 mmHg per unit rate. The titration innovations rule that out (lag 1 is +0.06 ± 0.08 per unit).
- **So something else is going on:** either a small "presence" effect that saturates (irrelevant to ×2 and ×½ for patients already on the drug), or an artefact at the cut-off.

The start probability just above 64.5 is 22%, not 0. MAP readings are probably integers and the true rule is probably MAP ≤ 65, so my cut-off may be misplaced. A shift to arterial-line readings after a start could also create a jump if arterial and cuff read differently.

Next I will:
- tabulate P(start) at each integer MAP to find the real cut-off;
- redo the start comparison there with arterial-only and cuff-only outcomes, and at placebo cut-offs;
- check whether arterial-reading frequency jumps after a start;
- estimate the titration effect in low-rate rows (r < 0.2), where a saturating effect would show its slope.

If the jump survives at the correct cut-off in both arterial and cuff outcomes and not at placebo cut-offs, the start effect is real but tiny, and the low-rate slope will show whether it matters for ×½. If not, it is an artefact.

```python
t0=time.time()
st=(NE0[sN0,tN0]>0)
print('P(start) by policy MAP (int):',{m:round(st[np.round(mN)==m].mean(),3) for m in range(58,72)})
src_art=~np.isnan(GA[sN0,tN0])
print('P(start) by art MAP:',{m:round(st[src_art&(np.round(mN)==m)].mean(),3) for m in range(62,69)})
print('P(start) by cuff MAP:',{m:round(st[~src_art&(np.round(mN)==m)].mean(),3) for m in range(62,69)})
print('MAP values integer?', np.mean(np.isclose(GA[~np.isnan(GA)]%1,0)).round(3), np.mean(np.isclose(GC[~np.isnan(GC)]%1,0)).round(3))
# arterial-reading frequency after start vs not, near threshold
for k in (1,2,3):
    tt=tN0+k; ok=tt<=25
    near=(mN>=60)&(mN<70)&ok
    pa=~np.isnan(GA[sN0[near],tt[near]])
    print(' k%d P(arterial present) started %.3f vs not %.3f'%(k,pa[st[near]].mean(),pa[~st[near]].mean()))
print('time',round(time.time()-t0,1))
```
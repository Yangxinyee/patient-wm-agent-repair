The start rule is now clear:
- **MAP ≤ 64:** norepinephrine always starts (100%).
- **MAP 65:** it starts 54% of the time.
- **Above 65:** start probability falls smoothly (12%, 9%, 6%, 5%, 3% at 66–70).

MAP readings are integers, so the sharp jump is between 64 and 65. Arterial readings become only slightly more common after a start (about 50% vs 46%). That composition shift is too small to explain a 0.5 mmHg jump.

The key question is whether the start jump tells me anything about ×2 and ×½ for patients already on the drug, so I will run three checks:
1. **Titration slope by current rate:** if the effect is steep at low doses and saturates (which would reconcile a positive start effect with a flat titration slope), low-rate rows (r < 0.2) should show a clearly positive lag-1 slope. That would matter for ×½.
2. **Start comparison by reading type:** run it separately for arterial and cuff outcomes at the real 64/65 cut-off.
3. **Placebo cut-offs:** repeat at 61.5 and 68.5. A real start effect appears only at 64.5; similar jumps elsewhere would show the design is unreliable.

```python
t0=time.time()
s_,t_=RNf_R['si'],RNf_R['ti']; r_prev=NE0[s_,t_-1]
for nm,m in [('r1<0.2',r_prev<0.2),('0.2<=r1<0.5',(r_prev>=0.2)&(r_prev<0.5)),('r1>=0.5',r_prev>=0.5)]:
    L,Ls,ro,ros=window_stats(RNf_R,m,Aw); print('%-12s n %6d L1-4 %s se %s readout %.2f (%.2f)'%(nm,m.sum(),L[:4].round(2),Ls[:4].round(2),ro,ros))
# RD at starts with separate outcomes and placebo cutoffs
for c in (61.5,64.5,68.5):
    for nm,outc in [('art',GA),('cuff',GC),('comb',MAPc)]:
        rd_pooled(sN0,tN0,mN,[c],outc,K=4,bw=6,label='RD cut %.1f %-4s'%(c,nm))
print('time',round(time.time()-t0,1))
```
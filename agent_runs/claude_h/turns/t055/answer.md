The test refutes the "titrate on the true MAP" idea, which is useful:
- **Arterial dominates:** in rows with both readings, the rule follows arterial (−0.0035 per mmHg). Cuff adds essentially nothing (−0.0002), and adding it to the fit makes the prediction no better.
- **No dilution:** arterial-only and cuff-only slopes are equal (−0.0035 vs −0.0037), with no dilution for the noisier cuff.
- **Rule shape:** step-like. The rate goes up about 0.03 per hour if MAP < 65, holds between 65 and 74, and goes down if MAP > 75. It is capped near 1.0.

The rule therefore uses the charted reading, and in hours with a reading at t the innovation should be clean.

One loose end could explain the large minimal-versus-full gap. 12,291 continuing rows (5.5%) have norepinephrine given with no MAP reading at t. Those rows must be titrated on an older reading. The minimal controls (current readings only) cannot see that reading, so their "innovation" partly contains the response to the last low or high MAP. That would bias the estimate negative. My earlier "cuff-only" subset also wrongly included these rows.

Next I will:
1. Restrict to rows with a MAP reading at t and compare minimal against full controls. If my explanation is right, they should now agree, and that estimate is the cleanest I can get.
2. Separate arterial-only, cuff-only and both properly.
3. Look at how no-reading rows are titrated.
4. Run the same minimal-versus-full test for insulin, whose rule is fully captured.

If the restricted estimates agree and come out near 0, the earlier −0.3 to −0.4 was contamination from the no-reading rows. If they agree near −0.3, the effect is more credible.

```python
t0=time.time()
hasread=~(np.isnan(GA)&np.isnan(GC))
contR=cont&hasread
RNf_R=build_resid(NE0, contR, feats, MAPc)
RNm_R=build_resid(NE0, contR, featsMin, MAPc)
for nm,Rz in [('full|read',RNf_R),('min|read',RNm_R)]:
    L,Ls,ro,ros=window_stats(Rz,allm(Rz),Aw); print('%-10s n %d readout %.3f (%.3f) L1-8 %s'%(nm,len(Rz['si']),ro,ros,L[:8].round(2)))
s_,t_=RNf_R['si'],RNf_R['ti']; a_=~np.isnan(GA[s_,t_]); c_=~np.isnan(GC[s_,t_])
for nm,m in [('both',a_&c_),('art-only',a_&~c_),('cuff-only',~a_&c_)]:
    L,Ls,ro,ros=window_stats(RNf_R,m,Aw); print('  %-9s n %6d readout %.3f (%.3f) L1-6 %s'%(nm,m.sum(),ro,ros,L[:6].round(2)))
for nm,m in [('t<12',(t_-1)<12),('t>=12',(t_-1)>=12)]:
    L,Ls,ro,ros=window_stats(RNf_R,m,Aw); print('  %-9s n %6d readout %.3f (%.3f)'%(nm,m.sum(),ro,ros))
# none rows: how titrated? dr vs last MAP
nr=cont&~hasread; sN,tN=np.nonzero(nr)
lastmap=np.full(len(sN),np.nan); age=np.full(len(sN),np.nan)
MAPpol=np.where(np.isnan(GA),GC,GA)
for j in range(1,6):
    v=np.where(tN-j>=0, MAPpol[sN,np.maximum(tN-j,0)], np.nan); fill=np.isnan(lastmap)&~np.isnan(v); lastmap[fill]=v[fill]; age[fill]=j
drN=NE[sN,tN]-NE0[sN,tN-1]
print('none rows',len(sN),'age dist',np.bincount(np.nan_to_num(age).astype(int)))
for lo,hi in [(0,62),(62,65),(65,70),(70,75),(75,200)]:
    m=(lastmap>=lo)&(lastmap<hi); print('  last MAP [%d,%d) n %d mean dr %.3f'%(lo,hi,m.sum(),drN[m].mean()))
# insulin minimal vs full
ifeatsMin={'gp0':Gpol,'gf0':GF,'gl0':GL}
RIm=build_resid(INS0, irows_pos, ifeatsMin, GF)
for nm,Rz in [('INS full',RI),('INS min',RIm)]:
    L,Ls,ro,ros=window_stats(Rz,allm(Rz),AwI); print('%-9s readout %.3f (%.3f) L1-8 %s'%(nm,ro,ros,L[:8].round(2)))
print('time',round(time.time()-t0,1))
```
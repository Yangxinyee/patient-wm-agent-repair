The leak is confirmed and it is large.

- **What the readouts see.** Applied to the slot tokens, the insulin readout gives about +1,200 ± 600 instead of 0. Head 1 puts about 0.04% attention on each slot, so every bundle receives a spurious insulin signal worth roughly 2 units/h. The event and z tokens give readouts in the thousands; head 1 appears to give them essentially no attention, but I haven't confirmed that.
- **Worst case.** At each stay's first bundle (hospital admission, hour −2) the edit changes the state by 422, about half the slot norm, identically for every stay. That points to the learned initial state being read by the readout. The change then decays over the following bundles.
- **Why dose_edit missed it.** Each dose_edit shift compares two rollouts of the same edited model, so this constant contamination largely cancels there. The factual fit would have been badly hurt.

The other numbers also suggest a simpler design. The injected vector passes into the new slots almost unchanged (|Δh|/|p| ≈ 0.9). So the injection should point along the model's own "glucose is higher" direction in the slots (nuG, about 1 unit of norm per mg/dL). If dI is poorly aligned with it, that explains the tiny effect.

Next, in one run:
1. Print exactly how much attention head 1 gives the event and z tokens.
2. Find the initial-state tensor and the decay parameter `flow.a`.
3. Measure how well dI and dN align with the glucose and MAP directions nuG and nuM.
4. Fit a ridge readout of the original model's predicted glucose mean from the context vector before each reading, then project the v3 state changes onto it. That shows directly how many mg/dL v3 moved the predictions, and where.
- **Explanation confirmed:** dI is nearly orthogonal to nuG, and the projected glucose changes are dominated by the first-bundle leak.

```python
np.set_printoptions(linewidth=150)
print('head-1-like attention on e (tok4) and z (tok6): mean %.2e max %.2e | %.2e max %.2e ; slots total mean %.2e max %.2e'%(attd[:,:,4].mean(),attd[:,:,4].max(),attd[:,:,6].mean(),attd[:,:,6].max(),attd[:,:,:4].sum(-1).mean(),attd[:,:,:4].sum(-1).max()))
print([n for n in NAMES if any(k in n for k in ('init','h0','slot','flow','start'))], SHAPES.get('core.flow.a'))
cs=lambda u,v: float(u@v/np.linalg.norm(u)/np.linalg.norm(v))
print('cos(dI,nuG_s)',[round(cs(dI,nuG[s]),3) for s in range(4)],' cos(dN,nuM_s)',[round(cs(dN,nuM[s]),3) for s in range(4)], '|nuG_s| %.3f |nuM_s| %.3f'%(np.linalg.norm(nuG[0]),np.linalg.norm(nuM[0])))
# map factual rows to bundles
bidx={(int(s),round(float(h),4)):i for i,(s,h) in enumerate(zip(sid,hr))}
rowsF=[];ip=[];dts=[]
for i,(s,h) in enumerate(zip(fa.stay_id.values,fa.hour.values)):
    b=bidx.get((int(s),round(float(h),4)))
    if b is None or b==0 or sid[b-1]!=sid[b]: continue
    rowsF.append(i); ip.append(b-1); dts.append((hr[b]-hr[b-1])*3600.)
rowsF=np.array(rowsF); ip=np.array(ip); dts=np.array(dts)
print('mapped factual rows', len(rowsF),'of',len(fa))
Ho=xo[:,192:960].reshape(-1,4,192); He=xe[:,192:960].reshape(-1,4,192)
try:
    Co=np.asarray(ctx_of(xo[:,:192],Ho,ip,dts)); Ce=np.asarray(ctx_of(xe[:,:192],He,ip,dts)); print('ctx_of ok',Co.shape)
except Exception as e:
    print('ctx_of failed',e); Co=xo[ip,:192]; Ce=xe[ip,:192]
from sklearn.linear_model import RidgeCV
evs=fa.event.values[rowsF]; pm=fa.predicted_mean.values[rowsF]
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    m=evs==ev; r=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Co[m],pm[m])
    dp=r.predict(Ce[m])-r.predict(Co[m]); pr=r.predict(Co[m])
    print(f'{ev}: n {m.sum()} R2 {1-((pm[m]-pr)**2).mean()/pm[m].var():.4f}  implied v3 pred change: mean {dp.mean():+.3f} sd {dp.std():.3f} max|.| {np.abs(dp).max():.2f}')
    WR_fit={} if 'WR_fit' not in globals() else WR_fit; WR_fit[ev]=r
```
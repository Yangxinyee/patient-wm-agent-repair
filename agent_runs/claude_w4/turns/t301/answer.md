Two things came out of this run.

**I damaged my own kernel.** In the previous run I reused the variable name `c` in a loop, overwriting the global channel scale that `build_v4` depends on. Also, the global `rbar`, which `build_v4` uses as the insulin centring rate, now reads 0.52, the NE value. The notes say 1.77. So I must restore both from /work/v4_parts.npz before rebuilding anything.

**The replica is exact, and the channel does leak at reading bundles.**
- `prelude_b` returns the after-attention and after-FFN outputs, and both match the stored drug-bundle results exactly.
- At drug bundles the channel coordinates are zero on average (sd 0.0006), as designed.
- At reading bundles the insulin channel sits at +0.0097 on average (sd 0.0044). That is 15 times its drug-bundle spread, so a dose-independent leak exists. The NE channel there is near zero (−0.001).
- By my design scaling, though, a +0.0097 leak should push glucose down by only about 0.1 mg/dL per bundle. The observed drift is +2 mg/dL per hour, upward.

So either the reading-bundle tokens in my replica are inaccurate (the reading-atom encoding was never replicated exactly), or the realised effect of the injected state change differs from the design. A large written vector in directions the next output barely sees could still steer later steps.

This run will:
- restore `c` and `rbar` and check the stored channel directions;
- compute the leak in the design's units;
- compare the size of the written vectors with typical `b_in` outputs;
- save the fwd_8 stay list, so the next forward pass can be a factual "both" run on those stays.

That factual run will give the factual NLL cost of v4 and the per-reading prediction differences. From those differences I can separate the push at reading bundles from the push at insulin and NE bundles, under recorded readings.

If the written vectors are hundreds of times larger than normal `b_in` outputs, the "invisible directions" explanation becomes likely, and the fix is a regularised target. Ordinary sizes favour the reading-bundle leak or a centring problem.

```python
import numpy as np, json
vp = np.load('/work/v4_parts.npz')
c = float(vp['c']); print('restored c =', c, ' v4_parts rbar =', float(vp['rbar']), ' global rbar was', rbar)
print('qI==qs[:,0]?', np.abs(vp['qI']-qs[:,0]).max().round(6), ' qN==qs[:,1]?', np.abs(vp['qN']-qs[:,1]).max().round(6))
sr = float(vp['sig_ref'])
leak_f = sM.mean((0,1))*sr/c
print('sig_ref', round(sr,2), ' meas-bundle leak in f-units (I,N):', leak_f.round(4), ' designed: ins news', (-0.87*leak_f[0]).round(3), 'mg/dL, NE news', (4.3*leak_f[1]).round(3), 'mmHg per bundle')
Ed = rd['E']; bo = Ed @ Wbin.T
print('|Wbin E| at drug bundles mean', np.linalg.norm(bo,axis=-1).mean().round(2), ' sd across bundles of Wbin E (norm of std)', np.linalg.norm(bo.std((0)),axis=-1).mean().round(2))
for nm in ['tI','tN']:
    v = globals()[nm]; print(nm, 'norm', np.linalg.norm(v).round(2), ' vs v4_parts', np.abs(v - vp[nm]).max().round(5))
qE = gp[:,None]*qs; print('|Wbin qE| per channel', np.linalg.norm(Wbin @ qE, axis=0).round(3))
inj_per_f = c/80.5; print('b_in write per unit f (k=1):', (np.linalg.norm(tI)*inj_per_f).round(3), '(ins)', (np.linalg.norm(tN)*inj_per_f).round(3), '(NE)')
st8 = sorted(int(s) for s in de.stay_id.unique()); json.dump(st8, open('/work/st8.json','w'))
print('st8 saved', len(st8))
with open('/work/notes.md','a') as f:
    f.write("- Run 269: drift NOT from insulin injection: rows with no insulin after h12 drift +25.2; drift ~ 3.1 + 1.96*dh - 0.60*cen. x2 shift ~ 7.6 - 0.42*sumr (dose part right sign, weak).\n"
            "  TRUTH on fwd_8 stays (IRF): ins x2 -13.17 (by hour -3.4..-18), NE x2 +4.23.\n"
            "- Run 270-271: head-1 path clean at u_const (f=[.0016,-.0009,.0003]). prelude_b(T,Wkv,Wo)->(x1 after attn, x2 after FFN) EXACT vs rep_drug.\n"
            "  Channel coord s=q.LN(x2): drug bundles mean 0 sd .0006; MEAS bundles (Tm, approx e_k) chI mean .0097 sd .0044, chN -.001 sd .005 -> leak at meas bundles.\n"
            "  WARNING: global c was overwritten in run 270 (restored from v4_parts in 272); global rbar is 0.52 (v4_parts rbar checked in 272).\n"
            "  build_v4(kI,kN,cent=True,save=True) uses globals qs,c,rbar,M,Wo,A,Wbin,tI,tN,gp,kv3. st8 = fwd_8 stays saved /work/st8.json.\n")
```
The ×½ direction is a dead end at low cost. The ×½ edit barely moves the state: its size is 0.085, against 0.87 for ×2. Its out-of-sample D² is only about 0.2, so halved doses look like ordinary factual doses to the model.

| Setting (θ₂, θₕ) | NE ×2 | NE ×½ | Factual cost |
|---|---|---|---|
| (2, 0) | +5.65 | −0.59 | +0.00014 |
| (0, 3) | | −3.85 | +0.026 to +0.029 |
| (−0.66, 1.53), solved | +5.16 | −2.41 | +0.005 to +0.007 |
| truth (400 tuning stays) | +4.47 | −2.24 | |

- **×2 direction:** at θ₂ = 2 it adds about 3.1 per unit to norepinephrine ×2, with no ×½ response and almost no factual cost.
- **×½ direction:** it moves ×½ by about −1.09 per unit θₕ, but costs about 0.0028·θₕ² in factual fit.
- **Solved setting:** it reached roughly both targets but costs too much, so it was correctly not saved.
- **Insulin:** stays put at about −13.7/+6.66 offline in every setting, as it should.

So the full ×½ repair would use up the factual budget. I will take a partial one instead:
- **θₕ = 0.7:** costs about +0.0014. It should move ×½ from −0.6 to about −1.3, out of −2.24.
- **θ₂:** about 0.55, to bring ×2 back to the truth of +4.47, given θₕ's side effect of +4.7 per unit on ×2.

This run evaluates that setting on all 400 tuning stays. It then corrects θ₂ with the measured slope of 3.13 and evaluates again. The edit is saved only if both factual costs stay below 0.002 and insulin is unchanged.

Expected outcome:
- **Confirmed if** norepinephrine ×2 lands within about ±0.3 of +4.47, ×½ comes out near −1.3, and the factual cost is about +0.0015.
- **Rethink needed if** the cost is clearly higher, meaning it is superlinear in θₕ. Then I would lower θₕ.

```python
import time, numpy as np, torch, traceback as tb
T0=time.time()
best=None
try:
    th2,thh=0.55,0.7
    for it in range(2):
        P_,r_,cF_,cE_=evalcfg(th2,thh)
        ok=(max(cF_,cE_)<0.002 and abs(r_['insulin_x2']+13.75)<0.6 and abs(r_['insulin_half']-6.66)<0.4)
        if ok and (best is None or abs(r_['norepinephrine_x2']-4.47)<abs(best[2]['norepinephrine_x2']-4.47)): best=(th2,thh,r_,cF_,cE_,P_)
        th2=th2+(4.47-r_['norepinephrine_x2'])/3.13
    if best is not None:
        th2b,thhb,rb,cFb,cEb,Pb_=best
        for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']: save_edit(n,Pb_[n])
        NEsol=(th2b,thhb); print('SAVED th2',round(th2b,3),'thh',thhb,rb,'costF',round(cFb,5),'costE',round(cEb,5),'edited',len(edited()))
        with open('/work/notes.md','a') as f: f.write(f'- Run288-290 NE plan B: u2 = x2 detector (Sigma^-1 D2 lam1e-4, outD2 97; x2 doses >1.0 off-manifold; NO half response). uh = half dir: |Dh| .085 tiny, outD2h ~0.2 -> half costs ~0.0028*thh^2 (thh=3: cost .026, NE half -3.85). Slopes (400 tuning): x2 per th2 3.13, per thh 4.68; half per thh -1.09. Units k3={k3},k4={k4} (value MLP, mean-ablated), gate LDA map_arterial vs rest (margin G=1.5*smax+10), tilt (centre-74)/26.69 bins1-16.\n  SAVED th2={th2b:.3f} thh={thhb}: offline 400 tuning {rb} (truth ins -13.41/6.71, NE 4.47/-2.24) cost F {cFb:.5f} E {cEb:.5f} (rel. to insulin-only edit).\n')
    else: print('nothing saved')
except Exception: print('err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```
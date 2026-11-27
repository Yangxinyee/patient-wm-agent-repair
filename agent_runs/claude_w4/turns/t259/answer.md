The per‑edit hour profile settles it. In the rollout, the v3 model's insulin ×2 shift at hour 13 is −0.04 mg/dL, against −0.19 for the original. The drug‑bundle injection does not lower predicted glucose at all. The ridge estimate of −1 was an artefact of applying a fitted linear readout to off‑manifold state changes. Everything else stays tiny too, and the u_const leak only adds noise. v3 is dead: I need a carrier with

- exactly zero effect at non‑drug bundles, and
- an injection direction that really moves the glucose (and MAP) belief.

The most principled target is the model's own "glucose news" direction. nuG holds the per‑mg/dL change in the new slots y when a glucose reading comes in higher than predicted; nuM is the same for MAP.

If my stored Jacobian Jm (new slots y with respect to an E perturbation shared across slots) can reproduce −nuG and nuM with small residual, I can inject an E‑direction at drug bundles that makes the state look as if lower glucose (or higher MAP) had been seen. So I'll check the semantics and fit of the stored Jacobians against nuG/nuM.
- A small least‑squares residual would confirm this route.
- A large residual would mean the E path cannot mimic glucose news, and I would have to fall back to direct empirical calibration.

```python
for nm in ['nuG','nuM','dI','dN','Jm','J','Js','Jc','Jx','aI3','aN3']:
    if nm in globals():
        v=np.asarray(globals()[nm]); print(nm,v.shape,'norm %.4g'%np.linalg.norm(v))
Jm_=np.asarray(Jm)
for nm in ['nuG','nuM']:
    tgt=np.asarray(globals()[nm]).reshape(-1)
    sol,res,rk,sv=np.linalg.lstsq(Jm_,tgt,rcond=None)
    print(nm,'rank',rk,'rel resid %.3f'%(np.linalg.norm(Jm_@sol-tgt)/np.linalg.norm(tgt)),'|sol| %.3g'%np.linalg.norm(sol),'sv top/bot',np.round(sv[:3],3),np.round(sv[-3:],5))
    print('  per-slot cos nuG slots', [round(float(cs(np.asarray(globals()[nm])[0],np.asarray(globals()[nm])[s])),3) for s in range(4)])
print('Jm@dI cos -nuG', float(cs(Jm_@np.asarray(dI), -np.asarray(nuG).reshape(-1))), ' Jm@dN cos nuM', float(cs(Jm_@np.asarray(dN), np.asarray(nuM).reshape(-1))))
```
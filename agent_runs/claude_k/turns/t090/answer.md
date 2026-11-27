The dose_edit output gives a shift per stay-hour, with a "reached" flag that is set from the hour after the first edited dose on. For each stay, the shift at the first reached hour has a tiny standard error (about 0.04). It depends only on the first edited dose, with no feedback from earlier sampled readings. So one "both" request will measure two things per stay:
- my probe's real response to a dose change, from the first reached hour;
- how the model carries the resulting shift forward, from the later hours.

Thinking through the rollout exposed a design problem for norepinephrine. Its rate is titrated multiplicatively from the previous rate. Under "x2 from hour 12", only the first edited dose looks like a one-step shock; after that, each doubled rate is consistent with the doubled previous rate. A probe of one-step shocks would therefore fire once and then go quiet, while the true effect builds up under a sustained rate change. My earlier factual-error regression points the same way: the model misses about 2.5 mmHg per unit for shocks up to 4 hours old, i.e. nearly the whole effect of a lasting rate deviation.

So the norepinephrine probe should target the rate's deviation from what the MAP history implies (a level deviation), not the one-step shock. That deviation is larger and lasts, so it may also be decoded with less noise.

I seem to have built policy datasets already (`Xne`, `pred_ne`, `nu_ne`, `nu_log`, `feats_ne`). This run checks what they contain and how the hour columns map, before I build the level-deviation target.

```python
import numpy as np
print("hrs:", hrs)
print("featcols:", featcols)
print("feats_ne:", feats_ne)
print("jcols keys:", list(jcols.keys())[:25])
print("shapes: Xne", Xne.shape, "yne", yne.shape, "pred_ne", pred_ne.shape, "nu_ne", nu_ne.shape, "nu_log", nu_log.shape, "logne", logne.shape, "ne_prev", ne_prev.shape)
on_ = ne_all > 0
print("ne_all on frac %.3f; corr(eps_ne, ne_all - pred_ne) on on-rows: %.3f" % (on_.mean(), np.corrcoef(eps_ne[on_], (ne_all - pred_ne)[on_])[0,1]))
print("corr(eps_ne, nu_ne): %.3f   corr(nu_log, log-shock?): sd nu_log %.3f" % (np.corrcoef(eps_ne[on_], nu_ne[on_])[0,1], nu_log[on_].std()))
print("mean ne_all - pred_ne on on-rows %.4f, sd %.4f; sd eps_ne %.4f" % ((ne_all-pred_ne)[on_].mean(), (ne_all-pred_ne)[on_].std(), eps_ne[on_].std()))
print("eps_ins: corr with ins_all - ? ; sd on insulin-on rows %.3f; mean %.4f" % (eps_ins[ins_all>0].std(), eps_ins[ins_all>0].mean()))
print("sample stay 0 ne_all:", np.round(ne_all[0],2)); print("pred_ne:", np.round(pred_ne[0],2)); print("eps_ne:", np.round(eps_ne[0],3))
print("gbest[0]:", np.round(gbest[0],0)); print("mbest[0]:", np.round(mbest[0],0))
```
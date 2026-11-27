The matched-filter decoder on the head input is too poor to use. Its factual noise is 1.4–2.6 dose units against a lag-1 response of only 0.3–0.5 under ×2, and the NE edits leak into it strongly. The response is also non-linear: under ×2 it is much weaker per unit of dose change than under ×½. Adding this to glucose with the gain the rollout needs (about 3 mg/dL per u/h) would cost far more than the 0.005 factual budget.

The session-5 ridge decoders, trained on samples whose doses had been randomly perturbed, did much better (decode error about 0.57 u/h, small factual gain on set A). Those samples look like `V`/`F`/`Lp_` (39,178 rows = 21,456 finger + 17,722 arterial). Before refitting a decoder on them, I'll check what those columns hold.

```python
np.set_printoptions(suppress=True, linewidth=160)
print("F rows", np.round(F[:3], 3)); print("F col uniques", [len(np.unique(F[:,i])) for i in range(6)])
print("Lp_", np.round(Lp_[:3],3), "spD", np.round(spD[:3],3))
for nm in ['cp_','sp_','spc','ms','ints']:
    a_ = g[nm]; print(nm, a_.dtype, np.round([a_.min(), a_.mean(), a_.max()],3), len(np.unique(a_)))
print("V vs Vp diff norm", np.round(np.linalg.norm(V - Vp, axis=1).mean(),3), "spX vs V", np.round(np.linalg.norm(spX - V, axis=1).mean(),3))
```
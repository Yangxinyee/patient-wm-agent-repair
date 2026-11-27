The replica's head matches `L0` exactly. However, removing unit 97 *lowers* the replica NLL by 0.108, which a trained model should not allow. My reading is that `Xf` comes from the approximate replica, which uses the z prior mean instead of averaging over z draws, so replica-based factual-cost estimates are biased. The safest units to repurpose are therefore ones that are effectively dead: pre-activation strongly negative on every row, so their output is about zero whatever the small replica errors. Next I'll look for such units and report their removal cost and output size.

```python
Apre = Xf @ W1.T + b1
mx = Apre.max(0); q99 = np.quantile(Apre, 0.999, axis=0)
dead = np.where(q99 < -2.5)[0]
print("n units with q99.9 pre-act < -2.5:", len(dead))
o2 = np.argsort(q99)[:10]
print("lowest q99.9 units", o2, np.round(q99[o2],2), "max", np.round(mx[o2],2))
print("their |dn|", np.round(np.abs(dn[o2]),5), "mean |H|", np.round(np.abs(Hh[:, o2]).mean(0),4))
print("small |dn| units", np.argsort(np.abs(dn))[:8], np.round(np.sort(np.abs(dn))[:8],6))
```
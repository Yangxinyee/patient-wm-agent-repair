The paired differences from pass 5 are informative.

- At insulin bundles without norepinephrine, the dose-driven change in `out` is low-rank. The top direction carries 75% of its variance and the top three 96%.
- Its size along that direction depends on rate in a ReLU-like way: about zero for rates 0.5–1.5, a peak of about 0.14 at 2–2.5 U/h, then a decline at higher rates.
- So the action pathway responds only to above-median z and folds over at large z. Input scaling cannot make it linear.
- The edit also moved the state strongly at norepinephrine bundles (out 0.49), which explains the flattened norepinephrine responses.
- The dose-driven change (norm 0.079) is small next to the natural change across insulin bundles (1.29).

The most promising downstream handle is `core.out_norm`, the LayerNorm that produces `out`. All heads and the z context read `out`. Its 1-D gain can be rescaled coordinate by coordinate, and its input is observable, since `out` = γ ⊙ LN(mean of slots) + β.

That only works if a few `out` coordinates carry the dose change while varying little otherwise. Next I compute, per coordinate, the dose-driven change across insulin bundles relative to that coordinate's natural spread over all bundles, and how concentrated the top dose direction is.
- High ratios on a few coordinates mean a selective gain edit is possible.
- Uniformly low ratios mean any amplification there would mostly amplify unrelated variation.

```python
r1 = Vo[0]; r2 = Vo[1]; r3 = Vo[2]
outo = xo[:, 0:192]
for nm_, rv in [('r1', r1), ('r2', r2), ('r3', r3)]:
    pr = outo @ rv
    print(nm_, "natural sd of out.r: all %.3f read %.3f ins %.3f ne %.3f | dose-induced (z x5) range ~%.3f" % (
        pr.std(), pr[kinds=='read'].std(), pr[kinds=='ins'].std(), pr[kinds=='ne_only'].std(), np.abs(DD[mi,0:192]@rv).max()))
d_i = DD[mi, 0:192].std(0)
n_i = outo.std(0)
ratio = d_i / n_i
order = np.argsort(-ratio)
print("top coords by dose/natural ratio:", order[:12], ratio[order[:12]].round(3))
print("n_i of those:", n_i[order[:12]].round(3), " d_i:", d_i[order[:12]].round(4))
print("concentration of r1: top-10 coords share of r1^2 = %.2f" % (np.sort(r1**2)[::-1][:10].sum()))
print("median ratio %.4f" % np.median(ratio))
```
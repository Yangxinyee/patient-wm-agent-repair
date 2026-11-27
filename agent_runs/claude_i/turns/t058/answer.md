The NLL check settles it: my replica averaged over 16 prior draws gives a mean value NLL of 1.4810, exactly the tool's 1.4810. With z at the prior mean it gives 1.4757. So the head replica is correct, and the remaining per-reading differences come from the tool's random z draws. My mean-differences above used the prior-mean version, which is biased by averaging inside the softmax.

Next, I will confirm with the 16-draw average: the per-code error should match pure sampling noise, with no systematic mean difference. Then I'll move on to what matters. I'll take each glucose reading after insulin and look at the head's hidden layer and the state, to see how well the dose deviations (dev_t, dev_t−1) and the decision reading can be linearly decoded. That decides whether a head-level correction is feasible.

```python
mn16 = (pav[:,1:]*np.stack([CEN[e] for e in evs])).sum(1)
d16 = y - mn16
for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    ix = evs==e
    print(e, "MAD", np.abs(d16[ix]).mean().round(2), "mean diff", d16[ix].mean().round(3), "corr(d,mine)", np.corrcoef(d16[ix], mn16[ix])[0,1].round(3),
          "slope", np.polyfit(mn16[ix], y[ix],1)[0].round(3))
# NLL per event
for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    ix = evs==e; print(e, "nll tool", F.nll.values[ix].mean().round(4), "mine", nll_mine[ix].mean().round(4))
```
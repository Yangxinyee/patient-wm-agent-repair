Relative errors of 12–15% look worse than my notes, but the notes may use a different metric (`fitstat`) over all 2,319 drug bundles. I'll run the two best modes on all drug bundles and score them with `fitstat`, so the comparison with the recorded figures (err 39 to 79, R2 0.998) is like for like.

```python
allr = np.arange(len(yt))
for mode in ['ageKV','lnM_ageKV']:
    y_ = step_cv(allr, E0, htf, mode)
    o_ = out_from_y(y_)
    print(mode, 'fitstat y', fitstat(y_, yt), 'out', fitstat(o_, ot), 'rel %.4f' % float((y_-yt).norm()/yt.norm()))
```
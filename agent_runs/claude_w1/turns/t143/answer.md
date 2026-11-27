Run 133 changed the picture.
- **Decoders:** trained on perturbed doses, they work. Factual decoding error sd is about 0.53 u/h for insulin and 0.18 for norepinephrine, with slopes 0.8–0.9.
- **Sign:** the replica residuals load on (decoded dose − rule expectation) with the wrong sign: +2.5 for insulin, −2.2 for norepinephrine. A decoded-dose correction would then hurt the factual fit.
- **Replica reliability:** the replica residuals are 18% (glucose) and 13% (MAP) linearly predictable from the head input x. That suggests my approximate head replica, not the real model, is producing much of this structure. Its mean was also biased by +3.3 mg/dL.

So I can't judge the factual-fit cost from replica residuals. I must use the real model's teacher-forced means, which I already have for setA from fwd_2.

Next I'll redo the same analysis on setA using the tool's predicted means:
- residuals on rule innovations;
- residuals on decoded dose minus rule expectation (decoded from my replica head inputs for those rows);
- CV R² of residuals on x.

How to read the outcome:
- Tool residuals barely predictable from x, but loading negatively on insulin innovations: the earlier sign problem was a replica artifact.
- The positive loading on the decoded dose persists: I need a different edit design.

First I need the layout of the stored tool outputs, so this run also prints it.

```python
for nm_ in ['F2','f2','fa2']:
    print(nm_, list(g[nm_].columns), g[nm_].shape); print(g[nm_].head(3).to_string()[:600])
print('wc_', wc_.shape, 'prow', prow[:5], 'obs', obs[:5], 'pm', pm[:5].round(2), 'err', np.round(err[:5],2), 'mA', mA[:5], 'mG', mG[:5])
print('lastv', lastv[:5], 'lastage', np.round(lastage[:5],3))
import glob; print(glob.glob('/work/fwd_2/*'), glob.glob('/work/fwd_1/*'))
```
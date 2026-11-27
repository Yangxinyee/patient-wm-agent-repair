The record-based truth reproduces my notebook figures (-13.07 and +6.54). By rate band it is x2 [-6.3, -16.0, -24.7] and half [+3.2, +8.0, +12.4], linear in dose as expected. The model is nowhere near that at mid and high doses, and N5 makes the top band worse (-0.98 versus the original's -2.70). Next is the same banded comparison for insulin x1/2. I'm checking whether the saturation is specific to x2's large upward z-shift; if so, a milder numeric factor should balance the two readouts better.

```python
for k,m in [(1,'original'),(6,'edited'),(9,'edited'),(25,'edited')]:
    print(k, band_tab(f'/work/fwd_{k}/dose_edit_{m}.csv','insulin_half'))
```
Heads 1 and 2 at x4 weaken the insulin response in every band, and head3 adds only a little at low doses. Head0 is the only head that amplifies mid/high-dose responses, and its amplification is what costs factual fit. So there is no cheaper head to use.

On insulin, N5 alone remains the safe choice. Next I'll look at norepinephrine by rate band (≤0.5, 0.5-1, cap) for the original and N5. I want to see whether N5's wrong-signed x1/2 (+0.06) comes from cap stays or is spread evenly. If one band drives it, a narrow fix there might restore the correct sign cheaply.

```python
ne=tu[(tu.event=='norepinephrine')&(tu.hour>=12)].groupby('stay_id').rate.mean()
cuts=[0,0.5,0.99,1.01]
for k,m in [(1,'original'),(6,'edited'),(18,'edited'),(25,'edited')]:
    f=f'/work/fwd_{k}/dose_edit_{m}.csv'
    print(k, band_tab(f,'ne_x2',ne,cuts)[0], band_tab(f,'ne_half',ne,cuts)[0])
```
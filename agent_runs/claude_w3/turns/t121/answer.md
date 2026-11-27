e_k does carry correct information: the error rises from 0.10 to about 0.19 with a zero or wrong-bundle e_k. The leftover error must come from somewhere else, possibly the time-decay convention. The timescales are stated as 10 minutes to 2 years, so I'll check the range of `core.flow.a` to see whether exp(a) is a rate per second or a timescale. I'll also check which embedding rows are trained (norm ≠ 0.28).

```python
fa_ = P0['core.flow.a'].numpy()
print('flow.a range', fa_.min().round(3), fa_.max().round(3), ' exp(-a) s range', np.exp(-fa_).min().round(1), np.exp(-fa_).max().round(1))
print('exp(a) range', np.exp(fa_).min(), np.exp(fa_).max())
for t_ in ['embed.kq.weight','embed.kind.weight','embed.ekind.weight','embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.text.weight','embed.categorical.status.weight','embed.bins.value.weight']:
    print(t_, np.round(P0[t_].norm(dim=1).numpy(), 2))
```
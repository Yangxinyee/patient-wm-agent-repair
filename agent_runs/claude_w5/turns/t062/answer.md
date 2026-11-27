The error per bundle swings from 0.1 to 2.4 under every setting, so a systematic piece is still wrong. Next I'll print, for the first 13 bundles of this stay, the hour, time gap, codes and values. I'll also check the range of `exp(flow.a)` to confirm the decay rates are per second. I'm looking for whether the large errors go with particular codes (for example lab glucose or arterial MAP) or with particular gaps.

```python
print('flow rate range', float(flow.min()), float(flow.max()), '1/tau range', 1/63115200, 1/600)
for bi in range(1,14):
    t_,ats=bundles_B[s0_][bi]; dt=(t_-bundles_B[s0_][bi-1][0])*3600
    print(bi, t_, int(dt), [(a_['code'][:6],a_['kind'],a_.get('value',a_.get('rate'))) for a_ in ats], zerr(bi,True,'logs'))
```
`V`, `Vp` and `spX` are the same array, and `F`, `Lp_` and `spD` hold the same doses, so there is only one sample set. Decoders fit on it barely respond to the edit responses stored in `EDR`. Either those samples carry recorded rather than perturbed doses, or `EDR` came from a different version of the replica. I don't want to trust either stored set further. Instead I'll regenerate head inputs myself with the exact free-run replica, once with recorded doses and once with doses doubled from hour 12. First I need the record format and the per-stay data objects (`G3`, `H100`).

```python
print('G3' in g, type(g.get('G3')).__name__, 'H100' in g, (len(g['H100']) if 'H100' in g else None))
with open('/data/train.csv') as fh_:
    for _ in range(4): print(fh_.readline().strip()[:150])
if 'G3' in g:
    k0_ = list(G3.keys())[0] if isinstance(G3, dict) else None
    d0_ = G3[k0_] if k0_ is not None else None
    print(type(d0_).__name__, getattr(d0_, 'shape', None), list(getattr(d0_, 'columns', []))[:12])
```
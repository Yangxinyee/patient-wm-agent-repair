Run 98 gets drug bundles to about 17% error at best: no unit or rate_unit combination makes them exact. The best settings leave out the rate and dose on END atoms. The best unit choice is 3 for both drugs, but the differences between variants are small. So something structural in how drug atoms are embedded is still off.

To locate it, I'll group the drug-bundle errors by composition: a single new record, a single END atom, END plus new record, one drug or both. If, say, bundles with a lone new record are exact but those with END atoms are not, the END atoms are the problem. I'll run this over more stays so that rare compositions show up.

```python
o_best = {'unit': dict(unit_meas, insulin=3, norepinephrine=3), 'rate_unit': {'insulin': 2, 'norepinephrine': 3}, 'side': 'left', 'end_attrs': False}
print(zp_full(o_best, list(bundles_B.keys())[:40], 'drug', by_comp=True))
```
Run 97 makes the measurement side exact. glucose_finger takes no unit (index 0), glucose_lab takes unit index 5, and MAP readings take index 2. With these, glucose bundles match the recorded z posterior exactly, like the MAP bundles. That leaves unit indices 3 and 4 as the likely drug units.

Next are the drug bundles, where the error is about 24%. I'll search:
- the unit index for insulin and norepinephrine (0, 3, 4);
- the rate_unit index (0, 2, 3);
- whether END atoms carry their rate and dose into the embedding.

An exact match on drug bundles would complete the encoder side.

```python
unit_meas = {'glucose_finger':0, 'glucose_lab':5, 'map_arterial':2, 'map_cuff':2}
rd2 = []
for iu, nu in [(3,4),(4,3),(0,0),(3,3),(4,4)]:
    for ir, nr_ in [(2,3),(3,2),(0,0)]:
        for end_attrs in [True, False]:
            o_ = {'unit': dict(unit_meas, insulin=iu, norepinephrine=nu), 'rate_unit': {'insulin': ir, 'norepinephrine': nr_}, 'side': 'left', 'end_attrs': end_attrs}
            rd2.append((zp_full(o_, list(bundles_B.keys())[:10], 'drug', empty_e='enc', max_n=100), iu, nu, ir, nr_, end_attrs))
rd2.sort(key=lambda t: t[0][0]); print(rd2[:6])
```
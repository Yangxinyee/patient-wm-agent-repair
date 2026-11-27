`atom_vec`, `set_enc` and `mlp` exist from an earlier restart, so I may already have an offline version of the atom embedding and set encoders. The relevant facts for the design are:
- Only insulin and norepinephrine atoms ever pass through `encode_action`, so its mlp1 sees a small, enumerable set of inputs.
- Norepinephrine dose is exactly 4.8 × rate, so z_dose equals z_rate.
- Norepinephrine has its own rate unit (mcg/kg/min), which helps the gate.

Before building an mlp1 correction on these helpers, I need to know whether they were ever checked against the real model. I also need to know whether END atoms carry rate and dose. A wrong atom vector would misplace the gate.

I will search my notebook for what I recorded about them, and read the data README's description of the drug rows.

```python
nt = open('/work/notes.md').read()
for l in nt.split('\n'):
    if re.search(r'atom_vec|set_enc|first_err|encode|END atom|atom vec|mlp1|kq', l): print(l[:220])
rd = open('/data/README.md').read(); print(len(rd))
for l in rd.split('\n'):
    if re.search(r'END|end_hour|rate|dose|unit', l): print(l[:200])
```
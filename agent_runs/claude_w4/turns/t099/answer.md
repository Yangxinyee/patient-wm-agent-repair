These numbers are the calibration targets on the 100 tuning stays, using the tool's reached rows (each has an outcome reading in that hour). Under my estimated impulse responses:
- insulin x2 should move finger glucose by about −13 mg/dL, and x1/2 by +6.5;
- norepinephrine x2 should move arterial MAP by about +4.9 mmHg, and x1/2 by −2.5;
- the targets are linear in the multiplier by construction.

The original model gives −0.35, +0.21, +0.02 and −0.25, so both drugs need a large repair.

Next I rebuild the injection components. The first piece is a gate and rate readout from u_k (the drug-encoder output) that works for insulin-only, NE-only and mixed bundles. That needs my exact u_k replica, so I check the replica functions' signatures and record the targets in the notebook.

```python
for fn in ['ins_atom','ne_atom','u_set','u_from_atoms','enc_set','u_bundle','drug_atom_parts','ctx_of','e_drug_bundle','gelu','LNf','mlp_','zsc','first_dose_from12','true_shift']:
    try: print(fn, inspect.signature(globals()[fn]))
    except Exception as ex: print(fn, 'ERR', ex)
with open('/work/notes.md','a') as f_:
    f_.write("""
## Runs 79-84 (restart 2)
- dose_edit summary mean_shift = POOLED mean over 'reached' rows (stay-hours with an outcome reading, after drug ran from 12).
- IRF-based TRUTH on sel100 reached rows (hF finger, hA art): insulin x2 -12.96, x1/2 +6.48 (x0 +12.96, x1.5 -6.48, x3 -25.9);
  NE x2 +4.94, x1/2 -2.47 (x0 -4.94, x3 +9.88). Original model: -0.35, +0.21, +0.02, -0.25.
""")
```
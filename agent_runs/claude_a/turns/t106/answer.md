Combined kills are not additive, but one group works well. K8b (dimensions 155, 100, 128, 151, 171, 52, 130, 99) improves the factual NLL by 0.0124 per reading, and every reading code improves:

| Code | ΔNLL |
|---|---|
| map_cuff | −0.022 |
| glucose_finger | −0.0105 |
| map_arterial | −0.0118 |
| glucose_lab | −0.009 |

Groups that include dimension 144 or 73, or more than 10 dimensions, lose most of that gain.

**Choice.** I will build four antisymmetric memory pairs from K8b: (155,100), (128,151), (171,52), (130,99), giving fast and slow channels for insulin and for norepinephrine. That leaves about 0.012 nats of margin for the readout. This gain still has to be confirmed on held-out stays with the tool.

**Why I need more data first.** To set the decay rates and readout weights by regression on teacher-forced residuals, I need far more than 100 stays. Each dose-deviation coefficient has a standard error of roughly 15/(0.5·√N).

**This run.** Rebuild the full teacher-forced inputs for any set of training stays: bundles, e and u encodings, and the reading register features. I validate the builders on the sel2 stays against the arrays I already trust.

**How I will read it.** An exact match (e/u errors near 1e-6, identical register features and bins) confirms the builders. Any mismatch tells me which feature convention to fix before scaling up.

```python
t0_ = time.time()
try:
    atoms_s, bund_s = make_atoms(tr, list(stays_SB))
    print('make_atoms ->', type(atoms_s).__name__, getattr(atoms_s, 'shape', None), type(bund_s).__name__, getattr(bund_s, 'shape', None), round(time.time()-t0_, 2), 's')
    print('atoms cols', list(atoms_s.columns)); print('bundle cols', list(bund_s.columns))
    print(bund_s.head(3).to_string())
    same_b = (bund_s['stay_id'].values == SB['stay_id'].values).all() and (bund_s['minute'].values == SB['minute'].values).all()
    print('bundle order identical to SB:', same_b)
    t1_ = time.time()
    eo_ = encode_bundles(atoms_s, list(range(len(bund_s))))
    print('encode_bundles ->', type(eo_).__name__, [getattr(t, 'shape', None) for t in eo_] if isinstance(eo_, tuple) else None, round(time.time()-t1_, 2), 's')
    if same_b:
        print('e err', (eo_[0] - eALL).abs().max().item(), ' u err', (eo_[1] - uALL).abs().max().item())
except Exception as ex:
    import traceback; traceback.print_exc()
print('Rr head:'); print(Rr.head(3).to_string())
print('hasl/ages/zl/lvb sample', hasl[:3], ages[:3], zl[:3], lvb[:3])
```
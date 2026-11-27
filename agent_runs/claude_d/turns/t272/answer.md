The regression is decisive.
- About 81% of the x2 target (3.97 of 4.87 mg-equivalent MAP units) comes from NE records whose doubled rate exceeds 1.0.
- The edited model's coefficient on that part is 0.06, essentially no response. Its coefficient on the in-range part is 1.54 (half edit: 1.81, intercept 2.26).

So my head channel works for in-range doses, but atoms pushed into bin 16 with z beyond the training range give the downstream state nothing it reads as "more NE". The fix belongs in `encode_action.mlp1`, specific to NE atoms in bin 16. These never occur in the records, so in principle the fix has zero factual cost.

Next I will rebuild the NE atom vectors offline over a rate grid 0.02–2.0 and get the bin tables and code/unit indices. Then I can compare the encoder output for the original, a "bin-capped" version (bin 16 → top NE bin, numeric z extended) and a linear extrapolation. A smooth, roughly linear in-range trend that breaks at bin 16 would support the capped-bin design.

```python
open('/work/notes.md','a').write("""
## Restart 9 (runs 239-245)
- fwd11 per-stay-hour regression (lag = h-floor(t) reproduces true_shift_ne; gam_ne_clean padded to 13): ne_x2 target 4.87 = in-range part .90 + OUT-OF-RANGE part 3.97 (doubled rate>1).
  model shift ~ -.58 + 1.54*tin + 0.06*tout  -> out-of-range NE atoms give ~ZERO response. half: shift ~ 2.26+1.81*tgt.
- mlp1 = Linear(192,768)-GELU-Linear(768,192), no pre-norm. Plan: gated NE&bin16 correction units in encode_action.mlp1 (zero factual cost).
""")
print([k for k in VOCAB.keys()])
print(VOCAB['codes'] if isinstance(VOCAB['codes'],(list,dict)) else type(VOCAB['codes']))
for k in VOCAB:
    v=VOCAB[k]
    if isinstance(v,dict): print(k, list(v.keys())[:12])
```
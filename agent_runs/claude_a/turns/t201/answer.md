Both drugs are recorded as hourly 1-hour infusions. Every row starts at minute 5 (hour + 0.083) and ends exactly one hour later, and insulin and NE share the same bundle when both are given. Rates (5/50/95%) are 0.5/1.5/3.5 for insulin and 0.05/0.46/1.0 for NE.

So the same circuit fits both drugs. Each drug bundle is one hour of dosing, which makes an additive memory of decayed rates the natural state, with no level-replacement logic needed. MAP has two codes (cuff and arterial); NE's evaluation reads map_arterial.

**Next.** I will reuse my exact prelude replica in /work/lib2.py to fit the head-3 staging decoder, including synthetic doubled and halved insulin encodings so the decoder stays linear out to 2x rates. That needs its exact interface, so I will print the file (about 3.3k chars). I will also print the shapes of the atom-encoder first layers, since the gate dim must be cut from their input columns.

**What would change the plan.** If `prelude()` returns the attention weights and per-head values, the decoder and gate fits can be done offline directly. If not, I will extend it.

```python
print(open('/work/lib2.py').read())
print({n:SHAPES[n] for n in NAMES if n.startswith(('encode.mlp1','encode_action.mlp1'))})
```
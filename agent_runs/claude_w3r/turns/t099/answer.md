Run 92 (summary only, to keep this short):
- **Policy:** with a fresh glucose reading, the insulin rate is set almost entirely by the glucose band. Without a fresh reading it is almost always off.
- **Deviation decodability:** measured against this proper policy expectation, the head's inputs barely encode the dose deviation. The CV R2 is 0.05 with a slope of only 0.08 on the true deviation; the lag-2 deviation gives 0.01. The raw rate decodes well (R2 0.73), but only because the band predicts it.
- **Model's response:** the model's predicted glucose still moves with the deviation, with the wrong sign (+1.2 per unit).

So a head-only edit cannot work. It would need a ~13× rescaling, which would mostly add noise and badly break the factual fit. The dose information is lost in the core.

Any working edit must open a channel upstream: action encoder → core state → head. I'll record this and inspect the core's tensor structure to find where such a channel can go.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- Run 92 (panel policy cells): insulin given ~only when a FRESH gf reading that hour: band means 0/1.03/2.07/3.11/4.15 (P on ~.99-1); not fresh -> P on ~.02.
  With proper policy mean, dev1 sd .56: probe R2 from with_code only 0.05 (slope k .076); dev2 R2 .01. Raw rate R2 .73 (via band).
  Model pm slope on dev1 = +1.2 (wrong sign). => HEAD-ONLY EDIT NOT VIABLE (needs 13x rescale -> big NLL hit).
  Need upstream channel carrying the insulin (and norepi) rate into the head-time state. Next: inspect core structure, dead units.
""")
for n in NAMES:
    if n.startswith(('core.','encode_action','z_post','z_in','encode.')):
        print(n, SHAPES[n])
```
Two things came out of this:
- The 226 END-only bundles are now encoded exactly.
- The 200 admission-containing bundles still failed. That covers the 100 hospital_admission bundles and all 100 icu_admission bundles, which also hold readings. So `meas_parts` breaks on visit codes, and something in those mixed bundles fails too.

I'll check the vocab for the event-kind and categorical tables so I can build a visit-event atom with `atom_emb`.

```python
print(list(VOCAB.keys()))
for k in VOCAB:
    if any(s in k.lower() for s in ['kind','categ','unit','visit','event']):
        v = VOCAB[k]; print(k, v if len(str(v)) < 300 else str(v)[:300])
```
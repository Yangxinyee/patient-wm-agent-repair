fwd_3 holds states for the same 100 training stays as the fwd_2 factual pass: 5156 bundles, each with out, the 4 slots of h and the z posterior mean. So I can rebuild the value head offline and check it against the tool's 5306 predicted means. The value head reads only the previous bundle's state (time-flowed), z draws from the prior, the type and code embeddings, and the "last value" register.

Next, I will read the vocabulary: codes, event and atom kinds, the numeric normalisation, the bin edges and centres, and their notes. These fix the code index for each event, how z-scores and bins are computed, and how the predicted mean is formed. If the notes match the architecture text, I can write the head replica directly. If they reveal extra register features (`last_numeric` has 8 inputs, not the 4 documented), I'll test the variants against fwd_2.

```python
for k in ['codes','event_kinds','atom_kinds','numeric_note','value_bin_note']:
    print(k, json.dumps(VOCAB[k])[:900])
print('numeric', json.dumps(VOCAB['numeric'])[:2500])
print('centres', json.dumps(VOCAB['value_bin_centres'])[:2500])
```
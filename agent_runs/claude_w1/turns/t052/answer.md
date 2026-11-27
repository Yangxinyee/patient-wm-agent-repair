The vocab answers several open questions:
- Readings are binned per code into 16 fine bins; the predicted mean is Σ p(bin) × centre.
- z-scores are (x − median)/IQR, clipped.
- Code ids: map_cuff 5, glucose_finger 6, map_arterial 7, insulin 8, norepinephrine 9, glucose_lab 10.
- `type_emb` has 15 rows, which probably means 3 atom kinds × 5 event kinds.

The factual file gives `predicted_mean` per reading. That is a clean target for checking a local head replica built from the states.

Before writing the replica I need the remaining tables: the bin edges for the glucose and drug codes, and the format of the bin centres.

```python
num={ (e['code'],e['attribute']):e for e in VOCAB['numeric']}
for key_,e in num.items():
    print(key_, 'med',e['median'],'iqr',e['iqr'],'edges',e['fine_bin_edges'][:3],'...',e['fine_bin_edges'][-3:], 'n',len(e['fine_bin_edges']))
vc=VOCAB['value_bin_centres']
print(type(vc), (list(vc.keys())[:10] if isinstance(vc,dict) else len(vc)))
k0=list(vc.keys())[0] if isinstance(vc,dict) else 0
print(k0, str(vc[k0])[:300])
```
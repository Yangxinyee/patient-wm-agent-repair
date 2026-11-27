The labels exist. hasI_d and hasN_d mark insulin (70%) and NE (89%) presence at the 2,319 drug bundles; RI_d and RN_d hold rates, NaN when absent; dr3 maps drug bundles to rows of the 5,218-bundle state file. To build the dose-surprise target (rate minus the sliding-scale band mean of the latest glucose) I need each drug bundle's stay and hour, plus the tuning records. I'll check the state-file keys and which kernel dataframe holds the tuning stays.

```python
print('S3 files', S3.files)
for nm in ['df','dfs','df_s','tu','TXT','drugs_tu','panel','wide','hist']:
    v = globals().get(nm, None)
    if v is None: continue
    if isinstance(v, pd.DataFrame): print(nm, 'DataFrame', v.shape, list(v.columns)[:14])
    else: print(nm, type(v).__name__, len(v) if hasattr(v,'__len__') else '')
```
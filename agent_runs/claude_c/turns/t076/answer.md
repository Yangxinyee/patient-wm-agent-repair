Run 67 shows the raw state (four slots of y) is large in scale and changes at reading bundles within only about 25 effective dimensions. Along the plain dose direction, the dose-induced change (sd 12) is swamped by reading-bundle changes (sd 135). The generalized direction reaches a held-out ratio of about 1, but only in low-variance directions that the heads may not read. That reading-bundle variation also partly contains the decay of what drug bundles wrote earlier. Any write-side amplification would also scale the constant part of the write and so shift the state at every bundle. Write-side edits look risky.

The dose inputs themselves (z-scores centred at the median, plus bins) are the only place where amplification is naturally centred and limited to drug atoms. Before choosing the next probe, I need the exact input encoding:
- the per-code median and scale for the drug z-scores, and the bin edges for insulin and norepinephrine rate and dose;
- how far ×2 and ×½ move doses in z and in bins, on the tuning records.

That tells me how out-of-range the edited doses are and which input features carry the edit.

```python
cfg=VOCAB.get('config',{})
print('vocab keys:',list(VOCAB.keys()))
print('config:',{k:cfg[k] for k in list(cfg)[:40]})
for key in VOCAB.keys():
    if key in ('codes','config'): continue
    val=VOCAB[key]
    if isinstance(val,dict):
        print('==',key,'subkeys:',list(val.keys())[:20])
codes=VOCAB['codes']; print('codes:',codes)
def show(key):
    val=VOCAB[key]
    for c_ in ['insulin','norepinephrine','glucose_finger','map_arterial']:
        if isinstance(val,dict) and c_ in val: print(key,c_,val[c_] if not isinstance(val[c_],list) else np.round(val[c_],3))
for key in VOCAB.keys():
    if key not in ('codes','config') and isinstance(VOCAB[key],dict):
        show(key)
```
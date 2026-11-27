The innovation is partly present at the insulin bundles themselves:
- z posterior: R² 0.41, or 0.57 once the glucose band is added;
- `out` and `h`: R² about 0.35.

So the information enters the network, but it fades within an hour; at lags 2 and later, R² is about 0.05.

A readout edit can therefore still use the lag-1 information. The plan:
- Add a rank-1 term to the value head's last layer that reads a decoded "insulin innovation" feature and tilts the glucose bins.
- Let the model's own feedback of its drawn glucose carry the effect forward.
- Calibrate the gain with dose_edit.
- Do the same for norepinephrine on the MAP bins.

To fit this I need a local replica of the value head that runs from the returned states. I can validate it against the per-reading predicted means in the fwd_2 factual file. First I'll look at the vocab notes and that file's layout, kept short.

```python
for kk in ['event_kinds','atom_kinds','numeric_note','value_bin_note']:
    print(kk, str(VOCAB[kk])[:300])
print('numeric', str(VOCAB['numeric'])[:400])
print('codes', VOCAB['codes'])
fa2=pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa2.shape, list(fa2.columns)); print(fa2.head(3).to_string()[:600])
```
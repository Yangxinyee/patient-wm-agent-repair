The probe states pass is back: 5,353 bundles per model. I will now check that the "original" rows match my saved states (XALL) for these stays, which validates the row mapping. Then I rebuild the head inputs at every reading from both state sets and take the difference.

The key question is how strong the injection's footprint in with_code is at map_arterial readings one, two and three bundles after an NE bundle, compared with natural variation. First I need to recall how my HI / ctx_of / zpr / wc_from pipeline is wired, so this run inspects it and does the alignment check.

```python
P0=np.load('/work/fwd_12/states_original.npz'); P1=np.load('/work/fwd_12/states_edited.npz')
print(list(P0.keys()), P0['x'].shape)
print('HI', type(HI), (list(HI.keys()) if isinstance(HI,dict) else len(HI)))
print('RD cols', list(RD.columns), 'STALL cols', (list(STALL.columns) if hasattr(STALL,'columns') else type(STALL)))
print('ctx_of names', ctx_of.__code__.co_names, 'zpr names', zpr.__code__.co_names)
st=STALL[STALL.stay_id.isin(selP)].sort_values(['stay_id','minute']) if hasattr(STALL,'columns') else None
o=np.lexsort((P0['hour'],P0['stay_id'])); print('npz sorted?', np.all(o==np.arange(len(o))), 'n STALL rows', len(st))
```
Projecting out last_numeric does not help. On setA's head inputs, both decoders vary within cells by 4-13, against 0.3-1.3 on the train-replica rows. These high-norm decoders (|w| 64-373) are hypersensitive to small differences between head-input constructions, so they would behave unpredictably in the real model. I will drop them.

A second puzzle: the ridge decoders decode perturbed doses well (slope about 0.86) but respond only 16% to the replica's x2 edit. Before choosing a robust low-norm decoder I need to know which is right. I will check the replica interfaces so I can recompute, for a few train stays, the head query under recorded versus insulin-x2 doses with the exact core replica.

```python
for fn in ['bundles_of_stay','run_stay','gen_rows_all','gen_samples','dose_arr']:
    f_=globals()[fn]; print(fn, str(inspect.signature(f_))[:150], f_.__code__.co_names[:14])
print(type(G3[selT[0]]).__name__, list(G3[selT[0]].columns)[:12] if hasattr(G3[selT[0]],'columns') else '')
```
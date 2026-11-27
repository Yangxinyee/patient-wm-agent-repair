Run 196 shows that EDR stores, for each of the four dose edits, the replica's per-row change in head input (dX) and in dose (dd). solve_unit fits a decoder direction with unit response to its own drug's edits, zero response to the other drug, and minimum factual noise (cell-residualised Rf). vI and vN look like finished decoders of that kind. To check, I will look at EDR's structure and the code ids, then run diag on vI and vN.

```python
e1=EDR[('insulin',2.0)]
print({k:(getattr(v,'shape',None) or len(v)) for k,v in e1.items()})
print('cf uniq',np.unique(cf)[:10],'C4',C4,'CODES4',CODES4)
print('ins_rows codes',np.unique(cf[ins_rows]),'map_rows codes',np.unique(cf[map_rows]))
```
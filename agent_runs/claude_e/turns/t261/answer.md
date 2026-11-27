On tu1 (2020000200..299) the −2.6 edit gives insulin x2 −12.47 and x1/2 +7.19, against the original model's −1.95 and +1.01. NE is unchanged at −0.11 and −0.54. Unlike f3, here x1/2 is larger than half of x2, so the asymmetry depends on which stays are in the batch.

To judge calibration I need my record-based truth for these stays and stay-hours. I will reuse `truth_ins` and `truth_ne`, the functions that built the earlier truth estimates from the record-based response curves. First I check their signatures and try them on the fwd_22 rows.

If the edited/truth ratio for x2 is again about 0.95–1.0, −2.6 is confirmed. A ratio well off that range would mean the batch-to-batch noise is larger than the remaining calibration error. In that case the right choice is the gain that is best on average over both batches.

```python
print(inspect.signature(truth_ins), inspect.signature(truth_ne), inspect.signature(truth_rows) if 'truth_rows' in globals() else '')
print(truth_ins.__code__.co_varnames[:truth_ins.__code__.co_nlocals][:20])
e22=pd.read_csv('/work/fwd_22/dose_edit_edited.csv'); o22=pd.read_csv('/work/fwd_22/dose_edit_original.csv')
try:
    r_=truth_ins(e22); print(type(r_)); print(r_ if not hasattr(r_,'head') else r_.head())
except Exception as ex:
    import traceback; traceback.print_exc(limit=1)
```
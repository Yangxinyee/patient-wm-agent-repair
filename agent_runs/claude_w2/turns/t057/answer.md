The search finished, but sorting the results failed: ties in error led Python to compare `None` with integers. I sort on the error alone and print the best variants. The `res_` list from the last run is still in memory, so nothing needs to be recomputed.

```python
res_.sort(key=lambda t: t[0]); print(res_[:10]); print(res_[-3:])
```
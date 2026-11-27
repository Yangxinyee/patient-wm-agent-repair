The dose-edit file has one row per stay-hour and edit, with the shift, its standard error, the factual prediction and a `reached` flag. The edits are insulin x2/x1/2 and norepinephrine x2/x1/2.

To split the readouts by norepinephrine cap status and by insulin dose level, I need each tuning stay's recorded drug rates from hour 12 on. This run shows the layout of the tuning records and the exact edit labels, so I can build those per-stay attributes in the next run.

```python
print(de1['edit'].unique())
print(list(tu.columns)); print(tu.head(4).to_string())
```
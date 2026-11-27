Training batch 0 at −2.6 costs +0.00164 per reading. glucose_finger is +0.005 per row; the other events are near zero. Across five batches the estimates are now 0.0043, 0.0032, 0.0041, 0.0010 and 0.0016, with a mean of about 0.0028. That is lower than the tuning-only estimate.

I will run the other three training batches to pin down the mean. That needs their stay ids, so this run logs the result and prints the ids. If the full mean stays around 0.003 or below, I keep −2.6.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- Run 228: saved ins_only(-2.6) [hash 924a84db1e] (W0,b0 same as -2.5; W2 differs). TR original.
- fwd_17 factual tu2 (2020000300..399) at -2.6: +0.00100 (gf +.0058/row, gl -.0043). Batch spread large: tu0 .00428 (-2.6), f3 .00283, tu1 .00369 (-2.5).
- Run 230 bootstrap over 400 tuning stays: -2.6 mean .00314, P(>.005) N=100 10%, N=200 4%, N=400 .65%; -2.5 mean .0028 (P 4.7%/1.25%/.08%).
- Run 231: cost NOT from outliers (51% from |dpm| 2-4 rows); residual slope on dpm .156 (factual optimum gain ~.16x current). No cheap cost cut.
- Run 232: /work/sel_cost.json = 4 batches x 100 unused train stays. fwd_18 (batch0) at -2.6: +0.00164 (gf +.0052/row).
""")
for i in [1,2,3]: print(i, json.dumps(batches[i]))
```
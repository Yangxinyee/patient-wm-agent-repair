The training records settle the bin question. NE never exceeds rate 1.0 or dose 4.8, so NE never reaches bin 16 in training. Insulin does reach bin 16 (0.7% of records, rates up to 14.5 U/h), so that row is trained, but only by insulin.

Under x2, 54% of NE atoms in hours 12-24 jump to bin 16, which gives an NE token carrying an embedding learned from high-dose insulin. That is a plausible cause of the wrong-signed NE x2 response. One candidate fix is to set the rate and dose bin-16 rows equal to row 12 (the NE cap bin) using three rank-1 row-mixing ops per table. Its factual cost would fall only on rare insulin >5 U/h bundles.

Before spending a pass, I will check that the model's NE x2 response really goes wrong in the OOD stays. I will split the original per-stay-hour NE x2 shifts by the stay's NE rate band (r≤0.5 stays in range, 0.5<r<1, r=1 cap). First I need to find the dose_edit file layout.

```python
print(sorted(os.listdir('/work/fwd_1')))
print(list(de_o.columns), de_o.shape)
print(de_o.head(3).to_string()[:600])
```
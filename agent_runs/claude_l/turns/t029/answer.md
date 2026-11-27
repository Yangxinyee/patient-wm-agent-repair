The bundle layout matches my reading of the format. END atoms come at start + 1 h and share a bundle with the next infusion start. For example, 5.0833 holds insulin, insulin END, NE and NE END, which is 4 atoms. An END with no new start (8.0833, after insulin stopped) stays in the infusion-minute bundle, and `has_insulin` flags only a new insulin record. Stays have up to 54 bundles.

Next I'll print the full event list next to the bundle rows for stay 2020000000, and the tail of a full-length stay, to find:
- what makes up bundles 53–54 (a final END at 25.08 and possibly a stay-end bundle at 25.17);
- how n_atoms counts at hours −1 and 0, including the admission rows.

Then I'll write a bundle builder and check that it reproduces every stay's bundle count and atom counts.

```python
r = tu[tu.stay_id==2020000000].sort_values('hour')
print(r[['hour','event','value','rate','dose','end_hour']].head(12).to_string())
print(r[['hour','event','value','rate','dose','end_hour']].tail(8).to_string())
print(sx[sx.stay_id==2020000000].tail(8).to_string())
s16 = sx[sx.stay_id==2020000016]; print(len(s16)); print(s16.tail(6).to_string())
r16 = tu[tu.stay_id==2020000016].sort_values('hour'); print(r16[['hour','event','value','rate','end_hour']].tail(8).to_string())
```
The inspection shows the edit is clean and does what I intended:
- In both bin tables, rows 1..16 are now all equal to the original row 8, the median bin. Row 0, the "absent" row used by every non-drug atom and by the heads, is untouched.
- In the numeric weight, only the z_dose and z_rate columns were zeroed. z_value, has_value, has_dose and has_rate are unchanged.

So each drug atom now gets the embedding of a median-dose atom of its own code, whatever its rate. Measurement atoms and the glucose/MAP heads are unaffected.

The deciding question for this plan is whether removing dose magnitude costs factual fit. The limit is 0.005 nats per reading, and I want a clear margin. I will run a paired factual pass, original vs edited, on the same 100 tuning stays I used before. First I need the stay ids written out, so this run just prints them compactly.

```python
print(json.dumps(sel1))
```
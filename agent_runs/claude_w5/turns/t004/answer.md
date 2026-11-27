That run showed the code ids:
- 5 map_cuff, 6 glucose_finger, 7 map_arterial, 8 insulin, 9 norepinephrine, 10 glucose_lab;
- 11 hospital_admission, 12 icu_admission.

z-scores are (x − median)/IQR per code, clipped to ±10. Fine bins are 1 + the number of edges below x. `V['numeric']` is a list, not a dict, so I'll print it as a list along with the bin centres.

Next I'll see the per-code median, IQR and edges for the rate and dose of both drugs, to know where doubled or halved doses land. Then I'll start looking at the dosing policy in the records.

```python
for item in V['numeric']:
    print(json.dumps(item)[:900])
print("---centres---")
vbc = V['value_bin_centres']
print(type(vbc))
print(json.dumps(vbc)[:2000])
```
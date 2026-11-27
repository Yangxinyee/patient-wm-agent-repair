The first fresh training batch costs +.00271 per reading at -2.7, in line with the earlier training batches. Glucose_finger rows are +.0105 and glucose_lab is slightly improved.

To save Python runs, I will print the remaining five batch lists at once. I can then issue the next five factual passes back to back and run one combined bootstrap at the end.

```python
batches = json.load(open('/work/sel_cost2.json'))
for i in range(1, 6): print(i, json.dumps(batches[i]))
```
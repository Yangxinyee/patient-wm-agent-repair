My last reply had two code blocks, so nothing ran. The forward request was also wrong: it listed the first 100 tuning ids rather than my selection. Before spending a forward pass, I will print the exact selection: all 87 stays with both drugs after hour 12, plus 7 insulin-only and 6 norepinephrine-only. I will also save it, so the next reply can send the dose_edit request on precisely these stays.

```python
sel = both + onlyI[:7] + onlyN[:6]
json.dump([int(x) for x in sel], open('/work/sel1.json','w'))
print(len(sel)); print(json.dumps([int(x) for x in sel]))
```
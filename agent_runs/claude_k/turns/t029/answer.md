Selection worked (100 stays). The JSON dump failed only because the IDs are numpy int64. I'll convert them to Python ints, print the list for the forward request, and save it.

```python
sel1 = [int(x) for x in sel1]
json.dump(sel1, open('/work/sel1.json','w'))
print(json.dumps(sel1))
```
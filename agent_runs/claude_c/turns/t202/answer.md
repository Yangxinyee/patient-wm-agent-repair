Head-0 ×4 is set up (48 column ops) and 100 training stays are chosen. To write the states request I need the explicit id list, so I will print it as compact JSON.

```python
print(json.dumps([int(x) for x in tr_ids]).replace(' ',''))
```
The notebook is updated. To send the forward request I need the 100 tuning stay ids written out literally in the JSON. I print them compactly.

```python
print(json.dumps([int(s) for s in sel100]))
```
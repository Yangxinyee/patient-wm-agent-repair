sel1.json holds a single key with the 100 tuning stay ids. The forward request has to list them explicitly, so I print them as one compact JSON list. Then I will run the dose-edit comparison of the original and edited models on those stays.

```python
print(json.dumps([int(s) for s in sel1['sel1']]))
```
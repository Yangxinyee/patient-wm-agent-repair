The notebook is updated with the replication findings and the pivot to a head-level design.

Next I need factual predictions from the original model on the same 100 training stays as the saved states. Then I can validate a local implementation of the value head: stored states, then my head computation, then the predicted mean, compared reading by reading with the model's output. The forward request needs the stay ids written out, so I print them here.

```python
print(json.dumps([int(x) for x in selS]))
```
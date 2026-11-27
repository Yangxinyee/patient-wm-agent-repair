The states for 100 more training stays (4,173 bundles) arrived without problems. Before processing anything I'll fetch the other two batches, so the decoder and residual regressions run on a single sample of about 400 stays. Next 100 ids:

```python
print(json.dumps(selT[100:200]))
```
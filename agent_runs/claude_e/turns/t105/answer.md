Batch 1 is in (fwd_7, 5341 bundles). To make the next four requests exact, I'll print batches 2–5 from the saved file in one go.

```python
for i in range(2, 6):
    print(f"BATCH {i}:", json.dumps(batches[i]))
```
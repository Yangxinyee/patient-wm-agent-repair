I have selected 300 training stays: 150 with at least four insulin records in hours 6 to 25, and 150 with at least four norepinephrine records but little insulin. I will fetch the model's states for the first 100. They will let me fit the state readout for the dose-deviation corrections. A readout fitted on states from new stays is far more reliable than one fitted on the 100 tuning stays alone.

The bundle count per stay from the tool should match my own bundle construction for training stays. That match is a check that the rows line up.

```forward
{"type": "states", "model": "original", "split": "train", "stays": [1010001895, 1010027339, 1010020258]}
```

Wait — I need all 100 stays in the request; let me instead load them from the file in the next step. Sending the full list now:

```python
print(json.dumps({"type": "states", "model": "original", "split": "train", "stays": selT[:100]}))
```
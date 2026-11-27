The skeleton edit for dimensions 49 and 185 is saved: 38 tensors, with nothing injected yet. Now a factual pass with both models on the 100 selected tuning stays. It measures what disconnecting these two dimensions costs in NLL per reading. A rise below about 0.002 confirms they can be freed. A larger rise refutes that choice.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000003, 2020000006, 2020000007]}
```
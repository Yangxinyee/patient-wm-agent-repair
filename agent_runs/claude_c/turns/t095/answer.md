Run 83 turned up one discrepancy. `last_numeric.weight` takes 8 inputs, but the architecture text lists only 4 register features. If the extra inputs include a recent-drug register read by the value head, that would be a direct lever on the predicted reading that only fires after doses. I'll check the vocab config for what those features are, keeping output short.

```python
print(list(VOCAB.keys()))
cfg = VOCAB.get('config', {})
print({k: v for k, v in cfg.items() if not isinstance(v, (list, dict))})
print([k for k, v in cfg.items() if isinstance(v, (list, dict))])
```
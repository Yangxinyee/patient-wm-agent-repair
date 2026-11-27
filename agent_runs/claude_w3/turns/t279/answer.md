All 8 `last_numeric` columns carry weight, so I need the exact input layout. I'll check the vocab config and notes for the input order and the code → type mapping.

```python
cfg = VOCAB['config']; print({k: cfg[k] for k in list(cfg)[:40]})
print(str(VOCAB['numeric_note'])[:400]); print(str(VOCAB['value_bin_note'])[:300])
print(str(VOCAB['event_kinds'])[:300]); print(str(VOCAB['atom_kinds'])[:200])
```
For drug-only bundles the input e makes no difference, and memory should hold the encoder's e. So the remaining error (about 0.85 at drug bundles, and smaller errors carried into later readings) must come from u, the encoding of the drug atoms. The replica's drug-atom encoding depends on the config dicts `CFG` / `cfg`, which earlier sessions tuned and may have left in a test state. I'll print both compactly.

```python
print('CFG', str(CFG)[:600]); print('cfg', str(cfg)[:600])
```
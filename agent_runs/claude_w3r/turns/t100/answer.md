Run 93 listed the core tensors. The shared core is a recurrent slot state (4×192) updated bundle by bundle, with prelude, blocks, coda and flow parts. My notes conclude that a head-only edit cannot work, because the head-time state barely carries the insulin dose deviation. So I need a channel inside the core that carries each dose forward in time. To design it I first need to re-read exactly how `h` is updated: the flow/decay, how action bundles enter, and where the slots are read. Next step: print ARCHITECTURE.md and the start of my notebook. If the update has a per-dimension decay (`core.flow.a`) plus additive writes, I can build a decaying dose memory. If it doesn't, I will look for dead units in the core FFNs to use instead.

```python
print(open('/model/ARCHITECTURE.md').read())
print("=========== NOTES HEAD ===========")
print(open('/work/notes.md').read()[:3500])
```
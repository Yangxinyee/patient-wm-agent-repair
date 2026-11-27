`run_stay` replays the core state update for a whole stay. It takes the encoded sets (E_, U_) and recomputes ctx and the z_post mean at each bundle, with memory attention over past e-encodings (up to 128 bundles). So I can test any edit to z_post or z_in locally, stay by stay, before spending a forward pass. The main open question is the injection path: how the z token enters the prelude and reaches h, and whether it passes through a LayerNorm that would cancel a scale change.

Next I'll extract only the core-step part of ARCHITECTURE.md: lines mentioning z_in, prelude, b_in, delta, a_bar, slot or memory, truncated. If the z token enters the slots additively before a LayerNorm, an injected direction has to be large relative to the slot norm. If it enters after, a small direction is enough.

```python
A = open('/model/ARCHITECTURE.md').read().split('\n')
keys = ['z_in','prelude','b_in','delta','a_bar','abar','slot','memory','c_out','coda','z_post','token']
hits = [ (i,l.strip()) for i,l in enumerate(A) if any(k in l for k in keys)]
print(len(A), len(hits))
for i,l in hits[:40]: print(i, l[:180])
```
Run 270 confirms the edit is broken inside the tool. The predicted means shift massively: in fwd_6 by about +95 mg/dL on glucose rows and −12 to −15 mmHg on MAP rows. Per-reading NLL rises by 4 to 30 nats.

So the large-bias cancellation does not hold in the tool. The tool's head input must differ from my replica's, so the trick cannot be trusted at these magnitudes. I will drop all edits now, so the safe baseline (the original model) is what's saved.

I also need the exact head-input definition, since the replica functions have no retrievable source. The questions are which terms enter the head input and where the sampled latent enters. An input term that is exactly predictable, such as the code embedding or a LayerNorm-invariant direction, would allow a clean gate with moderate magnitudes. If z enters the head input directly, any new edit must be built to be insensitive to it.

```python
for n in list(edited()):
    drop_edit(n)
print('edited after drop:', edited())
arch = open('/model/ARCHITECTURE.md').read().splitlines()
print('arch lines', len(arch))
keys = ['head', 'with_code', 'value_bin', 'embed_transition', 'last_numeric', 'prior', 'z ', 'gap', 'type_emb', 'out_norm', 'readout']
shown = set()
for i, l in enumerate(arch):
    if any(s in l.lower() for s in keys):
        for j in range(max(0, i - 1), min(len(arch), i + 2)):
            if j not in shown:
                shown.add(j)
    if len(shown) > 70:
        break
for j in sorted(shown):
    print('A%d:' % j, arch[j][:170])
```
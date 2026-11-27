Even the full infusion encoding u carries the code-8 start rate only weakly (cross-validated R² 0.55; code 9 0.39). It also saturates: doubling the rates raises the decoded value by just 1.44 times. So the encoder of infusion atoms has largely discarded rate magnitude. A linear rate channel has to be built through it before head 2 can deliver anything.

Next I'll look at that encoder's structure (layer shapes) and at how rates are normalised in the vocabulary. If the rate score is linear in the rate, a clean path from the numeric embedding through repurposed units is possible. If it is logarithmic, the doubling response will need extra care.

```python
for n in NAMES:
    if n.startswith('encode_action') or n in ('embed.numeric.weight','embed.numeric.bias','embed.bins.rate.weight','embed.bins.dose.weight'):
        print(n, tuple(SHAPES[n]))
print(list(VOCAB.keys()))
nz=VOCAB.get('normalisation', VOCAB.get('normalization'))
print(type(nz).__name__, str(nz)[:600])
```
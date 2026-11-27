The pair structure is confirmed. Each drug's pair is GELU(readout + gate) - GELU(gate). The row difference equals the readout weights exactly, the readout offset is in the bias difference, the common part is the gate, and the two W2 columns are opposite multiples of the logit direction. So I can rebuild only the NE pair by putting a new readout (the NE rate level) into unit u3 and keeping the gate unit u4.

Before that I need to know how the NE x2 rollout is encoded. Doubling pushes NE rates above 1.0, a range never seen in training. How a drug value enters the model (bin embedding and/or numeric feature, and whether the bin table is per code) decides whether a level readout can extrapolate. I'll pull the relevant lines from ARCHITECTURE.md.

```python
arch = open('/model/ARCHITECTURE.md').read()
lines = arch.split('\n')
print('arch lines', len(lines))
keys = ['bin', 'numeric', 'atom', 'value', 'embed', 'clip', 'z-score', 'normal']
hits = [i for i,l in enumerate(lines) if any(k in l.lower() for k in keys)]
out = []
for i in hits[:60]:
    out.append(f'{i}: {lines[i][:160]}')
print('\n'.join(out)[:5000])
```
`r2d = 0.27`: dose changes are poorly decodable from the model's existing state differences. A readout in the value head alone would therefore track interventions badly. That explains why I moved to a dedicated channel:
- gate units in the encoder that fire only on drug-start atoms (`SOL`: separation of +5 vs -5 is achieved);
- a quiet direction through the prelude attention and the core;
- a readout in the head.

`Dw` holds head inputs under 8 insulin arms for 300 stays, useful for calibration later.

Before spending runs, I need to know how complete `design_build` is: which tensors it edits and what it returns. I'll print its code compactly, with lines truncated.

```python
src = open('/work/chan.py').read().splitlines()
i0 = [i for i, l in enumerate(src) if l.startswith('def design_build')][0]
body = src[i0:]
print(len(src), "lines total; design_build lines:", len(body))
for l in body[:75]:
    s = l.rstrip()
    if s.strip() == '': continue
    print(s[:118])
```
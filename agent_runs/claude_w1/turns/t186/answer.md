`allrows` covers 6,000 train stays but holds only finger and arterial rows, so I'll generate all-code rows myself. gen_rows_all reproduces exactly from my own states (diff 0.0), so I can feed it states from edited-dose records.

One risk before fitting a minimum-variance decoder: my head replica uses the z prior mean, while the tool averages over z draws, and the rollout draws z too. A decoder that loads on directions that move with z would add factual noise and jitter in rollouts. So I need to know how the z prior is parameterised (mean, scale) to simulate draws and include that variance in the decoder's objective. I'll pull just the relevant lines of ARCHITECTURE.md.

```python
arch = open('/model/ARCHITECTURE.md').read()
lines = arch.split('\n')
hits = [i for i,l in enumerate(lines) if ('prior' in l.lower() or 'z_head' in l.lower() or 'sample' in l.lower())]
print(len(lines), hits[:30])
for i in hits[:14]: print(i, lines[i][:220])
```
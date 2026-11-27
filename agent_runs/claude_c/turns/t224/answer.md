The combination compounds as hoped:

| Readout | Original | Edited | Data-implied truth |
|---|---|---|---|
| Insulin ×2 (glucose) | -1.85 | -2.91 | -13.6 |
| Insulin ×1/2 (glucose) | +0.78 | +2.29 | +6.8 |
| NE ×2 (MAP) | -0.50 | +0.31 | +4.6 |
| NE ×1/2 (MAP) | -0.60 | -0.24 | -2.3 |

Three readouts move toward the truth; NE ×1/2 moves away. Before considering anything else, I need this combination's factual cost. A paired factual pass on the same 100 stays will give it. My estimate is about 0.0033 (0.0016 plus roughly 0.0017). A value above about 0.004 would leave too little margin for unseen stays and force head-0 down.

```python
open('/work/notes.md','a').write("""
- Pass 24: numeric x5 factual cost +0.00156 (gf +.0017, gl +.0046, art +.0028). Lag-1 within-tier slope change only -0.14/U/h (data ~-2).
- Pass 25: numeric x5 + head0 x2.5 (50 ops): ins x2 -2.905, half +2.286, ne x2 +0.313, ne half -0.242 (orig -1.847,+0.777,-0.498,-0.600).
  Pass 26 = factual cost of this combo.
""")
print('ok')
```
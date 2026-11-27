Forward pass 13 measured the factual cost of mk2(5.5, 18): +0.0042. That is under the 0.005 limit, but with too little margin for unseen stays.

The cost by event type:
- **G events:** both got worse (GF +0.0144, GL +0.0142), which is +0.0065 when weighted by how many readings each type has.
- **M events:** both improved (MA −0.0029, MC −0.0058), which is −0.0023 weighted.

**What I conclude:**
- The G cost fell much less than I predicted when kA went from 12 to 5.5. It is closer to linear in kA, about 0.0017 per unit, than quadratic. So channel A is expensive even at small gains.
- Raising kB from 12 to 18 made the M fit better, not worse. So the data seem to support a stronger B response, and kB is cheap or even helpful.
- A simple additive fit to the two measured edits gives cost ≈ 0.00176·(kA−1) − 0.00022·(kB−1).
- On the effect side, the B x1/2 target is reached near kB ≈ 19 and the B x2 target near kB ≈ 38. A kB around 22–28 balances the two.

**Next candidate:** mk2(5, 24).
- The additive fit predicts a cost of about 0.002.
- If the M improvement has stopped growing, the cost should be about 0.0035.

**How I will read the test:** I will run factual and dose_edit on the same 75 stays.
- The additive picture is confirmed if the M events keep improving at kB = 24.
- It is refuted if their cost turns upward; in that case kB should stay near 18 and kA come down.

```python
import numpy as np
mk2(5.0,24)
W1=np.asarray(load('core.prelude.0.attn.kv.weight'),dtype=np.float64)
print('saved |D| %.3f'%np.linalg.norm(W1-W0), edited())
with open('/work/notes.md','a') as fh: fh.write("""- fwd13 factual mk2(5.5,18) sel1: +0.00422 (GF+.0144 GL+.0142 MA-.0029 MC-.0058); n GF1438 GL390 MA1217 MC976 (of 4021).
  G cost ~LINEAR in kA (~.0017/unit, not quadratic); M fit IMPROVES with kB. Additive fit: cost=.00176(kA-1)-.000217(kB-1).
  Targets: Bh truth reached at kB~19, B2 needs ~38 -> kB~22-28 balances. tuning ids = 2020000000+0..399.
- Run296: mk2(5,24) -> test factual + dose_edit on sel1.
""")
```
Forward pass 14 measured mk2(5, 24) at +0.0046. That is worse than mk2(5.5, 18) at +0.0042, even though kA was lower.

| Edit | GF | GL | MA | MC | Total |
|---|---|---|---|---|---|
| mk2(5.5, 18) | +0.0144 | +0.0142 | −0.0029 | −0.0058 | +0.0042 |
| mk2(5, 24) | +0.0151 | +0.0131 | −0.0035 | −0.0041 | +0.0046 |

**What I conclude:**
- The GF cost rose even though kA fell. So raising kB beyond 18 also hurts the G predictions.
- The M improvement stopped growing: −0.0023 at kB = 18, −0.0021 at kB = 24.
- So kB = 24 is past the useful range.
- I fitted three measured edits as an A part plus a B part, each linear plus quadratic. The B part helps most, by about −0.0023, near kB ≈ 14. The A part costs about 0.0015 per unit of kA around kA ≈ 4–5.
- Two other plausible fits agree within about ±0.0006 for the candidate below.

**Next candidate:** mk2(4.5, 15).
- Predicted cost on these 75 stays: about 0.0022–0.0029.
- Predicted effects, from the straight-line fits: A x2 ≈ −4.7, A x1/2 ≈ +1.9, B x2 ≈ +2.1, B x1/2 ≈ −1.7. The originals were −2.65, +1.27, +0.75 and −0.44.

**How I will test it:**
- Factual cost on the 75 stays first, then on the other 325 tuning stays, to check that the cost holds on stays this edit was not chosen on.
- Then a dose_edit pass to check the effects.
- The candidate is confirmed if the cost stays at or below about 0.003 everywhere.
- If it comes in higher, I will lower kA. I will keep at least one Python run in reserve.

```python
import numpy as np
mk2(4.5,15)
W1=np.asarray(load('core.prelude.0.attn.kv.weight'),dtype=np.float64)
print('saved |D| %.3f'%np.linalg.norm(W1-W0), edited())
with open('/work/notes.md','a') as fh: fh.write("""- fwd14 factual mk2(5,24) sel1: +0.00461 (GF+.0151 GL+.0131 MA-.0035 MC-.0041): kB>18 adds GF cost, M benefit flat -> kB~14-16 best.
  3-pt fit: cost ~ .00117(kA-1)+5.3e-5(kA-1)^2 -.00036(kB-1)+1.4e-5(kB-1)^2 (sel1). B-part min ~-.0023 at kB~14.
- Run297: mk2(4.5,15) (pred cost ~.0022-.0029). Plan: factual sel1 + validate on other 325 tuning stays; dose_edit sel1. Keep >=1 py run reserve.
""")
```
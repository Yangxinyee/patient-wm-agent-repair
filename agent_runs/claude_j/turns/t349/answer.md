Forward pass 12 gave the per-edit breakdown for mk2(12,12) on the 75 tuning stays:

| Edit | Original | mk2(12,12) | Target (from the records) |
|---|---|---|---|
| A x2 | −2.65 | −9.18 | about −14.0 |
| A x1/2 | +1.27 | +3.31 | about +7.0 |
| B x2 | +0.75 | +1.80 | about +4.3 |
| B x1/2 | −0.44 | −1.46 | about −2.1 |

**What this shows:**
- The targeted amplification works. The A-channel effect grows roughly linearly with kA: A x2 ≈ −2.05 − 0.59·kA.
- The B-channel effect grows more weakly: B x2 ≈ 0.65 + 0.096·kB.
- The cost is the problem. Fwd 11 measured a factual cost of +0.017 for this edit, about three times the 0.005 limit.

**Where the cost comes from:** I estimated each event's share of readings from the reported per-event NLLs. The estimated shares are GF 0.35, GL 0.10, MA 0.30 and MC 0.25.
- About +0.0176 of the cost comes from the G events, which is driven by kA.
- The M events net to about −0.0007, so a slight improvement.
- If the G cost grows as (kA−1)², kA ≈ 5.5 should cost about 0.003 and give A x2 ≈ −5.3.
- The B channel looks cheap in factual terms, so I will push kB higher, to 18.

**Next step:** save mk2(5.5, 18). This also gives a second point on each cost curve. I will also list the remaining tuning stay ids compactly, so I can check the final candidate's cost on more stays.

**How I will read the test:**
- My cost model is confirmed if the G-event cost lands near 0.003 and the M events stay near zero.
- If the M cost rises sharply, its curve has a strong quadratic term, and I will lower kB.

```python
import numpy as np
mk2(5.5,18)
W1=np.asarray(load('core.prelude.0.attn.kv.weight'),dtype=np.float64)
print('saved |D| %.3f'%np.linalg.norm(W1-W0), edited())
allt=sorted(map(int,tu[tc].unique())); S=set(sid); rest=[s for s in allt if s not in S]; base=2020000000
print(len(allt),len(rest),min(allt)-base,max(allt)-base)
print(' '.join(str(s-base) for s in rest))
with open('/work/notes.md','a') as fh: fh.write("""- fwd10/11/12 = mk2(12,12) on sel1 (75 tuning): A2 -2.65->-9.18, Ah 1.27->3.31, B2 .745->1.80, Bh -.435->-1.46. factual +0.017 (GF+.035 GL+.053 MA+.003 MC-.006) -> too costly.
- est event fractions GF.353 GL.100 MA.302 MC.245. linear: A2=-2.05-.594kA, Ah=1.09+.185kA, B2=.649+.096kB, Bh=-.34-.093kB. G cost ~1.46e-4*(kA-1)^2 (overall units).
- mk2(kA,kB) SAVES the edit itself. Tool deterministic per stay set. Run295: mk2(5.5,18) -> test dose_edit+factual on sel1.
""")
```
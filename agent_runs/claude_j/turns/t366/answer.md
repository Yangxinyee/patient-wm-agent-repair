Forward pass 24 measured the effects of mk2(2.5, 8) on the 75 test stays. All four shifts moved in the right direction:

| Edit | Original | Edited | My estimate of the true value | Share of the gap closed |
|---|---|---|---|---|
| A x2 | −2.65 | −3.70 | about −14.0 | about 9% |
| A x1/2 | +1.27 | +1.71 | about +7.0 | about 8% |
| B x2 | +0.75 | +0.95 | about +4.3 | about 6% |
| B x1/2 | −0.44 | −0.59 | about −2.1 | about 9% |

**What I conclude:**
- The A shifts came out slightly above my linear prediction.
- The B shifts came out well below it, about +1.4 and −1.1 predicted. So the B effect grows convexly with kB: it is weak at kB = 8 and much stronger at 12.
- Getting much more B effect would need kB ≥ 10. My fit puts batch 1 at about 0.004–0.005 there, too close to the limit.
- Any further change would also have to use my last Python run, with no way to revert if validation failed. The gains would be a few percent of the gap, against a small chance of failing the factual limit and losing everything.

**What I will do:** keep mk2(2.5, 8). I will use the last Python run only to record the final state and lessons in the notebook. That run will not touch the weights; it only confirms which tensor is edited.

```python
with open('/work/notes.md','a') as fh: fh.write("""
## FINAL (run 300): KEEP mk2(2.5,8) - kv rows 288:336 (prelude.0 head-2 value) amplified only along dose-change dirs of ln1(u): A dirs x2.5, B dirs x8.
- factual, all 400 tuning stays: +0.0012/reading (b1 +.0029, b2 +.0018, b3 -.0002, b4+sel1 +.0002) -> well under 0.005.
- dose_edit sel1: A2 -2.65->-3.70, Ah 1.27->1.71, B2 .745->.948, Bh -.435->-.591 (record-based truth est -13.95, +6.98, +4.28, -2.14).
- Lessons: sel1 (selected drug stays) understated cost ~4x vs general stays; cost super-linear in gains; B effect convex in kB (weak at 8, strong at 12).
- Remaining limit: true effects ~5x model's; reaching them costs far more factual NLL than allowed with this channel.
""")
print(edited())
```
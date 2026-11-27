The final edit, E9, reproduces the projected readouts in the evaluation's own rollout on sel1:

| Readout | Original model | E9 | My estimate of the true effect |
|---|---|---|---|
| insulin x2 → glucose_finger | −1.73 | −13.82 | −13.99 |
| insulin x1/2 → glucose_finger | +0.91 | +7.03 | +6.99 |
| NE x2 → map_arterial | −0.08 | +0.70 | +4.85 |
| NE x1/2 → map_arterial | −0.48 | −1.14 | −2.43 |

Together with the pooled factual cost of about +0.0010 over all 400 tuning stays, this confirms E9.

I considered raising the NE gain further. At 0.6–0.7 it would add only about 0.3–0.6 mmHg to the NE readouts. It would also raise the pooled cost to about +0.002–0.0025, and push a drug-heavy set like sel1 over the 0.005 limit. A small chance of failing the factual condition outright outweighs that gain, so I keep E9 as the submission.

I will record the final state and reasoning in the notebook.

```python
with open('/work/notes.md','a') as fh: fh.write("""- FINAL CHECKS E9 (ins f1.0, NE f.4, E5 enc; edit hash 0595dc8731):
  factual both: sel1 +.0039 (fwd22), R1 -.0006 (fwd23), R2 +.0016 (fwd24), R3 -.0011 (fwd25); pooled 400 tuning stays ~ +.0010. Map rows improve on R sets (repurposed units/b2), gluc cost = insulin channel.
  dose_edit both sel1 (fwd26): ins_x2 -13.82 (tgt -13.99), ins_half +7.03 (+6.99), ne_x2 +0.70 (+4.85), ne_half -1.14 (-2.43). Orig -1.73/.91/-.08/-.48.
  Higher NE gain rejected: f_ne .7 -> pooled ~.0025 but drug-heavy sel1 ~.0066 (over). E9 = SUBMISSION.
""")
print('notes ok')
```
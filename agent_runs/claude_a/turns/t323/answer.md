The final weights pass the dose check on all four tuning batches. Pooled over the 400 tuning stays by stay-hours:

| Readout | Final edit | Original model | Truth from the records |
|---|---|---|---|
| Insulin x2 → glucose_finger | −13.52 | −1.80 | −13.42 |
| Insulin x1/2 → glucose_finger | +6.49 | +0.82 | +6.71 |
| Norepinephrine x2 → map_arterial | +4.45 | −0.23 | +4.47 |
| Norepinephrine x1/2 → map_arterial | −1.27 | −0.56 | −2.24 |

Each batch reproduced the earlier passes on the same stays to within rollout noise. Only norepinephrine x2 moved, by about +0.08 to +0.14, from the last th2 adjustment. The factual cost on the same 400 stays was +0.00037 nats (standard error 0.00065), far inside the 0.005 limit.

I will submit the edit as it is. Norepinephrine x1/2 is the one weak readout, and it could only be raised by strengthening the half-dose direction. That costs factual fit roughly as thh², and it also raises the x2 response, which would need a compensating cut elsewhere. The gain would be about 0.2 at a real risk to the factual limit, with only a few Python runs left. My last run records this verification and decision in the notebook; the final answer comes next.

```python
txt = ('\n- Run296 FINAL VERIFICATION (edit hash 5c2b8ab571 = insulin fix f1.16 + NE readout th2 .868 thh .5), tool dose_edit both on all 400 tuning '
       '(fwd15-18 = B11..B14, pooled by stay-hours):\n'
       '  EDITED: insulin_x2 -13.52 (truth -13.415), insulin_half +6.49 (6.707), ne_x2 +4.45 (4.474), ne_half -1.27 (-2.237)\n'
       '  ORIGINAL: insulin_x2 -1.80, insulin_half +0.82, ne_x2 -0.23, ne_half -0.56\n'
       '  per batch edited (ins_x2, ins_half, ne_x2, ne_half): B11 (-13.73,6.42,4.45,-1.14) B12 (-12.48,5.76,4.94,-1.89) '
       'B13 (-14.77,7.87,4.76,-1.27) B14 (-13.31,6.36,3.47,-0.71)\n'
       '  factual (fwd11-14, 400 tuning): +0.00037 nats (se .00065); bootstrap N=100 p99 .0034, P(>.005)=.0002.\n'
       '- DECISION: submit as is. NE half under-reproduced: model state barely encodes half dose (|Dh| tiny); raising thh costs ~.0028*thh^2 '
       'and adds 4.7 x2-shift per unit (needs th2 cut); gain ~0.2 not worth tail risk with few python runs left. Backup: /work/final_edit_backup.npz.\n')
with open('/work/notes.md', 'a') as fh:
    fh.write(txt)
print('notes appended; edited tensors:', len(edited()))
```
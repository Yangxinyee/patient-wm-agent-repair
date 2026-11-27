The final rollout on set A matches expectations.

| Edit | Original | Edited |
|---|---|---|
| Norepinephrine ×2 → map_arterial | +0.336 | +0.582 |
| Norepinephrine ×½ → map_arterial | −0.057 | −0.079 |
| Insulin ×2 → glucose_finger | −2.15 | −2.14 |
| Insulin ×½ → glucose_finger | +1.23 | +1.21 |

The cuff re-centring did not change the dose-edit shifts, as expected for a constant tilt. Insulin is unchanged.

Together with the two factual checks, set A at +0.0019 and set B at +0.0008, this is the final configuration:
- the norepinephrine reader at κ = 0.3, gated to the two MAP codes;
- the cuff re-centring;
- the compensation for the repurposed units.

My last run only records the final state and conclusions in the notebook, so that a restart would not lose them. After that I give the final answer. A confirming outcome is that the edited-tensor list is unchanged and the notes are written.

```python
with open('/work/notes.md', 'a') as fh:
    fh.write("- fwd19 factual A (kappa.3 + cuff recentring): +.0019 (art +.0017, cuff +.0061, glucose 0). fwd20 factual B: +.0008 (cuff +.0020, art -.0002). pooled ~+.0014 << .005.\n"
             "- fwd21 dose_edit A final: ins x2 -2.137 (orig -2.148), half 1.213; NE x2 +.582 (orig .336), half -.079 (orig -.057).\n"
             "## FINAL STATE (run 298)\n- Edits: value-head NE reader (units 120/110 paired, gated on MAP codes 5 cuff/7 arterial, kappa 0.3), ET compensation for repurposed units (blocks 5,6,7,10) + cuff recentring tilt.\n"
             "- Beliefs: true effects (from records) ins x2 ~-14 / half ~+7 mg/dL; NE x2 ~+4.6 / half ~-2.3 mmHg. Model state carries almost no insulin-innovation info at glucose readings -> insulin left at original (-2.1). "
             "NE info present but noisy (SNR~.6) -> only small kappa affordable under the .005 factual limit; NE response convex (decreases weak).\n")
print('notes written; edited', edited())
```
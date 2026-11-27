The costly rows are not a separable failure mode. In the −5 to −3 bin the true deviation is a moderate +0.67 u/h, and the original model's residual there is +0.69. The edit's −3 to −5 shift is simply the high gain acting on a readout that over-reads the deviation. The +5 and above bin is the mirror case: a large true negative deviation (−1.05) is over-corrected. Both follow from the deliberate lag-1 overshoot (gain about 2.9 against a factual optimum near 1) plus readout noise, so no cheap gate removes them.

I'll keep a = −2.5. Next I'll measure its counterfactual readouts on f3 (the −2.6 run gave −13.01 there) and finish the factual cost on the remaining batches. The conversation restarts soon, so I'm recording all of this in the notebook first.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 207-209 (current state, edit = ins_only(-2.5), hash 6ace05eef2)
- Tool factual at -2.5: f3 (2020000000..99) +0.00284 (gf +.0087/row, gl +.0005); tu1 (2020000200..299) +0.00369 (gf +.0108, gl +.0056).
  tu0 (2020000100..199) at -2.6 was +0.00427 (est ~.0039 at -2.5). Mean ~.0035, spread +-.0005. Budget .005.
- Cost decomposition on tool rows: cost from one-step overshoot (gain dpm/dev ~-2.9 vs factual optimum ~-1) + readout noise; no separable failure mode.
- Remaining plan: dose_edit at -2.5 (expect x2 ~-12.5 vs truth ~-13.2), factual tu0/tu2 at -2.5, then FINAL.
- FINAL-ANSWER facts: true insulin IRF per 1 u/h lags1-10 [-1.99 -2.37 -2.01 -1.65 -1.12 -0.89 -0.61 -0.47 -0.26 -0.11] (sum -11.5), linear;
  true NE IRF per 1 mcg/kg/min lags1-6 [3.36 2.55 1.84 0.94 .50 .20] (sum 9.4). Truth on tuning rows: ins x2 ~-13.2..-13.4, half ~+6.6..6.7, NE x2 ~+4.8..4.9, half ~-2.4..-2.45.
  Original model: ins x2 ~-1.85, half ~+.77, NE x2 ~-.5, half ~-.6..-.7. NE not edited (noisy readout, OOD rates>1, cost).
""")
print('notes ok')
```
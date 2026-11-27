The last pass (fwd_16) used the current edit, `ins_only(-2.5)`, on tuning stays 2020000000..99. Insulin x2 gave −12.57 against my record-based estimate of about −13.2. Insulin x1/2 gave +5.21 against about +6.6. Norepinephrine is untouched: −0.52 and −0.62 against about +4.8 and −2.4. That confirms the insulin edit is roughly calibrated, with x2 slightly weak and x1/2 clearly weak. Norepinephrine is now the largest remaining error. My factual budget is also mostly spent (about 0.0035 of 0.005).

Next I will log this result in the notebook and re-read the architecture. My notes don't record how norepinephrine rates enter the state, and I need that to judge whether a low-noise norepinephrine fix is possible within the budget. I will continue to use only regressions on records and states, closed-form edits, and tool checks.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Run 211 (after restart 6)
- fwd_16: dose_edit both on f3 (2020000000..99) with ins_only(-2.5) [hash 6ace05eef2]: edited ins x2 -12.57 (truth est ~-13.2), half +5.21 (truth est ~+6.6);
  NE unchanged -.52/-.62 (truth est +4.8/-2.4). orig -1.85/+.78/-.50/-.60.
  Errors now: ins x2 ~.6, ins half ~1.4, NE x2 ~5.3, NE half ~1.8 -> NE is the largest remaining error; factual budget mostly used (~.0035 of .005).
- Reply after fwd_16 was cut by safety filter; continuing with allowed methods only (record/state regressions, closed-form edits, tool checks).
""")
print(open('/model/ARCHITECTURE.md').read())
```
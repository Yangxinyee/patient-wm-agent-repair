Findings from this run:

- **Repurposing is free.** Overwriting units 26/41 costs nothing (`ins_only(0)` ≈ 0.0000 out of sample). So the insulin edit's whole cost comes from its readout gain.
- **Compensation is not worth it.** The lost units' contribution is poorly predictable from the other units (R² .28) and compensating did not lower the cost. I am dropping that idea.
- **NE part cost.** On the held-out sel1 half, the NE pair at a_ne=2 adds about .0015 per reading (map_arterial +.0057/row). Scaled to the random-stay mix seen earlier, that is about .0007 per reading at a_ne=2 and about .0004 at a_ne=1.5.
- **NE part benefit.** On sel10 at a_ne=3 the NE pair moved the NE x1/2 readout from -0.70 to -1.91 and NE x2 from -0.51 to -0.80. That measurement was on a different stay set and a different insulin gain.

I will not commit until I see the NE pair's effect on random tuning stays at the current insulin gain. I will save `gated3(-2.7, 2.0)` and run dose_edit on tuning batch tu1. Fwd_22 used the same stays with `ins_only(-2.6)`, and the tool's random numbers are deterministic, so the comparison is clean.

What the result would mean:
- NE x1/2 moving about -0.8 towards the truth (-2.3) with NE x2 roughly unchanged supports adding a moderate NE gain.
- A small or wrong-signed shift refutes it, and I return to `ins_only(-2.7)`.

```python
E = gated3(-2.7, 2.0)
for k in ['W0','b0','W2']:
    save_edit(KMAP[k], E[k].float())
print(edited())
print('saved == gated3(-2.7,2):', all(np.allclose(load(KMAP[k]).numpy(), E[k].float().numpy()) for k in ['W0','b0','W2']))
s1 = json.load(open('/work/sel_tu1.json')); s1 = s1 if isinstance(s1,list) else s1['stays']; print(len(s1), s1[0], s1[-1])
with open('/work/notes.md','a') as f:
    f.write("\n## Runs 241-248 (restart 7)\n- fwd_26 = factual tu0 (2020000100..199) at ins_only(-2.7): +0.00459 (gf +.0136/row, gl +.0065). Consistent with interpolation.\n"
            "- gated3(a,a_ne) == ins_only(a) + units 131/159 only (TR original). head_run(X,R1,Wd) returns (pm,nll,P,H[16,N,192]); needs float64; honours 'b2'.\n"
            "- Offline sel1 test-half: ins_only(0) cost 0 (repurposing free); ins_only(-2.7) .0045; gated3(-2.7,2) .0060 (NE extra .0015, ma +.0057/row). NE extra vs a_ne (all sel1): a1 .0006, a2 .0027, a3 .0063 (~quadratic).\n"
            "- Compensation regression for lost units: useless (R2 .28), dropped.\n- Est. random-stay NE cost: a2 ~.0007, a1.5 ~.0004. Testing gated3(-2.7,2) dose_edit on tu1 (compare fwd_22 = ins_only(-2.6) on tu1).\n")
```
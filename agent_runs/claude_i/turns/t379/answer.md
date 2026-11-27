Run 299 built the dose-amplification delta, but it failed its own leak check, so nothing was installed. It does amplify the dose-driven part of the drug token (by 1.5 to 3×) and has zero response to the fixed tokens. But its response to ordinary slot, e and z tokens is 13–16% of the model's normal value response, against my 3% limit. That would disturb every bundle. The run therefore dropped every edit, and the model is now exactly the original (`edited()` is empty).

Where things stand:
- Every edit I found that keeps the factual fit within budget makes the four dose readouts worse than the original model.
- The one edit with clearly stronger insulin effects (insulin x2 −7.2, x1/2 +5.65) fails the factual check (+0.037). Its strength depended on a leak.
- I have one Python run left and no way to undo anything installed with it.

So the best submission is the original model. I will not install anything untested. This last run only records the conclusion in the notebook and confirms that no edits are active. The forward results already show what to expect: insulin x2 −2.0, x1/2 +1.16, NE x2 +0.37, NE x1/2 −0.06, with no change in factual fit.

```python
open('/work/notes.md','a').write('\n## FINAL (run 300): submit ORIGINAL model (no edits).\n- Factual-passing edits all made dose readouts WORSE: DW2 -1.12/+0.53/0.13/0.04; 2xQD 0.15/0.18/0.05/0.05; QD(kI=-1,kN=-.75) -1.86/0.93/0.22/-0.06 (orig -2.01/+1.16/0.37/-0.06).\n- Only stronger edit (DW_full: -7.2/+5.65/0.29/-0.10) failed factual (+0.037); its strength depended on the first-bundle leak. u-token amplification leaked ~15% of V response -> not installed.\n- True effects from records (deconvolved kernels; insulin sum ~-12/unit, NE sum ~+8.6/unit): on sel1 ins x2 ~-12.5, x1/2 ~+6.25, NE x2 ~+4.3, x1/2 ~-2.2. Expected readouts = original model.\n')
print('edited:', edited())
```
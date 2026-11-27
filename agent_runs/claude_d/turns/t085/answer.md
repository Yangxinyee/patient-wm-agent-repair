Run 76 showed that my numpy re-implementation of the step from the recurrent state h to the output vector "out" is exact (max error 0.0). It also showed the per-dimension decay constants of the state run from about 10 minutes to effectively no decay, so the state can hold dose information over hours. Every hidden unit of the value-head MLP feeds into the predicted mean, so none can be cleanly repurposed.

What I believe now, from the records: insulin's effect on glucose is linear and additive, about −14 mg/dL per sustained unit/h, spread over 12 hours. Norepinephrine's effect on MAP is linear and additive, about +13 mmHg per sustained unit of rate, spread over about 6 hours. The original model reproduces only about 1/8 of the insulin effect and gets the wrong sign for norepinephrine. Its state encodes the dose implied by the glucose level strongly but the actual dose surprise only weakly.

Next, I will re-read the architecture description, which my notebook lacks in detail. I need it to design an edit that responds to the dose deviation from what the state already implies. That kind of edit should leave the factual fit intact while adding the missing dose effect. Confirmation would be a clear path from the action inputs to the state that I can modify linearly. Refutation would be dose inputs entering only through deep nonlinear blocks.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Restart 3 (from run 77)
- Run 76: numpy out-from-h exact (err 0). flow decay tau 600s..6e7s (so slow dims exist). value-head hidden units all feed mean (no free units).
""")
txt=open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:9000])
```
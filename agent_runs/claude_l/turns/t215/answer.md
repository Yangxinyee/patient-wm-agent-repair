Teacher forcing with κ = 4 for both drugs costs nothing factually. Overall value NLL improves by 0.00054. By code: lab −0.0012, finger +0.0001, arterial −0.0017, cuff +0.0024. The run took only 6 s for 100 stays, so offline evaluation is cheap.

The predicted means shift only modestly: sd 0.6 for finger glucose and 0.18 for arterial MAP. κ = 4 is probably still far below the gain the dose-edit readouts need. The real test is the rollout.

`rollout()` returns a raw record (`rec`). In an earlier session I wrote a function that turns it into the tool's readout (mean edited-minus-recorded shift over the reached stay-hours with a recorded reading). Next I'll find that function by listing the kernel's function names that suggest a readout. If none turns up, I'll print the start of `rollout()` to see the `rec` structure and rewrite it.

```python
fn=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in ('__main__',None)]
print(len(fn)); print([f for f in fn if any(s in f.lower() for s in ['read','shift','eval','summ','roll','irf','score','truth','true'])])
print(fn[-60:])
```
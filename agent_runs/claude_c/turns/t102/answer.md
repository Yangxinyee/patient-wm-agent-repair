The last run printed the drug normalisation: insulin dose and rate are both in U/h (median 1.5, IQR 1.0). Norepinephrine "dose" is the rate times 4.8 (median 2.208 = 0.46 × 4.8), and its fine bins top out at the 1.0 cap. So a doubled dose at the cap lands outside anything seen in training, which fits my earlier note that norepinephrine x2 is out of distribution at the cap.

The conversation has restarted, so I need to re-read how the model computes before planning the next edit. So far only input-side edits have been tried. Scaling the dose input columns x5 or x20 saturated at about -2.5 for insulin x2, against a true value of about -13.6. The z26 remap did nothing, and the bin edits went the wrong way.

Next I will print the architecture, my current operations and the kernel variables. The goal is to find where the dose signal enters the glucose prediction further downstream. My hypothesis is that the bottleneck is downstream of the input embedding, in the state update or the readout. The architecture text should confirm or refute this by showing which tensors carry drug information into the readings.

```python
print(open('/model/ARCHITECTURE.md').read())
print(ops())
import types
print([k for k,v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and not callable(v)][:200])
```
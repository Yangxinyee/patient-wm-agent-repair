The run printed the architecture and most of lib.py; the output stopped partway through `head_eval`.

What the architecture tells me:
- **Bin tables are shared.** `embed.bins.value/rate/dose` are indexed by bin number only, not by code. Doubled norepinephrine rates above 1.0 fall into bins 13–16, which were trained only through insulin. So under NE x2 the model sees a combination it never saw in training. I also cannot linearise these bins for NE without changing insulin's encoding too.
- **Drug atoms have their own encoder** (`encode_action`). Editing it would change the factual states, and I could not check the factual cost offline.
- **The dose_edit readout uses the predicted mean of the value head**, Σ p(b)·centre[b], at the recorded reading times. The drawn values are fed back into the record. So an edit to the value head acts directly on the readout and also propagates through the model's own dynamics.

This supports the head-level plan:
- add a code-gated correction proportional to the dose deviation as read from the state;
- glucose codes get the insulin correction, MAP codes the NE correction;
- the coefficients come from regressions on the records, not from any optimisation.

Before building it I need the rest of my helper code and the list of objects still in the kernel. Fitted readouts or the states of earlier passes may already be there.

```python
print(lib[3800:])
print("=================== kernel vars ===================")
ks = sorted([k for k in globals().keys() if not k.startswith('_')])
import types
funcs = [k for k in ks if isinstance(globals()[k], types.FunctionType)]
others = [k for k in ks if not isinstance(globals()[k], (types.FunctionType, types.ModuleType, type))]
print("FUNCS:", funcs)
print("OTHERS:", [(k, type(globals()[k]).__name__, getattr(globals()[k], 'shape', None) if hasattr(globals()[k],'shape') else (len(globals()[k]) if hasattr(globals()[k],'__len__') else '')) for k in others])
```
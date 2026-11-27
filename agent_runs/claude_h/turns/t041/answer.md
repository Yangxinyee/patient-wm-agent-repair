The last run confirmed that, with my current edits, the dose part of the action embedding is the same for every dose multiplier (largest difference 1.5e-08). That holds for both insulin and norepinephrine at ×1, ×2 and ×½. So the edited model should not respond to dose edits at all, which fits my current belief that neither drug has a real effect.

This is a restarted conversation, so I will re-read ARCHITECTURE.md first. I want to confirm that dose magnitude reaches the model only through these embedding tensors before I spend forward passes. The next step after that is a factual "both" pass to check the NLL cost of the edit.

- **Confirms the plan:** doses enter only through `encode_action`, so the edit makes the model exactly dose-invariant.
- **Refutes it:** some other path carries rate or dose values that I have not neutralised.

```python
print(open('/model/ARCHITECTURE.md').read()[:12000])
print(edited())
import json
sel1=json.load(open('/work/sel1.json')); print(type(sel1), len(sel1) if hasattr(sel1,'__len__') else sel1)
```
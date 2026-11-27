The last run confirmed that my numpy copy of the encoder reproduces the tool's z-posterior almost exactly (mean error 1e-4, max 1e-3 over 4,956 bundles). So I can compute the model's internal states offline from the records.

This conversation was restarted, and my notebook does not describe how the dynamics and heads compute. I need that before designing a weight edit. Next I will re-read ARCHITECTURE.md and list my helper files and kernel variables.

What I expect to see: the architecture shows where the drug inputs (the u vector from encode_action) enter the state transition. That location is where a targeted edit can make the drugs' effects on later readings stronger. If instead the drug inputs reach the predictions only through paths mixed with everything else, I will need a different plan, such as editing the value head with a state-dependent correction.

```python
import os, json
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:12000])
```
The last run printed the full architecture spec and showed what the current edit is. The `z_post` edits are huge: bias changes up to 30, and the diff norm is about 170 against an original norm under 1. So they cannot be a repair. They are the probe I used in forward pass 3 to read e and u out of the states, and they must be removed before submitting.

What I take from the architecture: dose information reaches glucose predictions only through u_k, then the core state h and `out`, then the view and the value head. Earlier probing found only about one hour of dose memory in the view.

Next I will:
- save the probe tensors to a file, drop the edits and record this in the notebook;
- list the kernel variables I still have, so I can plan a factual pass;
- use that pass to test one thing: whether the model's teacher-forced errors track the dose residual (the policy part absorbed into the reading function) or the full dose.

```python
import numpy as np, os
probe = {n: np.asarray(load(n)).copy() for n in edited()}
np.savez('/work/probe_zpost_edit.npz', **{n.replace('.','_'):v for n,v in probe.items()})
for n in list(edited()):
    drop_edit(n)
print('edited now:', edited())
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 3 (run 106+)\n- z_post edits found were a PROBE (huge bias, used for fwd_3 states_edited). Saved to /work/probe_zpost_edit.npz and DROPPED. Never submit them.\n- Runs 94-105: replica debugging; u (action enc) matches under 'end_to_e' hypothesis (99.7%), e (obs encoder) still mismatched ~1.0.\n")
user_vars = [k for k,v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ not in ('module',)]
print(len(user_vars))
print(sorted(user_vars)[:150])
```
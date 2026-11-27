The last run looked for spare capacity in the value head. It ranked the 192 hidden units by how much the local NLL changes when each one is removed. The weakest units change it by at most about 0.002 nats. It also found a direction in the head's input that separates the codes: glucose codes score about +2.7 to +7.8, arterial MAP about -6.9 to -3.7, and cuff MAP about -5.4 to +0.7.

**Conclusion:** I could gate a correction by code. Two GELU units with opposite output weights, both driven by that direction, would give roughly a linear readout for glucose codes and nearly nothing for MAP codes. Cuff MAP partly overlaps, so the gate needs care. That only makes sense if a clean dose signal can be read from the state at the moment a reading is predicted.

This conversation restarted, so I don't have my code details in view. Before designing anything, I will reload the architecture description and my head re-implementation (/work/lib.py) and check what is still in the kernel. This run only reads files; it edits nothing.

**What would refute the plan:**
- The architecture shows the event encoder's value embeddings are per-code. Then an input-side fix, such as making NE rates above 1 extrapolate sensibly, becomes possible and may beat a head-only edit.
- The head's input cannot carry a lagged dose signal at all.

```python
import os, re
arch = open('/model/ARCHITECTURE.md').read()
lib = open('/work/lib.py').read()
print(len(arch), len(lib))
print(arch)
print("=================== lib.py ===================")
print(lib)
print("=================== kernel vars ===================")
print(sorted([k for k in globals().keys() if not k.startswith('_')])[:400])
```
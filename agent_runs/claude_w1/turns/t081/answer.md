The last run listed the tensor names inside the model's recurrent core. The core has a prelude, a stack of blocks and a coda. Each part has self-attention ("attn"), cross-attention ("cross"), a feed-forward layer and norms. There is also a small "age" MLP. After the restart I no longer have the details of how these parts connect. That is what I need to decide where an upstream edit should go.

My earlier finding is the basis for what comes next. A head-only edit cannot work well, because the dose innovation is only weakly decodable from the head input (R² about 0.24 for insulin and 0.09 for NE). The edit has to carry a decaying "dose minus policy-expected dose" signal into the state. The h-slots look like the natural carrier, because they decay at learned per-dimension rates (flow.a).

Next step: re-read ARCHITECTURE.md in full and list the helper functions and variables still alive in my kernel. Then I can reuse the exact encoder/z_post replica and the head replica instead of rebuilding them.

How this tests the plan:
- **Confirms it:** the document shows a linear or near-linear path from the per-bundle input (z or u) into the h-slots, with per-dimension decay. Then a dedicated dose-memory channel can be built by weight edits.
- **Refutes it:** h is only reachable through attention and nonlinear blocks. Then I would fall back to approximating the memory from existing state directions.

```python
import re, json, os
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:12000])
# list my own helper objects still alive in the kernel
mine = sorted(k for k,v in globals().items() if not k.startswith('_') and (callable(v) or hasattr(v,'shape')) and k not in ('load','load_original','save_edit','drop_edit','edited'))
print(mine[:200])
```
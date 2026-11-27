Several weaker settings have more factual-loss margin. I’ll test insulin weight 0.25 and norepinephrine weight 0.15: it preserves strong insulin contrasts while improving the norepinephrine doubling contrast over the validated baseline. If GPU factual loss passes, the rollout will show whether these one-step gains persist across hours.

```python
import numpy as np
import torch, dis
from pathlib import Path

chosen270 = (.25, .15)
T270 = grid269[chosen270]["T"]
W0270 = array267(load("attr_heads.value_bin.0.weight")).copy()
b0270 = array267(load("attr_heads.value_bin.0.bias")).copy()
W0270[units267] = arch263["rows"][:, :-1]
b0270[units267] = arch263["rows"][:, -1]
for n270, a270 in [
    ("attr_heads.value_bin.0.weight", W0270),
    ("attr_heads.value_bin.0.bias", b0270),
    ("attr_heads.value_bin.2.weight", T270[:-1].T),
    ("attr_heads.value_bin.2.bias", T270[-1]),
]:
    save_edit(n270, torch.tensor(np.ascontiguousarray(a270), dtype=torch.float32))
print("Staged", chosen270, "offline score", grid269[chosen270]["score"])

print("=== RAW FEATURE OBJECTS ===")
for n270 in ["Xraw204", "X225", "Hf214", "Fquiet234", "headbase214",
             "gatecanon200", "family204", "directions207", "pruneorder214"]:
    o270 = globals()[n270]
    print(n270, brief266(o270))
print("=== PAIR4 CONSTRUCTION ===")
print("\n".join(
    f"{i.offset:4} {i.opname:22} {i.argrepr}"
    for i in dis.Bytecode(make_arch215)
)[-5400:])
print("=== ORIGINAL/TRANSITION LOGIT OBJECTS ===")
for n270 in ["Lorig179", "Loriginal225", "offset204", "offset234"]:
    print(n270, tuple(globals()[n270].shape))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 270 / weaker broadened head staged\n"
            "Staged new four-unit decoder rows with grid269 (.25,.15) final head. "
            "Offline factual deltas +.000705/+.003921/+.003538; slopes "
            "[-1.826,-1.889,+2.559,+1.723]. Quiet wire unchanged. "
            "backup267 preserves validated old .5/.4; GPU audit pending.\n")
```
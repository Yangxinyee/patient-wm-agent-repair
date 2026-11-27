Only four hidden neurons carry the correction, so the failure may be fixable without changing the recurrent model. The cached factual check covers roughly 100 stays and underestimated the new cohort’s loss. I’ll inspect the gates and retained dose-response estimates. If the gates suppress lower-dose states or activate on unsupported factual states, that would explain the asymmetric rollout response and uneven factual fit.

```python
from pathlib import Path
import numpy as np, inspect
from scipy.special import ndtr

text187 = Path("/work/notes.md").read_text()
print("NOTEBOOK HEADINGS\n", "\n".join(
    line for line in text187.splitlines() if line.startswith("## ")
)[:2400])
p187 = text187.find("## Run 5\n")
print("\nPERSISTENCE EVIDENCE\n", text187[p187+3600:p187+7200])

print("\nFUNCTION ARGUMENTS")
for k, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        co = v.__code__
        print(k, co.co_varnames[:co.co_argcount])

print("\nRECENT ARRAY SHAPES")
for k in list(globals()):
    if k.startswith(("r175_", "r176_", "r177_", "r178_")):
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:100])

X187 = np.asarray(r177_X)
ev187 = np.asarray(r170_event)
W187 = np186(load("attr_heads.value_bin.0.weight"))
b187 = np186(load("attr_heads.value_bin.0.bias"))
for pair187 in ([107,109], [48,171]):
    common187 = X187 @ ((W187[pair187[0]] + W187[pair187[1]]) / 2)
    common187 += (b187[pair187[0]] + b187[pair187[1]]) / 2
    contrast187 = X187 @ ((W187[pair187[0]] - W187[pair187[1]]) / 2)
    contrast187 += (b187[pair187[0]] - b187[pair187[1]]) / 2
    effect187 = ((common187+contrast187)*ndtr(common187+contrast187)
                 -(common187-contrast187)*ndtr(common187-contrast187))
    print("\nPAIR", pair187)
    for e187 in np.unique(ev187):
        m187 = ev187 == e187
        print(e187, "n", int(m187.sum()),
              "common q", np.round(np.quantile(common187[:,m187], [.01,.1,.5,.9,.99]),3),
              "contrast q", np.round(np.quantile(contrast187[:,m187], [.01,.5,.99]),3),
              "activation q", np.round(np.quantile(effect187[:,m187], [.01,.5,.99]),3))
    print("NE probe common/contrast slopes",
          float(((W187[pair187[0]]+W187[pair187[1]])/2) @ r177_probe_direction),
          float(((W187[pair187[0]]-W187[pair187[1]])/2) @ r177_probe_direction))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 187: gate and persistence audit\nNo weights changed. Auditing paired-neuron common gates and dose contrasts; previous cached factual checks are insufficient for acceptance after Forward 26.\n")
```
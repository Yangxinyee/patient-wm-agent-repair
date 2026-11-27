The factual cost is concentrated in glucose; MAP fit is essentially unchanged. I’ll check the decoder’s probabilities and recover the insulin targets before attempting a calibration. Large reconstruction errors would rule out relying on local probability corrections without additional GPU validation.

```python
import torch, numpy as np, pandas as pd, inspect, re
from pathlib import Path

print("=== LOCAL DISTRIBUTION CHECK ===")
x160 = state_array46
r160 = readout125(x160[:,192:960].reshape(-1,4,192), x160[:,:192])
print("Returned:", [(type(v).__name__, getattr(v,"shape",None))
                    for v in r160] if isinstance(r160,(tuple,list)) else type(r160).__name__)
a160 = extract_readout143(r160)
print("Extracted:", [(type(v).__name__, getattr(v,"shape",None))
                     for v in a160] if isinstance(a160,(tuple,list)) else repr(a160)[:500])
for name160 in ["readout125","head_calc"]:
    fun160 = globals()[name160]
    print(name160, "constants:", fun160.__code__.co_consts)
print("=== VOCAB STRUCTURE ===")
print("Keys:", list(VOCAB))
for k160 in ["codes","config"]:
    print(k160, VOCAB.get(k160))
print("=== TRAINING IDS ===")
train_ids160 = sorted(panel.stay_id.unique().astype(int).tolist())
print(train_ids160[:100])
print("=== INSULIN RECORD TARGETS ===")
sections160 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
target_sections160 = [s for s in sections160
    if re.search(r"(insulin.*target|target.*insulin|schedule.*glucose|glucose.*schedule)", s, re.I)]
for s160 in target_sections160[:4]:
    print(s160[:1000])
print("=== CALIBRATION-RELEVANT LIVE ARRAYS ===")
for n160 in sorted(list(globals())):
    if re.search(r"(bins8|targetbin|finebin|observedbin|head_input|with_code|query|hidden0|trans|code8|edge)", n160):
        v160 = globals()[n160]
        if not callable(v160):
            print(n160, type(v160).__name__, getattr(v160,"shape",None))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 160: decoder calibration diagnostics\n"
            "Insulin-only factual cost is concentrated in glucose. Checked original-state distributional "
            "decoder interface and recovered training IDs, vocabulary structure and insulin target evidence. "
            "No weights changed. Python runs: 160; forwards: 21.\n")
```
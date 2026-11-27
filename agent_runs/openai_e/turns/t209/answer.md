The earlier record analysis supports a negative insulin effect and a positive norepinephrine effect, unlike the original model’s norepinephrine response. Those estimates still need their uncertainty and dynamics recovered. The output was truncated before showing the saved edit, so I’ll retrieve only the decisive summaries and tensor details. This will establish whether a smaller joint edit is worth testing.

```python
from pathlib import Path
import re, numpy as np, torch

notes179 = Path("/work/notes.md").read_text()
sections179 = re.split(r"(?=^## )", notes179, flags=re.M)

print("SELECTED EDIT SUMMARIES")
for wanted179 in [58, 63, 77, 81, 107, 148]:
    matches179 = [s for s in sections179
                  if re.search(r"(?:Run |run )" + str(wanted179) + r"\b", s.split("\n", 1)[0])]
    for s179 in matches179[:1]:
        print(s179[:1050])

print("\nLATER CAUSAL/TARGET SUMMARIES")
hits179 = [s for s in sections179
           if re.search(r"(causal|identification|impulse|target|persistence)", s, re.I)
           and not re.search(r"(?:Run |run )(?:1[5-7]\d|178)\b", s.split("\n", 1)[0])]
for s179 in hits179[-5:]:
    print(s179[:650])

print("\nARCHIVE KEYS")
for filename179 in ["baseline77.npz", "ne_candidate151.npz", "ne_candidate155.npz"]:
    with np.load("/work/" + filename179) as a179:
        print(filename179, [(k, a179[k].shape) for k in a179.files])

print("\nRELEVANT KERNEL NAMES")
keys179 = [k for k in globals()
           if re.search(r"(target|effect|causal|baseline|Bdir|Binput|fits150|ids|original|Wbase|delta)", k, re.I)]
print(", ".join(sorted(keys179)))
for k179 in ["Bdir141", "Binput142", "fits150", "ids", "ids100", "train_ids", "tuning_ids"]:
    if k179 in globals():
        o179 = globals()[k179]
        if hasattr(o179, "shape"):
            print(k179, tuple(o179.shape))
        elif isinstance(o179, dict):
            print(k179, "keys", list(o179.keys()))
        else:
            print(k179, str(o179)[:450])

w179 = load("core.b_in.weight")
wo179 = load_original("core.b_in.weight")
print("\nCORE B_IN", tuple(w179.shape),
      "edit norm", float(torch.linalg.vector_norm(torch.as_tensor(w179-wo179))))
print("VOCAB CONFIG", VOCAB.get("config"))
print("CURRENT EDITS", edited())

with Path("/work/notes.md").open("a") as f179:
    f179.write("\n## Run 179: bounded baseline and causal recovery\n"
               "Run178 output truncated before saved-candidate details. Recovering only decisive "
               "target/edit summaries and archive keys; no weights changed. "
               "Python runs: 179; forwards: 22.\n")
```

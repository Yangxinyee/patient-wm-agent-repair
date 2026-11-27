The earlier probes found little usable dose-residual information in the final prediction features, which supports editing the recurrent action pathway. Candidate63 also changed the feature itself, so its failure does not rule out strengthening candidate58. I’ll test a 1.6× version of that otherwise unchanged edit; it must pass the factual-fit limit before we assess its rollouts.

```python
import numpy as np
from pathlib import Path

weight_name67 = "core.b_in.weight"
original67 = load_original(weight_name67)
current67 = load(weight_name67)
base_delta67 = current67 - original67

def as_numpy67(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

print("Current delta norm:", np.linalg.norm(as_numpy67(base_delta67)))
print("Difference from candidate58:",
      np.linalg.norm(as_numpy67(base_delta67) - as_numpy67(candidate_delta58)))
assert np.allclose(as_numpy67(base_delta67), as_numpy67(candidate_delta58),
                   atol=2e-6, rtol=2e-5), "Unexpected current edit; do not rescale."

strength67 = 1.6
save_edit(weight_name67, original67 + strength67 * base_delta67)
print("Installed candidate58 strength:", strength67)
print("Edited tensors:", edited())
print("Validation IDs:", [int(x) for x in validation_ids])

print("\nRelevant retained helpers:")
for name, typ, desc in inventory65:
    if typ == "function" and any(s in name for s in
                                ("core_", "feature", "prelude", "contrast", "decode", "head_")):
        fn67 = globals().get(name)
        print(name, tuple(fn67.__code__.co_varnames[:fn67.__code__.co_argcount]))

print("\nCausal-data columns:")
print(list(panel.columns))
print("NE nuisance objects:",
      [(name, type(globals().get(name)).__name__)
       for name in ("sne", "ne_nuisance", "dynamic_rd", "dyn_fit", "dose_policy")])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 67: isolated strength test\n"
            "Verified current edit equals candidate58, then installed 1.6 times its original delta. "
            "Unlike candidate63, this changes no feature balancing or output direction. "
            "Candidate is unaccepted pending 100-stay factual validation, then paired rollouts. "
            "Python runs: 67; forwards: 8.\n")
```
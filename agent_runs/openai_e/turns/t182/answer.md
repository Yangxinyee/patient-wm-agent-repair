The shrunken edit still exceeds the limit on this sample: +0.00652 nats. I’ll now test the insulin-only baseline on exactly these stays, while recovering the decoder helpers. If most of the loss is already in that baseline, changing the NE penalty alone cannot solve the factual-fit problem.

```python
import numpy as np
import inspect
from pathlib import Path

baseline_weight156 = (original154 + baseline_delta154).astype(np.float32)
save_edit("core.b_in.weight", baseline_weight156)
print("Installed verified insulin-only baseline77 for isolation test.")
print("Saved-weight error:",
      np.max(np.abs(np.asarray(load("core.b_in.weight")) - baseline_weight156)))

print("\n=== DECODER HELPER INTERFACES ===")
for nm156 in ["head_calc", "decoder_query", "readout125", "extract_readout143"]:
    fn156 = globals().get(nm156)
    if callable(fn156):
        print(nm156, inspect.signature(fn156))
        print("Referenced objects:", fn156.__code__.co_names)
        print("Arguments/local variables:", fn156.__code__.co_varnames[:30])
        try:
            src156 = inspect.getsource(fn156)
            print(src156[:1700])
        except (OSError, TypeError):
            print("Notebook source unavailable through inspect.")

print("\n=== RELEVANT LIVE DATA ===")
shown156 = 0
for nm156 in sorted(list(globals())):
    if re.search(r"(state|register|readout|query|record|panel|probe)", nm156, re.I):
        obj156 = globals()[nm156]
        if hasattr(obj156, "shape"):
            print(nm156, type(obj156).__name__, obj156.shape)
            shown156 += 1
            if shown156 >= 45:
                break

print("\n=== ORIGINAL STATE ARCHIVES ===")
for path156 in sorted(Path("/work").glob("fwd_*/states_original.npz")):
    with np.load(path156) as z156:
        print(path156, {k: z156[k].shape for k in z156.files})

with open("/work/notes.md", "a") as f156:
    f156.write(
        "\n## Forward 20 / Run 156: isolate insulin baseline factual cost\n"
        "Candidate155 NLL=1.64113 vs original=1.63461 (+.00652) on first 100 tuning stays. "
        "Installed insulin-only baseline77 to measure its contribution on this exact sample. "
        "Recovered decoder interfaces and original-state archive layouts for a possible algebraic calibration, "
        "not a gradient-based weight fit. Python runs: 156; forwards: 20.\n"
    )
```
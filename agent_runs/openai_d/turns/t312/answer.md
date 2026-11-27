The corrected screen favors insulin 0.90 / norepinephrine 0.35: its cached NLL increase is 0.00114 on held-out training stays and 0.00245 on the cached tuning cohort. I’ll install it reversibly, including training-only last-reading-bin cancellation. GPU factual tests must confirm the margin, and rollouts must confirm that insulin remains close to target.

```python
import numpy as np, pandas as pd, hashlib
from pathlib import Path

n264_key = (.90,.35,4.)
n264_head,n264_table = n262_candidates[n264_key]
n264_weights = {k:np.asarray(v,dtype=np.float32).copy()
                for k,v in n264_head.items()}
n264_transition_name = "embed_transition.value_bin.weight"
n264_transition = np.asarray(load_original(n264_transition_name),
                              dtype=np.float32).copy()
n264_codes = VOCAB["codes"]
print("vocabulary codes:",n264_codes)

def n264_code(event):
    if isinstance(n264_codes,dict):
        if event in n264_codes:
            val = n264_codes[event]
            if isinstance(val,(int,np.integer)):
                return int(val)
            if isinstance(val,dict):
                return int(val["id"])
        for key,val in n264_codes.items():
            if val == event:
                return int(key)
    elif isinstance(n264_codes,list):
        return n264_codes.index(event)
    raise ValueError(f"Cannot resolve event {event!r}")

for (event,lastbin),v in n264_table.items():
    row = n264_code(event)*17+int(lastbin)
    n264_transition[row] -= 4.*np.asarray(v,dtype=np.float32)
n264_weights[n264_transition_name] = n264_transition

for nm,c in [("train",c210_train),("tune",c210_tune)]:
    ids = np.array([n264_code(e)*17+int(b)
                    for e,b in zip(c["meta"]["event"],c["meta"]["last_bin"])])
    actual = n264_transition[ids]-np.asarray(load_original(n264_transition_name))[ids]
    intended = -4.*r252_table_rows(n264_table,c)
    print(nm,"max transition reconstruction error:",
          float(np.max(np.abs(actual-intended))))
    assert np.max(np.abs(actual-intended)) < 2e-6

n264_backup = {k:np.asarray(load(k),dtype=np.float32).copy() for k in edited()}
np.savez("/work/rejected_8898bdc457.npz",**n264_backup)
assert all(tuple(v.shape)==tuple(SHAPES[k]) for k,v in n264_weights.items())
assert all(np.isfinite(v).all() for v in n264_weights.values())
for name,value in n264_weights.items():
    save_edit(name,value)
np.savez("/work/candidate90_35_bin80x4.npz",**n264_weights)

n264_hash = hashlib.sha256(
    b"".join(k.encode()+np.asarray(load(k),dtype=np.float32).tobytes()
             for k in sorted(edited()))).hexdigest()[:12]
print("installed",n264_key,"hash",n264_hash)
print(n262_screen[(n262_screen["ins"]==.90)&
                  (n262_screen["ne"]==.35)&
                  (n262_screen["cancel_scale"]==4.)].to_string(index=False))
with open("/work/notes.md","a") as f:
    f.write(f"\n## Run 264: reversible reduced-strength candidate\n"
            f"Installed insulin .90 / NE .35 with shrink80-scale4 training-only "
            f"previous-bin cancellation; hash {n264_hash}. Cached deltas: "
            "training fit .002196, held-out training .001139, tuning100 .002454. "
            "Saved exact tensors to /work/candidate90_35_bin80x4.npz and the "
            "previous rejected checkpoint to /work/rejected_8898bdc457.npz. "
            "Candidate not accepted until GPU factual and rollout validation.\n")
```

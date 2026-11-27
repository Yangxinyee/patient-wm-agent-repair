Installation stopped before saving any weights because the vocabulary uses a list of code descriptors, not a list of names. I’ll fix that lookup and repeat the verified installation. This is a bookkeeping correction; the statistical findings are unchanged.

```python
import numpy as np, hashlib

n265_code_map = {item["name"]:int(item["id"]) for item in VOCAB["codes"]}
def n264_code(event):
    return n265_code_map[event]

n264_transition = np.asarray(
    load_original("embed_transition.value_bin.weight"),dtype=np.float32).copy()
for (event,lastbin),v in n264_table.items():
    n264_transition[n264_code(event)*17+int(lastbin)] -= (
        4.*np.asarray(v,dtype=np.float32))
n264_weights = {k:np.asarray(v,dtype=np.float32).copy()
                for k,v in n264_head.items()}
n264_weights["embed_transition.value_bin.weight"] = n264_transition

for label,c in [("train",c210_train),("tune",c210_tune)]:
    indices = np.array([n264_code(e)*17+int(b)
                        for e,b in zip(c["meta"]["event"],c["meta"]["last_bin"])])
    actual = n264_transition[indices]-np.asarray(
        load_original("embed_transition.value_bin.weight"))[indices]
    expected = -4.*r252_table_rows(n264_table,c)
    error = float(np.max(np.abs(actual-expected)))
    print(label,"transition verification",error)
    assert error < 2e-6

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
print("installed hash",n264_hash,"tensors",edited())
print(n262_screen[(n262_screen["ins"]==.90)&
                  (n262_screen["ne"]==.35)&
                  (n262_screen["cancel_scale"]==4.)].to_string(index=False))

with open("/work/notes.md","a") as f:
    f.write(f"\n## Runs 264–265: reduced-strength candidate installed\n"
            "Run264 stopped on vocabulary lookup before changing weights. "
            f"Run265 resolved descriptor-list codes and verified transition "
            f"corrections before installing insulin .90 / NE .35 with "
            f"shrink80-scale4 cancellation; hash {n264_hash}. "
            "Cached deltas: training fit .002196, held-out training .001139, "
            "tuning100 .002454. Exact candidate saved to "
            "/work/candidate90_35_bin80x4.npz. Prior rejected weights saved to "
            "/work/rejected_8898bdc457.npz. Awaiting GPU validation.\n")
```

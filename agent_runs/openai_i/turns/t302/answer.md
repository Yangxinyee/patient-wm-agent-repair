The GPU indexing is correct. The private channels depend heavily on the action input, so a nonzero encoding of an empty action bundle could explain the failure. I’ll test that next; reproducing the GPU’s untreated values with an empty-bundle encoding would confirm it.

```python
import inspect, numpy as np, torch

print("=== EMPTY ACTION ENCODING ===")
empty267 = {}
for zero in (False, True):
    try:
        v = pool137([], "encode_action", zero_empty=zero)
        empty267[zero] = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
        print("zero_empty", zero, "shape", tuple(v.shape),
              "norm", float(empty267[zero].norm()),
              "first values", empty267[zero][:8].tolist())
    except Exception as ex:
        print("zero_empty", zero, type(ex).__name__, str(ex))
print("=== ALIGNMENT METADATA ===")
for name in ("stays156", "hours156", "state_cal161", "e_cal161", "u_cal161",
             "input196", "context"):
    v = globals().get(name)
    print(name, getattr(v, "shape", None),
          inspect.signature(v) if callable(v) else "")
print("=== EMPTY-ENCODING POSTERIOR TEST ===")
for zero, v in empty267.items():
    testx = xx266.clone()
    testx[inactive266, 384:] = v
    testz = torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(
            testx, torch.from_numpy(p0w266), torch.from_numpy(p0b266))),
        torch.from_numpy(p2w266), torch.from_numpy(p2b266))[:, :32]
    print("zero_empty", zero, "inactive private mean",
          testz[inactive266][:, [16, 12]].mean(0).tolist(),
          "RMS", testz[inactive266][:, [16, 12]].square().mean(0).sqrt().tolist())
    if len(stays156) == len(testx):
        lookup267 = {(int(s), round(float(h), 4)): i
                     for i, (s, h) in enumerate(zip(stays156, hours156))}
        go = gpu265["original"]
        ge = gpu265["edited"]
        matches = [(j, lookup267[(int(s), round(float(h), 4))])
                   for j, (s, h) in enumerate(zip(go["stay_id"], go["hour"]))
                   if (int(s), round(float(h), 4)) in lookup267
                   and not (go["has_insulin"][j] or go["has_norepinephrine"][j])]
        if matches:
            jj, ii = map(np.array, zip(*matches))
            pred = testz[ii][:, [16, 12]].numpy()
            observed = ge["x"][jj][:, 960 + np.array([16, 12])]
            print("matches", len(jj), "private RMSE",
                  np.sqrt(np.mean((pred - observed)**2, axis=0)),
                  "first predicted/observed", pred[:6], observed[:6])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 267 — test empty-action encoder discrepancy\n"
            "- GPU private-state indexing is correct.\n"
            "- Testing whether the forward implementation supplies an encoded empty set rather than literal zero u.\n"
            "- Neutral209 remains installed; no new weights changed.\n")
```
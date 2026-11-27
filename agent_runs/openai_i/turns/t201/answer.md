Doubling insulin changes its action encoding substantially; doubling norepinephrine produces a much smaller contrast. That may help explain the weak norepinephrine response, but it does not yet identify a repair. The run then stopped on an unsupported Torch function. I’ll fix that indexing call and check whether the encoding contrasts survive into the readout state.

```python
active180 = np.flatnonzero((u_cal161.square().sum(1) > 1e-8).numpy())
active180 = [int(i) for i in active180
             if previous[int(i)] >= 0 and 6 <= hours156[int(i)] <= 23.2]
probe180 = active180[0]
args180 = replay_inputs161(probe180, include_current=False, age_scale=1.)

print("=== REPLAY INPUT CONTRACT ===")
print("index/stay/hour:", probe180, int(stays156[probe180]), float(hours156[probe180]))
print("args type/length:", type(args180).__name__, len(args180))
for j, item in enumerate(args180):
    print(j, type(item).__name__,
          tuple(item.shape) if hasattr(item, "shape") else repr(item)[:120])

result180 = call_core161(args180, "key")
print("\n=== CORE OUTPUT CONTRACT ===")
for j, item in enumerate(result180):
    print(j, type(item).__name__, tuple(item.shape))
official_h180 = state_cal161[probe180:probe180+1, 192:960].reshape(1, 4, 192)
official_out180 = state_cal161[probe180:probe180+1, :192]
for j, item in enumerate(result180):
    if item.shape == official_h180.shape:
        print("h replay error/RMS",
              float((item-official_h180).square().mean().sqrt()),
              float(official_h180.square().mean().sqrt()))
    if item.shape == official_out180.shape:
        print("out replay error/RMS",
              float((item-official_out180).square().mean().sqrt()),
              float(official_out180.square().mean().sqrt()))

cap180 = capture159(official_h180, official_out180,
                    int(stays156[probe180]), float(hours156[probe180]), "glucose_finger")
print("\n=== READOUT CAPTURE ===")
print("return type/length:", type(cap180).__name__, len(cap180))
for j, item in enumerate(cap180):
    if isinstance(item, dict):
        print(j, [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in item.items()
                  if k in ["ctx", "view", "q", "with_code", "register", "last_bin", "logits"]])
    else:
        print(j, repr(item))

print("\n=== DRUG NORMALISATION ===")
for drug in ["insulin", "norepinephrine"]:
    for attr in ["rate", "dose"]:
        print(drug, attr, numeric[(drug, attr)])
    rr = records.loc[records.event == drug, "rate"]
    print("observed rate quantiles", rr.quantile([0, .25, .5, .75, 1]).to_dict())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 176–180: direct-head preparation\n"
            "- No active edits. Recovered official states, action caches, and readout helpers.\n"
            "- Doubling insulin changes the action encoder strongly; NE has a much smaller encoding contrast.\n"
            "- Python 179 stopped on torch.flatnonzero; replaced with numpy indexing.\n"
            "- Direct-head design will use paired replay differences anchored to exposed official states.\n")
```

The simpler original-model correction failed: it left 80–96% of the held-out dose-contrast error and required even larger weights. I won’t install it. Next I’ll replay the current repair for one step at each dose scale. Symmetric local responses would implicate rollout feedback; local asymmetry would point to the dose-to-state mapping itself.

```python
print("PRIOR LOCAL-RESPONSE FINDINGS")
for line in lines275:
    if re.search(r"Python run (21[3-9]|22[0-3])\b", line):
        print(line[:370])

print("\nCURRENT REPAIR: ONE-STEP PAIRED DOSE RESPONSES")
current_base280 = parts254(tokens259)
saved_r233_280 = r233_raw
local280 = {}
try:
    r233_raw = current_base280[2][indices278].clone()
    local_base280 = np.asarray(response_raw234(np.zeros((1,192),dtype=np.float32)))[0]
    print("Baseline; sensor order cuff/finger/arterial/lab:", local_base280)
    print("Calibration mean doses:", rates279[indices278].mean(axis=0))
    for key, tokens in fresh_tokens262.items():
        p = parts254(tokens)
        r233_raw = p[2][indices278].clone()
        shift = np.asarray(response_raw234(np.zeros((1,192),dtype=np.float32)))[0]-local_base280
        drug, scale = key
        expected = np.asarray(target255)[:,drug]*rates279[indices278,drug].mean()*(scale-1)
        local280[key] = (shift, expected)
        print(key, "model shift",np.round(shift,4),
              "record-based immediate reference",np.round(expected,4))
finally:
    r233_raw = saved_r233_280

print("\nREPLAY INTERFACES")
for name in ["state_context", "context_with_flow", "encode_groups80",
             "encode_bundles", "view171", "phys171"]:
    f = globals()[name]
    print(name, inspect.signature(f), f.__code__.co_names[:18])

print("\nORIGINAL TRAIN STATE IDENTIFIERS")
statefile280 = np.load("/work/fwd_3/states_original.npz")
print("Keys:", statefile280.files)
train_state_ids280 = np.unique(statefile280["stay_id"])
print("Count:",len(train_state_ids280),"first/last:",train_state_ids280[:5],train_state_ids280[-5:])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 280; forwards 24. Rejected original-prelude-only regression without installation: held-out dose contrast errors 0.79-0.96 and excessive weight norms. Replayed current repaired factual and all x0/.5/2/3 dose cases for one step to distinguish local response error from rollout feedback. Results: "+str(local280)+". Validated checkpoint273 remains installed.\n")
```
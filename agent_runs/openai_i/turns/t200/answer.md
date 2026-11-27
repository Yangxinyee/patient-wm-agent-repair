The saved states cover 100 training stays and include the action encodings needed for paired dose perturbations. I’ll check that changing a rate actually changes those encodings, then inspect the replay inputs. The next test will use differences between two replays, anchored to the exposed state; unstable or negligible differences would argue against a direct head repair.

```python
torch.set_num_threads(8)
W = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES}

print("=== CAPTURE CONTRACT ===")
for event, value in captured174.items():
    print(event, "tuple length", len(value))
    for j, item in enumerate(value):
        if isinstance(item, dict):
            print(j, "dict", [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                              for k, v in item.items()])
        elif hasattr(item, "shape"):
            print(j, type(item).__name__, tuple(item.shape))
        else:
            print(j, type(item).__name__, repr(item)[:100])

print("\n=== ACTION ENCODER CHECK ===")
U179 = {}
for label, scales in [("base", (1., 1.)), ("insulin2", (2., 1.)),
                      ("ne2", (1., 2.))]:
    U179[label] = torch.as_tensor(encode_actions168(scales), dtype=torch.float32)
    print(label, tuple(U179[label].shape),
          "RMS", float(U179[label].square().mean().sqrt()))
if U179["base"].shape == u_cal161.shape:
    print("base versus cached U RMS",
          float((U179["base"] - u_cal161).square().mean().sqrt()))
    for label in ["insulin2", "ne2"]:
        diff = U179[label] - U179["base"]
        print(label, "changed bundles", int((diff.square().sum(1) > 1e-10).sum()),
              "contrast RMS", float(diff.square().mean().sqrt()))

print("\n=== GROUP EXAMPLES ===")
print("groups:", act_groups152[:5])
print("rows:", [dict(r) for r in act_rows152[:3]])

active179 = torch.flatnonzero(u_cal161.square().sum(1) > 1e-8).numpy()
active179 = [int(i) for i in active179
             if previous[int(i)] >= 0 and 6 <= hours156[int(i)] <= 23.2]
probe179 = active179[0]
args179 = replay_inputs161(probe179, include_current=False, age_scale=1.)
print("\n=== REPLAY INPUT CONTRACT ===")
print("index/stay/hour:", probe179, int(stays156[probe179]), float(hours156[probe179]))
print("args type/length:", type(args179).__name__, len(args179))
for j, item in enumerate(args179):
    print(j, type(item).__name__,
          tuple(item.shape) if hasattr(item, "shape") else repr(item)[:120])
result179 = call_core161(args179, "key")
print("core result:",
      [(type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else None)
       for x in result179])
print("\n=== ACTIVE EDITS ===", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 176–179: direct-head preparation\n"
            "- All edits remain reverted. Recovered official recurrent states and action caches.\n"
            "- Record estimates still support immediate insulin about -2 and NE about +3 per rate unit;\n"
            "  controlled lag estimates suggest substantially larger accumulated rollout effects.\n"
            "- Testing action-perturbed replay differences anchored to official states, not treating\n"
            "  imperfect absolute reconstruction as factual certification.\n")
```

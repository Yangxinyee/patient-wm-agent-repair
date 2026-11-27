The posterior decoder failed: it predicts poorly on held-out stays and responds inconsistently to dose multipliers. I will not use it. Instead, I’ll temporarily replace the action encoder’s output with its empty-action vector to measure how much factual fit depends on that pathway. A small NLL change would leave room for a better dose representation; a large change would rule out wholesale replacement.

```python
print("--- Activation implementations ---")
for fname in ["act0", "wm_mlp", "norm0"]:
    fun = globals()[fname]
    print(fname, "; ".join(
        f"{i.opname} {i.argrepr}" for i in dis.get_instructions(fun)
        if i.opname not in {"RESUME", "PRECALL", "CACHE", "PUSH_NULL"}
    )[:1800])

empty_from_wm = wm_set(torch.zeros(0, 192), "encode_action").reshape(-1)
empty_from_batch = U4[np.flatnonzero(counts_start4 == 0)[0]].reshape(-1)
empty_verified = empty_action_local.reshape(-1)
print("Empty-action vector differences:",
      float(torch.max(torch.abs(empty_from_wm - empty_verified))),
      float(torch.max(torch.abs(empty_from_batch - empty_verified))))

for name in list(edited()):
    drop_edit(name)
save_edit("encode_action.norm.weight",
          torch.zeros_like(load_original("encode_action.norm.weight")))
save_edit("encode_action.norm.bias", empty_verified)
print("Temporary ablation edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 45–48: dose representation and failed decoder\n"
            "- Dose encoder varies sharply and non-monotonically with rate; infusion END has no "
            "dose attributes and goes through the measurement encoder.\n"
            "- Ridge decoding of treatment innovations from posterior hidden features, including "
            "synthetic dose multipliers, failed held-out factual and dose-response tests. Do not use it.\n"
            "- Temporary diagnostic: action-encoder norm output fixed to the verified empty-action "
            "vector. Test paired factual NLL before deciding whether replacement is feasible.\n")
```

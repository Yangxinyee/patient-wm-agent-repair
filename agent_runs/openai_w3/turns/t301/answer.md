The joint fit reduces the held-out dose-contrast error, though the current insulin update was already roughly linear in raw-state space. That means raw-state linearity alone may not explain the rollout asymmetry. I’ll test the gentlest correction first; it must pass the factual limit before I retain it.

```python
import hashlib, json, torch
chosen268 = 0.1
save_edit("core.prelude.0.ffn.2.weight", torch.from_numpy(candidates267[chosen268]["weight"]))
save_edit("core.prelude.0.ffn.2.bias", torch.from_numpy(candidates267[chosen268]["bias"]))
hash268 = hashlib.sha256(b"".join(
    load(n).detach().cpu().numpy().tobytes() for n in sorted(edited())
)).hexdigest()[:16]
print("Installed candidate268, gamma",chosen268,"hash",hash268)
print("Rollback checkpoint267 contains",len(checkpoint267),"tensors")
print(json.dumps({"type":"factual","model":"both","split":"tuning",
                  "stays":list(range(2020000000,2020000100))}))
with open("/work/notes.md","a") as f:
    f.write(f"\nPython run 268; forwards 20. Installed reversible total-response candidate gamma=.1, hash {hash268}; checkpoint267 retains validated checkpoint222. Held-out normalized-state RMS .03918 and contrast relative error .3225. Next factual test; accept only original value NLL +<=.005 nats.\n")
```